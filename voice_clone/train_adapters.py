"""Train SegmentGST + Kokoro L-adapters + SLM discriminator (Kokoro and WavLM frozen)."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from kokoro.model import KModel

from .adapters import AdapterRegistry
from .config import LossWeights, MelLossConfig, TrainConfig, kokoro_vocab_and_context_length, load_kokoro_config
from .dataset import VoiceCloneManifestDataset, collate_voice_clone_batch
from .losses import (
    MelReconstructionLoss,
    SLMFeatureDiscriminator,
    slm_discriminator_loss_hinge,
    slm_generator_loss_hinge,
    speaker_cosine_loss,
)
from .segment_gst import SegmentGST
from .wavlm_sv import WavLMSV


def _import_wandb():
    """Import wandb only when ``--wandb`` is used so inference-only installs can skip it."""
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb is required when using --wandb. Install training dependencies, e.g. "
            "pip install -r requirements-training.txt"
        ) from e
    return wandb


def _cuda_amp_context(use_amp: bool, reference_tensor: torch.Tensor):
    if use_amp and reference_tensor.is_cuda:
        return autocast("cuda", enabled=True)
    return contextlib.nullcontext()


def _configure_training_warnings() -> None:
    """Drop known-noisy PyTorch messages that are not actionable during adapter training."""
    warnings.filterwarnings(
        "ignore",
        message=r"Support for mismatched key_padding_mask and attn_mask is deprecated\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Mem Efficient attention on Current AMD GPU is still experimental\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Flash Efficient attention on Current AMD GPU is still experimental\..*",
        category=UserWarning,
    )


def _terminal_metrics_cr_line(global_step: int, metrics: Dict[str, float]) -> str:
    """Single-line status padded for carriage-return overwrite (no tqdm)."""
    parts = [f"step {global_step}"] + [f"{k}={v:.4f}" for k, v in metrics.items()]
    line = " ".join(parts)
    try:
        width = max(40, shutil.get_terminal_size(fallback=(100, 24)).columns - 1)
    except OSError:
        width = 100
    if len(line) < width:
        line = line + " " * (width - len(line))
    return line[:width]


def freeze_kokoro_except_adapters(kmodel: KModel) -> None:
    kmodel.train()
    for p in kmodel.parameters():
        p.requires_grad_(False)
    enc = kmodel.predictor.text_encoder
    if enc.adapters is not None:
        for p in enc.adapters.parameters():
            p.requires_grad_(True)
    if kmodel.decoder.decoder_adapters is not None:
        for p in kmodel.decoder.decoder_adapters.parameters():
            p.requires_grad_(True)
    if kmodel.decoder.generator.adapters is not None:
        for p in kmodel.decoder.generator.adapters.parameters():
            p.requires_grad_(True)


def generator_trainable_parameters(kmodel: KModel, gst: SegmentGST) -> List[nn.Parameter]:
    return list(gst.parameters()) + [p for p in kmodel.parameters() if p.requires_grad]


def _ensure_batch_time(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    if wav.dim() != 2:
        raise ValueError(f"waveform must be (time,) or (batch, time), got {tuple(wav.shape)}")
    return wav


def resample_mono(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Differentiable resample; ``wav`` is ``(batch, time)``."""
    import torchaudio

    w = _ensure_batch_time(wav)
    if orig_sr == new_sr:
        return w
    # torchaudio resample expects (batch, channels, time)
    x = w.unsqueeze(1)
    y = torchaudio.functional.resample(x, orig_sr, new_sr)
    return y.squeeze(1)


def build_mel_loss(kokoro_cfg: Dict[str, Any], mel_cfg: MelLossConfig, device: torch.device) -> MelReconstructionLoss:
    n_mels = int(kokoro_cfg["n_mels"])
    m = MelReconstructionLoss(
        sample_rate=mel_cfg.sample_rate,
        n_mels=n_mels,
        n_fft=mel_cfg.n_fft,
        hop_length=mel_cfg.hop_length,
        win_length=mel_cfg.win_length,
        f_min=mel_cfg.f_min,
        f_max=mel_cfg.f_max,
    )
    return m.to(device)


def build_models(
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[KModel, SegmentGST, WavLMSV, SLMFeatureDiscriminator, MelReconstructionLoss, Dict[str, Any]]:
    kokoro_cfg = load_kokoro_config(cfg.kokoro_repo_id)
    istft = kokoro_cfg["istftnet"]
    registry = AdapterRegistry.from_dims(
        d_model=int(kokoro_cfg["hidden_dim"]),
        z_style_dim=256,
        duration_nlayers=int(kokoro_cfg["n_layer"]),
        adapter_bottleneck=cfg.adapter_bottleneck,
        upsample_initial_channel=int(istft["upsample_initial_channel"]),
        num_upsamples=len(istft["upsample_rates"]),
    )
    kmodel = KModel(
        repo_id=cfg.kokoro_repo_id,
        config=kokoro_cfg,
        duration_encoder_adapters=registry.duration_encoder,
        decoder_adapters=registry.decoder,
        generator_adapters=registry.generator,
    )
    kmodel = kmodel.to(device)
    freeze_kokoro_except_adapters(kmodel)

    gst = SegmentGST(frame_dim=768, embed_dim=256, num_heads=8, ref_dim=256, style_dec_dim=128)
    gst = gst.to(device)

    wavlm = WavLMSV(cfg.wavlm_model_id, device=device, dtype=None)

    slm_cfg = cfg.slm_disc
    disc = SLMFeatureDiscriminator(
        in_dim=768,
        hidden_channels=slm_cfg.hidden_channels,
        num_layers=slm_cfg.num_layers,
        kernel_size=slm_cfg.kernel_size,
    ).to(device)

    mel_loss = build_mel_loss(kokoro_cfg, cfg.mel, device)
    return kmodel, gst, wavlm, disc, mel_loss, kokoro_cfg


def save_checkpoint(
    path: Path,
    *,
    gst: SegmentGST,
    disc: SLMFeatureDiscriminator,
    kmodel: KModel,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "step": step,
        "train_config": asdict(cfg),
        "segment_gst": gst.state_dict(),
        "slm_discriminator": disc.state_dict(),
        "duration_adapters": kmodel.predictor.text_encoder.adapters.state_dict()
        if kmodel.predictor.text_encoder.adapters is not None
        else None,
        "decoder_adapters": kmodel.decoder.decoder_adapters.state_dict()
        if kmodel.decoder.decoder_adapters is not None
        else None,
        "generator_adapters": kmodel.decoder.generator.adapters.state_dict()
        if kmodel.decoder.generator.adapters is not None
        else None,
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    gst: SegmentGST,
    disc: SLMFeatureDiscriminator,
    kmodel: KModel,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gst.load_state_dict(ckpt["segment_gst"])
    disc.load_state_dict(ckpt["slm_discriminator"])
    if ckpt.get("duration_adapters") and kmodel.predictor.text_encoder.adapters is not None:
        kmodel.predictor.text_encoder.adapters.load_state_dict(ckpt["duration_adapters"])
    if ckpt.get("decoder_adapters") and kmodel.decoder.decoder_adapters is not None:
        kmodel.decoder.decoder_adapters.load_state_dict(ckpt["decoder_adapters"])
    if ckpt.get("generator_adapters") and kmodel.decoder.generator.adapters is not None:
        kmodel.decoder.generator.adapters.load_state_dict(ckpt["generator_adapters"])
    if ckpt.get("optimizer_g"):
        opt_g.load_state_dict(ckpt["optimizer_g"])
    if ckpt.get("optimizer_d"):
        opt_d.load_state_dict(ckpt["optimizer_d"])
    return int(ckpt.get("step", 0))


def slm_discriminator_step(
    disc: SLMFeatureDiscriminator,
    wavlm: WavLMSV,
    opt_d: torch.optim.Optimizer,
    pred_wav_24k: torch.Tensor,
    target_wav_24k: torch.Tensor,
    *,
    grad_clip: Optional[float],
    scaler: Optional[GradScaler],
    use_amp: bool,
) -> torch.Tensor:
    """Train **D** on frozen WavLM features of real (target) vs fake (generated) 16 kHz audio."""
    pred_wav_24k = pred_wav_24k.detach()
    fake_w = resample_mono(pred_wav_24k, 24_000, 16_000)
    real_w = resample_mono(target_wav_24k.detach(), 24_000, 16_000)

    fake_feats, fake_mask, _ = wavlm.frame_hidden_states(
        fake_w, sampling_rate=16_000, grad_through_input=False
    )
    real_feats, real_mask, _ = wavlm.frame_hidden_states(
        real_w, sampling_rate=16_000, grad_through_input=False
    )

    opt_d.zero_grad(set_to_none=True)
    with _cuda_amp_context(use_amp, pred_wav_24k):
        loss_d = slm_discriminator_loss_hinge(disc, real_feats, fake_feats, real_mask, fake_mask)

    if scaler is not None and use_amp:
        scaler.scale(loss_d).backward()
        scaler.unscale_(opt_d)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
        scaler.step(opt_d)
        scaler.update()
    else:
        loss_d.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
        opt_d.step()
    return loss_d.detach()


def training_step_generator(
    kmodel: KModel,
    gst: SegmentGST,
    wavlm: WavLMSV,
    disc: SLMFeatureDiscriminator,
    mel_loss_mod: MelReconstructionLoss,
    opt_g: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    weights: LossWeights,
    *,
    speed: float,
    grad_clip: Optional[float],
    scaler: Optional[GradScaler],
    use_amp: bool,
) -> Dict[str, float]:
    """Single **G** step: mel + speaker cosine + SLM (D frozen); Kokoro forward is differentiable."""
    ref_16 = _ensure_batch_time(batch["ref_wav_16k"].to(kmodel.device))
    tgt_24 = _ensure_batch_time(batch["target_wav_24k"].to(kmodel.device))
    input_ids = batch["input_ids"].to(kmodel.device)
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be (batch, seq_len)")

    opt_g.zero_grad(set_to_none=True)
    with _cuda_amp_context(use_amp, ref_16):
        wv_ref = wavlm(ref_16, sampling_rate=16_000, grad_through_input=False)
        gst_out, _ = gst(wv_ref.frame_hidden_states, wv_ref.frame_mask)
        ref_s = gst_out.ref_s
        pred_wav, _ = kmodel.forward_with_tokens(input_ids, ref_s, speed=speed)
        pred_wav = _ensure_batch_time(pred_wav)

        out_mel = mel_loss_mod(pred_wav, tgt_24)
        l_mel = out_mel.loss

        w_gen_16 = resample_mono(pred_wav, 24_000, 16_000)
        emb_gen = wavlm.pooled_embedding(w_gen_16, sampling_rate=16_000, grad_through_input=True)
        emb_ref = wavlm.pooled_embedding(ref_16, sampling_rate=16_000, grad_through_input=False).detach()
        l_spk = speaker_cosine_loss(emb_ref, emb_gen)

        f_fake, m_fake, _ = wavlm.frame_hidden_states(
            w_gen_16, sampling_rate=16_000, grad_through_input=True
        )
        for p in disc.parameters():
            p.requires_grad_(False)
        l_slm = slm_generator_loss_hinge(disc, f_fake, m_fake)
        for p in disc.parameters():
            p.requires_grad_(True)

        loss_g = weights.lambda_mel * l_mel + weights.lambda_spk * l_spk + weights.lambda_slm * l_slm

    if scaler is not None and use_amp:
        scaler.scale(loss_g).backward()
        scaler.unscale_(opt_g)
        if grad_clip is not None:
            params = generator_trainable_parameters(kmodel, gst)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(opt_g)
        scaler.update()
    else:
        loss_g.backward()
        if grad_clip is not None:
            params = generator_trainable_parameters(kmodel, gst)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt_g.step()

    return {
        "loss_g": float(loss_g.detach()),
        "loss_mel": float(l_mel.detach()),
        "loss_spk": float(l_spk.detach()),
        "loss_slm_g": float(l_slm.detach()),
    }


def _kokoro_id_to_char_map(vocab: Dict[str, int]) -> Dict[int, str]:
    """Invert Kokoro vocab for captions; ids with multiple graphemes are omitted."""
    id_to_chars: Dict[int, List[str]] = {}
    for ch, tid in vocab.items():
        id_to_chars.setdefault(int(tid), []).append(ch)
    return {tid: chars[0] for tid, chars in id_to_chars.items() if len(chars) == 1}


def _caption_from_input_ids(input_ids: torch.Tensor, id_to_char: Dict[int, str], *, max_chars: int = 512) -> str:
    out: List[str] = []
    for tid in input_ids.flatten().tolist():
        ch = id_to_char.get(int(tid))
        if ch is not None:
            out.append(ch)
        if sum(len(c) for c in out) >= max_chars:
            break
    s = "".join(out)
    return s[:max_chars]


def _log_wandb_epoch_audio_table(
    wandb_run: Any,
    *,
    dataset: Dataset,
    kmodel: KModel,
    gst: SegmentGST,
    wavlm: WavLMSV,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    num_samples: int,
    vocab: Dict[str, int],
) -> None:
    """Log a small wandb Table of pred vs target 24 kHz audio at epoch end (eval forward, no grad)."""
    wb = _import_wandb()
    n = min(int(num_samples), len(dataset))
    if n <= 0:
        return

    id_to_char = _kokoro_id_to_char_map(vocab)
    prev_k_train = kmodel.training
    prev_g_train = gst.training
    kmodel.eval()
    gst.eval()
    try:
        columns = ["idx", "caption", "pred_audio", "target_audio"]
        rows: List[List[Any]] = []
        for i in range(n):
            sample = dataset[i]
            batch = collate_voice_clone_batch([sample])
            ref_16 = _ensure_batch_time(batch["ref_wav_16k"].to(device))
            tgt_24 = _ensure_batch_time(batch["target_wav_24k"].to(device))
            input_ids = batch["input_ids"].to(device)
            caption = batch.get("text")
            if not caption:
                caption = _caption_from_input_ids(batch["input_ids"].squeeze(0), id_to_char)

            with torch.no_grad():
                wv_ref = wavlm(ref_16, sampling_rate=16_000, grad_through_input=False)
                gst_out, _ = gst(wv_ref.frame_hidden_states, wv_ref.frame_mask)
                ref_s = gst_out.ref_s
                pred_wav, _ = kmodel.forward_with_tokens(input_ids, ref_s, speed=cfg.speed)
                pred_wav = _ensure_batch_time(pred_wav)

            pred_np = pred_wav.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            tgt_np = tgt_24.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            sr = 24_000
            rows.append(
                [
                    i,
                    str(caption)[:2048],
                    wb.Audio(pred_np, sample_rate=sr),
                    wb.Audio(tgt_np, sample_rate=sr),
                ]
            )
        table = wb.Table(columns=columns, data=rows)
        wandb_run.log({f"epoch_audio/epoch_{epoch + 1}": table}, step=global_step)
    finally:
        kmodel.train(prev_k_train)
        gst.train(prev_g_train)


def train_loop(
    dataloader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    *,
    epochs: int = 1,
    max_steps: Optional[int] = None,
    ckpt_dir: Optional[Path] = None,
    resume: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
    wandb_num_samples: int = 3,
) -> None:
    kmodel, gst, wavlm, disc, mel_loss_mod, kokoro_cfg = build_models(cfg, device)
    vocab: Dict[str, int] = kokoro_cfg["vocab"]
    params_g = generator_trainable_parameters(kmodel, gst)
    opt_g = torch.optim.AdamW(params_g, lr=cfg.lr_g, weight_decay=cfg.weight_decay_g)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg.lr_d, weight_decay=cfg.weight_decay_d)
    scaler: Optional[GradScaler] = GradScaler("cuda", enabled=cfg.use_amp) if device.type == "cuda" else None

    step = 0
    if resume is not None:
        step = load_checkpoint(resume, gst=gst, disc=disc, kmodel=kmodel, opt_g=opt_g, opt_d=opt_d, device=device)
        print(f"Resumed step {step} from {resume}")

    global_step = step
    try:
        expected_batches_per_epoch = len(dataloader)
    except TypeError:
        expected_batches_per_epoch = None

    try:
        from tqdm.auto import tqdm as _tqdm_cls
    except ImportError:
        _tqdm_cls = None  # type: ignore[misc, assignment]

    for epoch_idx in range(epochs):
        epoch_sums = {
            "loss_g": 0.0,
            "loss_mel": 0.0,
            "loss_spk": 0.0,
            "loss_slm_g": 0.0,
            "loss_d": 0.0,
        }
        epoch_count = 0
        batches_seen = 0

        # Manual tqdm updates: wrapping ``for batch in tqdm(dataloader)`` and then ``break`` skips the
        # generator's post-yield update, so the bar can freeze at (N-1)/N.
        pbar_ctx: Any = contextlib.nullcontext()
        use_tqdm_pbar = False
        if _tqdm_cls is not None and expected_batches_per_epoch is not None:
            n_tqdm = expected_batches_per_epoch
            if max_steps is not None:
                n_tqdm = min(expected_batches_per_epoch, max(0, max_steps - global_step))
            if n_tqdm > 0:
                use_tqdm_pbar = True
                pbar_ctx = _tqdm_cls(
                    total=n_tqdm,
                    desc=f"train epoch {epoch_idx + 1}/{epochs}",
                    leave=False,
                    dynamic_ncols=True,
                )

        with pbar_ctx as pbar:
            for batch in dataloader:
                global_step += 1
                batches_seen += 1
                with torch.no_grad():
                    ref_16 = _ensure_batch_time(batch["ref_wav_16k"].to(device))
                    tgt_24 = _ensure_batch_time(batch["target_wav_24k"].to(device))
                    input_ids = batch["input_ids"].to(device)
                    wv_ref = wavlm(ref_16, sampling_rate=16_000, grad_through_input=False)
                    gst_out, _ = gst(wv_ref.frame_hidden_states, wv_ref.frame_mask)
                    ref_s = gst_out.ref_s
                    pred_wav, _ = kmodel.forward_with_tokens(input_ids, ref_s, speed=cfg.speed)
                    pred_wav = _ensure_batch_time(pred_wav)

                for _ in range(cfg.slm_d_steps_per_g_step):
                    ld = slm_discriminator_step(
                        disc,
                        wavlm,
                        opt_d,
                        pred_wav_24k=pred_wav,
                        target_wav_24k=tgt_24,
                        grad_clip=cfg.grad_clip_d,
                        scaler=scaler,
                        use_amp=cfg.use_amp,
                    )

                metrics = training_step_generator(
                    kmodel,
                    gst,
                    wavlm,
                    disc,
                    mel_loss_mod,
                    opt_g,
                    batch,
                    cfg.loss_weights,
                    speed=cfg.speed,
                    grad_clip=cfg.grad_clip_g,
                    scaler=scaler,
                    use_amp=cfg.use_amp,
                )
                metrics["loss_d"] = float(ld.detach()) if isinstance(ld, torch.Tensor) else float(ld)

                epoch_count += 1
                for k in epoch_sums:
                    epoch_sums[k] += metrics[k]

                if pbar is not None:
                    pbar.update(1)

                at_log_interval = global_step % cfg.log_interval == 0
                at_max_stop = max_steps is not None and global_step >= max_steps

                if at_log_interval or at_max_stop:
                    if pbar is not None:
                        postfix: Dict[str, Any] = {"step": global_step}
                        postfix.update({k: round(float(v), 4) for k, v in metrics.items()})
                        pbar.set_postfix(**postfix, refresh=True)
                    else:
                        sys.stdout.write("\r" + _terminal_metrics_cr_line(global_step, metrics))
                        sys.stdout.flush()
                    if wandb_run is not None:
                        w = cfg.loss_weights
                        wandb_run.log(
                            {
                                "train/loss_g": metrics["loss_g"],
                                "train/loss_mel": metrics["loss_mel"],
                                "train/loss_spk": metrics["loss_spk"],
                                "train/loss_slm_g": metrics["loss_slm_g"],
                                "train/loss_d": metrics["loss_d"],
                                "train/weighted_mel": w.lambda_mel * metrics["loss_mel"],
                                "train/weighted_spk": w.lambda_spk * metrics["loss_spk"],
                                "train/weighted_slm": w.lambda_slm * metrics["loss_slm_g"],
                                "train/epoch": float(epoch_idx + 1),
                            },
                            step=global_step,
                        )

                if ckpt_dir is not None and global_step % cfg.checkpoint_interval == 0:
                    save_checkpoint(
                        ckpt_dir / f"checkpoint_{global_step}.pt",
                        gst=gst,
                        disc=disc,
                        kmodel=kmodel,
                        opt_g=opt_g,
                        opt_d=opt_d,
                        step=global_step,
                        cfg=cfg,
                    )

                if max_steps is not None and global_step >= max_steps:
                    break

        if not use_tqdm_pbar:
            sys.stdout.write("\n")
            sys.stdout.flush()

        full_epoch = (
            expected_batches_per_epoch is not None
            and expected_batches_per_epoch > 0
            and batches_seen == expected_batches_per_epoch
        )
        if wandb_run is not None and epoch_count > 0 and full_epoch:
            mean_payload = {f"epoch/mean_{k}": epoch_sums[k] / epoch_count for k in epoch_sums}
            wandb_run.log(mean_payload, step=global_step)
            _log_wandb_epoch_audio_table(
                wandb_run,
                dataset=dataloader.dataset,
                kmodel=kmodel,
                gst=gst,
                wavlm=wavlm,
                cfg=cfg,
                device=device,
                epoch=epoch_idx,
                global_step=global_step,
                num_samples=wandb_num_samples,
                vocab=vocab,
            )

        if max_steps is not None and global_step >= max_steps:
            return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train voice-clone GST + adapters + SLM discriminator.")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu | mps (default: auto)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--amp", action="store_true", help="CUDA autocast + GradScaler")
    p.add_argument("--kokoro-repo", type=str, default="hexgrad/Kokoro-82M")
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSONL manifest (see voice_clone/DATASET.md).",
    )
    p.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help="Directory for resolving relative paths in the manifest (default: manifest file's parent).",
    )
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (requires wandb package).")
    p.add_argument(
        "--wandb-project",
        type=str,
        default="voice-clone-kokoro",
        help="Wandb project name (used when --wandb).",
    )
    p.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name (optional).")
    p.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity / team (optional).")
    p.add_argument(
        "--wandb-samples",
        type=int,
        default=3,
        help="Number of pred vs target audio rows to log per epoch when --wandb (default: 3).",
    )
    p.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Log wandb in offline mode (no network upload until sync).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop manifest training after this many optimizer steps (across epochs).",
    )
    return p.parse_args()


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    # ROCm/MIOpen: default to errors-only logging (quiets IsEnoughWorkspace spam). Set MIOPEN_LOG_LEVEL
    # yourself (e.g. 4) if you need MIOpen warnings. See AMD MIOpen env variable docs.
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
    _configure_training_warnings()
    device = _resolve_device(args.device)
    cfg = TrainConfig(kokoro_repo_id=args.kokoro_repo, use_amp=args.amp)

    wandb_run: Optional[Any] = None
    if args.wandb:
        wb = _import_wandb()
        wandb_config = asdict(cfg)
        wandb_config.update(
            {
                "device": str(device),
                "manifest": str(args.manifest),
                "manifest_root": str(args.manifest_root) if args.manifest_root else None,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "resume": str(args.resume) if args.resume else None,
                "ckpt_dir": str(args.ckpt_dir) if args.ckpt_dir else None,
            }
        )
        init_kw: Dict[str, Any] = {
            "project": args.wandb_project,
            "config": wandb_config,
        }
        if args.wandb_entity:
            init_kw["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kw["name"] = args.wandb_run_name
        if args.wandb_offline:
            init_kw["mode"] = "offline"
        wandb_run = wb.init(**init_kw)
    try:
        _run_training(args, device=device, cfg=cfg, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _run_training(
    args: argparse.Namespace,
    *,
    device: torch.device,
    cfg: TrainConfig,
    wandb_run: Optional[Any],
) -> None:
    vocab, ctx = kokoro_vocab_and_context_length(cfg.kokoro_repo_id)
    ds: Dataset = VoiceCloneManifestDataset(
        args.manifest,
        kokoro_repo_id=cfg.kokoro_repo_id,
        vocab=vocab,
        context_length=ctx,
        manifest_root=args.manifest_root,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_voice_clone_batch)
    train_loop(
        dl,
        cfg,
        device,
        epochs=args.epochs,
        max_steps=args.max_steps,
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
        wandb_run=wandb_run,
        wandb_num_samples=args.wandb_samples,
    )


if __name__ == "__main__":
    main()
