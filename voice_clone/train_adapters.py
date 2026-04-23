"""Train SegmentGST + Kokoro L-adapters + waveform discriminator (Kokoro and WeSpeaker frozen)."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import warnings
from dataclasses import asdict, fields, replace as _dc_replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    discriminator_loss_lsgan,
    feature_matching_loss,
    generator_loss_lsgan,
    speaker_input_mel_from_waveform,
    speaker_cosine_loss,
)
from .mhubert_encoder import MHuBERTEncoder
from .segment_gst import SegmentGST
from .train_profiling import BreakdownAggregator, StepStopwatch, build_torch_profiler
from .wespeaker_sv import WeSpeakerSV, WeSpeakerSVOutput
from .discriminators.hifigan import HiFiGANMPDMSDDiscriminator


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
    line = f"step {global_step} loss_g={metrics['loss_g']:.4f}"
    try:
        width = max(40, shutil.get_terminal_size(fallback=(100, 24)).columns - 1)
    except OSError:
        width = 100
    if len(line) < width:
        line = line + " " * (width - len(line))
    return line[:width]


def freeze_kokoro_except_adapters(kmodel: KModel) -> None:
    """Freeze Kokoro except L-adapters; keep backbone in ``train()`` with no stochastic dropout.

    ROCm/MIOpen requires RNN backward in training mode. We use ``train()`` on the full model
    (so validation can restore with ``kmodel.train(prev_k_train)`` correctly) and disable
    dropout on ``nn.Dropout`` and ``nn.RNNBase`` modules instead of relying on ``eval()``.
    """
    for p in kmodel.parameters():
        p.requires_grad_(False)

    if kmodel.predictor.text_encoder.adapters is not None:
        for p in kmodel.predictor.text_encoder.adapters.parameters():
            p.requires_grad_(True)

    if kmodel.decoder.decoder_adapters is not None:
        for p in kmodel.decoder.decoder_adapters.parameters():
            p.requires_grad_(True)

    if kmodel.decoder.generator.adapters is not None:
        for p in kmodel.decoder.generator.adapters.parameters():
            p.requires_grad_(True)

    kmodel.train(True)
    for m in kmodel.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if isinstance(m, nn.RNNBase) and hasattr(m, "dropout"):
            m.dropout = 0.0


def _effective_optimizer_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int:
    """Outer-loop optimizer steps per epoch: ``ceil(num_batches / grad_accum_steps)``."""
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    nb = int(num_batches)
    ga = int(grad_accum_steps)
    return (nb + ga - 1) // ga


def strip_weight_norm_frozen(model: nn.Module) -> None:
    """Bake weight_norm decompositions into static parameters.

    Safe on frozen modules where the gradient-normalization benefit of
    weight_norm is irrelevant, and removing it lets AMP autocast caching
    work correctly on ROCm (avoids MIOpen 1-D fallback).
    """
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    for m in model.modules():
        if is_parametrized(m, "weight"):
            remove_parametrizations(m, "weight")


def _build_scheduler(
    opt: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: Optional[int],
    lr_min: float,
    *,
    start_factor: float = 1e-3,
) -> torch.optim.lr_scheduler.LRScheduler:
    """LinearLR warmup, then CosineAnnealingLR decay when *total_steps* is known."""
    wu = int(warmup_steps)
    if wu <= 0:
        if total_steps is not None and int(total_steps) > 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=int(total_steps), eta_min=lr_min,
            )
        # No horizon (e.g. D scheduler when ``disc_start`` consumes the whole run): hold base LR.
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda _epoch: 1.0)

    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=start_factor, end_factor=1.0, total_iters=wu,
    )
    ts = int(total_steps) if total_steps is not None else None
    if ts is not None and ts > wu:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=ts - wu, eta_min=lr_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[wu],
        )
    return warmup


def generator_trainable_parameters(kmodel: KModel, gst: SegmentGST) -> List[nn.Parameter]:
    return list(gst.parameters()) + [p for p in kmodel.parameters() if p.requires_grad]


def _rescale_optimizer_grads(opt: torch.optim.Optimizer, factor: float) -> None:
    """Renormalize already-accumulated grads without rebuilding the forward pass."""
    if factor == 1.0:
        return
    for group in opt.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.mul_(factor)


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


def load_universal_style_vector(path: str, *, ref_dim: int) -> torch.Tensor:
    """Load the SegmentGST base-voice bias vector from a checkpoint file."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Universal style vector not found at {p}")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)
    if obj.dim() != 1:
        raise ValueError(
            f"Universal style vector at {p} must be 1-D with shape ({ref_dim},), got {tuple(obj.shape)}"
        )
    if int(obj.numel()) != ref_dim:
        raise ValueError(
            f"Universal style vector at {p} has length {int(obj.numel())}, expected {ref_dim}"
        )
    return obj.detach().to(dtype=torch.float32).contiguous()


def build_models(
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[KModel, SegmentGST, MHuBERTEncoder, WeSpeakerSV, HiFiGANMPDMSDDiscriminator, MelReconstructionLoss, Dict[str, Any]]:
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
    strip_weight_norm_frozen(kmodel)

    mhubert = MHuBERTEncoder(
        repo_id=cfg.mhubert_repo_id,
        extract_layer=cfg.mhubert_extract_layer,
    ).to(device)

    sv_model = WeSpeakerSV.from_checkpoint(
        cfg.wespeaker_checkpoint_path,
        embedding_dim=cfg.wespeaker_embedding_dim,
        sample_rate=cfg.wespeaker_sample_rate,
        device=device,
        dtype=None,
    )

    universal_style_vector = load_universal_style_vector(
        cfg.universal_style_vector_path,
        ref_dim=256,
    )
    gst = SegmentGST(
        frame_dim=mhubert.hidden_size,
        embed_dim=256,
        num_heads=8,
        ref_dim=256,
        style_dec_dim=128,
        dropout=cfg.gst_dropout,
        universal_style_vector=universal_style_vector,
    )
    gst = gst.to(device)

    disc = HiFiGANMPDMSDDiscriminator().to(device)

    mel_loss = build_mel_loss(kokoro_cfg, cfg.mel, device)
    return kmodel, gst, mhubert, sv_model, disc, mel_loss, kokoro_cfg


def save_checkpoint(
    path: Path,
    *,
    gst: SegmentGST,
    disc: nn.Module,
    kmodel: KModel,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
    scaler_g: Optional[GradScaler] = None,
    scaler_d: Optional[GradScaler] = None,
    sched_g: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    sched_d: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    generator_updates: int = 0,
    discriminator_updates: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "step": step,
        "train_config": asdict(cfg),
        "segment_gst": gst.state_dict(),
        "waveform_discriminator": disc.state_dict(),
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
        "scaler_g": scaler_g.state_dict() if scaler_g is not None else None,
        "scaler_d": scaler_d.state_dict() if scaler_d is not None else None,
        "scheduler_g": sched_g.state_dict() if sched_g is not None else None,
        "scheduler_d": sched_d.state_dict() if sched_d is not None else None,
        "generator_updates": int(generator_updates),
        "discriminator_updates": int(discriminator_updates),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    gst: SegmentGST,
    disc: nn.Module,
    kmodel: KModel,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
    scaler_g: Optional[GradScaler] = None,
    scaler_d: Optional[GradScaler] = None,
    sched_g: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    sched_d: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    resume_state: Optional[Dict[str, Any]] = None,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gst.load_state_dict(ckpt["segment_gst"])
    # Backward compatibility: fall back to old discriminator key if present.
    disc_state = ckpt.get("waveform_discriminator") or ckpt.get("slm_discriminator")
    if disc_state is not None:
        disc.load_state_dict(disc_state)
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
    if ckpt.get("scaler_g") and scaler_g is not None:
        scaler_g.load_state_dict(ckpt["scaler_g"])
    if ckpt.get("scaler_d") and scaler_d is not None:
        scaler_d.load_state_dict(ckpt["scaler_d"])
    scheduler_g_loaded = False
    scheduler_d_loaded = False
    if ckpt.get("scheduler_g") and sched_g is not None:
        sched_g.load_state_dict(ckpt["scheduler_g"])
        scheduler_g_loaded = True
    if ckpt.get("scheduler_d") and sched_d is not None:
        sched_d.load_state_dict(ckpt["scheduler_d"])
        scheduler_d_loaded = True
    if resume_state is not None:
        step = int(ckpt.get("step", 0))
        resume_state.update(
            {
                "generator_updates": int(ckpt.get("generator_updates", step)),
                "discriminator_updates": int(ckpt.get("discriminator_updates", 0)),
                "scheduler_g_loaded": scheduler_g_loaded,
                "scheduler_d_loaded": scheduler_d_loaded,
            }
        )
    return int(ckpt.get("step", 0))


def generator_loss_backward(
    kmodel: KModel,
    gst: SegmentGST,
    mel_loss_mod: MelReconstructionLoss,
    opt_g: torch.optim.Optimizer,
    pred_wav: torch.Tensor,
    pred_wav_seg: torch.Tensor,
    ref_pooled_detached: torch.Tensor,
    gen_pooled: torch.Tensor,
    tgt_24: torch.Tensor,
    *,
    disc_waveform: HiFiGANMPDMSDDiscriminator,
    real_disc_features: Optional[List[List[torch.Tensor]]],
    weights: LossWeights,
    scaler: Optional[GradScaler],
    use_amp: bool,
    disable_amp_for_stft: bool = True,
    scale_factor: float = 1.0,
    pred_lengths: Optional[torch.Tensor] = None,
    target_lengths: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute mel + speaker + (optional) GAN losses and call ``.backward()`` only.

    ``scale_factor`` multiplies the total loss before backward (use ``1/N`` for gradient
    accumulation over *N* micro-steps).

    Returns per-component loss values (**unscaled**, detached) for logging.
    """
    with _cuda_amp_context(use_amp, pred_wav):
        if disable_amp_for_stft and pred_wav.is_cuda:
            with autocast("cuda", enabled=False):
                out_mel = mel_loss_mod(
                    pred_wav.float(), tgt_24.float(),
                    pred_lengths=pred_lengths, target_lengths=target_lengths,
                )
            l_mel = out_mel.loss
        else:
            out_mel = mel_loss_mod(
                pred_wav, tgt_24,
                pred_lengths=pred_lengths, target_lengths=target_lengths,
            )
            l_mel = out_mel.loss

        l_spk = speaker_cosine_loss(ref_pooled_detached, gen_pooled)

        # Optional GAN losses: only active when real_disc_features is provided (after disc_start_step).
        l_g_adv = torch.zeros((), device=pred_wav.device, dtype=pred_wav.dtype)
        l_fm = torch.zeros((), device=pred_wav.device, dtype=pred_wav.dtype)
        if real_disc_features is not None:
            fake_logits, fake_feats = disc_waveform(pred_wav_seg)
            l_g_adv = generator_loss_lsgan(fake_logits)
            # Detach real features so generator step does not backprop into D.
            real_feats_detached: List[List[torch.Tensor]] = [
                [f.detach() for f in per_disc] for per_disc in real_disc_features
            ]
            l_fm = feature_matching_loss(real_feats_detached, fake_feats)

        loss_g = (
            weights.lambda_mel * l_mel
            + weights.lambda_spk * l_spk
            + weights.lambda_adv * l_g_adv
            + weights.lambda_fm * l_fm
        )

    scaled = loss_g * scale_factor
    if scaler is not None and use_amp:
        scaler.scale(scaled).backward()
    else:
        scaled.backward()

    return {
        "loss_g": float(loss_g.detach()),
        "loss_mel": float(l_mel.detach()),
        "loss_spk": float(l_spk.detach()),
        "loss_adv_g": float(l_g_adv.detach()),
        "loss_fm": float(l_fm.detach()),
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


def _log_wandb_validation(
    wandb_run: Any,
    *,
    dataset: Dataset,
    kmodel: KModel,
    gst: SegmentGST,
    mhubert: MHuBERTEncoder,
    sv_model: WeSpeakerSV,
    mel_loss_mod: MelReconstructionLoss,
    cfg: TrainConfig,
    device: torch.device,
    global_step: int,
    num_samples: int,
    vocab: Dict[str, int],
    val_dataset: Optional[Dataset] = None,
) -> None:
    """Eval forward on the first ``num_samples`` items: ``val/*`` scalars and pred vs target audio table."""
    wb = _import_wandb()
    eval_ds = val_dataset if val_dataset is not None else dataset
    n = min(int(num_samples), len(eval_ds))
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
        mel_l1_sum = 0.0
        mel_l2_sum = 0.0
        spk_sum = 0.0
        for i in range(n):
            sample = eval_ds[i]
            batch = collate_voice_clone_batch([sample])
            ref_16 = _ensure_batch_time(batch["ref_wav_16k"].to(device))
            ref_lengths = batch["ref_lengths"].to(device)
            tgt_24 = _ensure_batch_time(batch["target_wav_24k"].to(device))
            target_lengths = batch["target_lengths"].to(device)
            input_ids = batch["input_ids"].to(device)
            input_ids_lengths = batch["input_ids_lengths"].to(device)
            caption = batch.get("text")
            if not caption:
                caption = _caption_from_input_ids(batch["input_ids"].squeeze(0), id_to_char)

            with torch.no_grad():
                ref_attn_mask = (
                    torch.arange(ref_16.size(1), device=device).unsqueeze(0) < ref_lengths.unsqueeze(1)
                ).long()
                mhubert_out = mhubert(ref_16, attention_mask=ref_attn_mask)
                gst_out, _ = gst(mhubert_out.hidden_states, mhubert_out.frame_mask)
                ref_s = gst_out.ref_s
                # Match training: Kokoro forward is B=1 and must not see token-padding as phonemes.
                ids = input_ids[0, : int(input_ids_lengths[0].item())].unsqueeze(0)
                pred_wav, _ = kmodel.forward_with_tokens(ids, ref_s, speed=cfg.speed)
                pred_wav = _ensure_batch_time(pred_wav)
                pred_wav = pred_wav.clamp(-1.0, 1.0)
                pred_lengths = torch.tensor([pred_wav.shape[1]], dtype=torch.long, device=device)

                out_mel = mel_loss_mod(
                    pred_wav,
                    tgt_24,
                    pred_lengths=pred_lengths,
                    target_lengths=target_lengths,
                )
                d = out_mel.mel_pred - out_mel.mel_target
                mel_l1_sum += float(d.abs().mean().item())
                mel_l2_sum += float((d**2).mean().item())

                wv_ref: WeSpeakerSVOutput = sv_model(
                    ref_16,
                    sampling_rate=16_000,
                    grad_through_input=False,
                    waveform_lengths=ref_lengths,
                    return_frame_features=False,
                )
                mel_gen = speaker_input_mel_from_waveform(
                    pred_wav,
                    mel_transform=sv_model._waveforms_to_mel,
                    amp_enabled=cfg.use_amp,
                    disable_amp_for_stft=cfg.disable_amp_for_stft,
                )
                wv_gen: WeSpeakerSVOutput = sv_model.forward_from_mel(
                    mel_gen,
                    grad_through_input=False,
                    return_frame_features=False,
                )
                spk_sum += float(speaker_cosine_loss(wv_ref.pooled_embedding, wv_gen.pooled_embedding).item())

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
        inv_n = 1.0 / float(n)
        table = wb.Table(columns=columns, data=rows)
        wandb_run.log(
            {
                "val/mel_l1": mel_l1_sum * inv_n,
                "val/mel_l2": mel_l2_sum * inv_n,
                "val/speaker_loss": spk_sum * inv_n,
                "val/audio_samples": table,
            },
            step=global_step,
        )
    finally:
        kmodel.train(prev_k_train)
        gst.train(prev_g_train)


def _apply_config_overrides(cfg: TrainConfig, overrides: Dict[str, Any]) -> TrainConfig:
    """Return a new ``TrainConfig`` with *overrides* applied.

    Keys that are fields of :class:`TrainConfig` are replaced directly.
    Keys that are fields of :class:`LossWeights` (e.g. ``lambda_mel``) are
    routed into the nested ``loss_weights`` sub-dataclass.
    """
    top_names = {f.name for f in fields(TrainConfig)}
    lw_names = {f.name for f in fields(LossWeights)}
    top_kw: Dict[str, Any] = {}
    lw_kw: Dict[str, Any] = {}
    for k, v in overrides.items():
        if k in top_names:
            top_kw[k] = v
        elif k in lw_names:
            lw_kw[k] = v
        else:
            raise ValueError(f"Unknown config override key: {k!r}")
    if lw_kw:
        top_kw["loss_weights"] = _dc_replace(cfg.loss_weights, **lw_kw)
    return _dc_replace(cfg, **top_kw)


def _should_save_and_validate(
    *,
    global_step: int,
    cfg: TrainConfig,
    max_steps: Optional[int],
    epoch_idx: int,
    epochs: int,
    epoch_done: bool,
    expected_batches_per_epoch: Optional[int],
    batches_seen: int,
) -> bool:
    is_periodic_step = global_step > 0 and global_step % cfg.checkpoint_interval == 0
    if not cfg.save_final_checkpoint:
        return is_periodic_step
    at_max_stop = max_steps is not None and global_step >= max_steps
    is_last_step = at_max_stop or (
        max_steps is None
        and epoch_idx == (epochs - 1)
        and (
            epoch_done
            or (expected_batches_per_epoch is not None
                and expected_batches_per_epoch > 0
                and batches_seen >= expected_batches_per_epoch)
        )
    )
    return is_periodic_step or is_last_step


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
    profile_breakdown: bool = False,
    profile_breakdown_steps: int = 3,
    torch_profiler_trace: Optional[Path] = None,
    val_dataset: Optional[Dataset] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    report_callback: Optional[Callable[[int, float], None]] = None,
) -> None:
    if config_overrides:
        cfg = _apply_config_overrides(cfg, config_overrides)

    if cfg.grad_accum_steps < 1:
        raise ValueError("cfg.grad_accum_steps must be >= 1")
    if cfg.log_interval < 1:
        raise ValueError("cfg.log_interval must be >= 1")
    if cfg.checkpoint_interval < 1:
        raise ValueError("cfg.checkpoint_interval must be >= 1")

    kmodel, gst, mhubert, sv_model, disc, mel_loss_mod, kokoro_cfg = build_models(cfg, device)
    vocab: Dict[str, int] = kokoro_cfg["vocab"]
    params_g = generator_trainable_parameters(kmodel, gst)
    adam_betas = (cfg.adam_b1, cfg.adam_b2)
    opt_g = torch.optim.AdamW(
        params_g, lr=cfg.lr_g, betas=adam_betas, weight_decay=cfg.weight_decay_g,
    )
    opt_d = torch.optim.AdamW(
        disc.parameters(), lr=cfg.lr_d, betas=adam_betas, weight_decay=cfg.weight_decay_d,
    )
    scaler_g: Optional[GradScaler] = GradScaler("cuda", enabled=cfg.use_amp) if device.type == "cuda" else None
    scaler_d: Optional[GradScaler] = GradScaler("cuda", enabled=cfg.use_amp) if device.type == "cuda" else None

    try:
        expected_batches_per_epoch = len(dataloader)
    except TypeError:
        expected_batches_per_epoch = None

    # Compute total effective steps for scheduler horizon.
    total_steps_g: Optional[int] = max_steps
    if total_steps_g is None and expected_batches_per_epoch is not None:
        per_epoch = _effective_optimizer_steps_per_epoch(
            expected_batches_per_epoch, cfg.grad_accum_steps,
        )
        total_steps_g = per_epoch * epochs
    total_steps_d: Optional[int] = None
    if total_steps_g is not None:
        total_steps_d = max(0, total_steps_g - cfg.disc_start_step)

    sched_g: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    sched_d: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if cfg.warmup_steps > 0 or (total_steps_g is not None and total_steps_g > 0):
        sched_g = _build_scheduler(opt_g, cfg.warmup_steps, total_steps_g, cfg.lr_min_g)
        sched_d = _build_scheduler(opt_d, cfg.warmup_steps, total_steps_d, cfg.lr_min_d)

    step = 0
    generator_updates = 0
    discriminator_updates = 0
    if resume is not None:
        resume_state: Dict[str, Any] = {}
        step = load_checkpoint(
            resume, gst=gst, disc=disc, kmodel=kmodel, opt_g=opt_g, opt_d=opt_d,
            device=device, scaler_g=scaler_g, scaler_d=scaler_d,
            sched_g=sched_g, sched_d=sched_d, resume_state=resume_state,
        )
        print(f"Resumed step {step} from {resume}")
        generator_updates = int(resume_state.get("generator_updates", step))
        discriminator_updates = int(
            resume_state.get("discriminator_updates", max(0, step - cfg.disc_start_step))
        )
        if sched_g is not None and not bool(resume_state.get("scheduler_g_loaded", False)):
            for _ in range(generator_updates):
                sched_g.step()
        if sched_d is not None and not bool(resume_state.get("scheduler_d_loaded", False)):
            for _ in range(discriminator_updates):
                sched_d.step()

    global_step = step

    try:
        from tqdm.auto import tqdm as _tqdm_cls
    except ImportError:
        _tqdm_cls = None  # type: ignore[misc, assignment]

    prof: Any = None
    profile_breakdown_k = max(1, int(profile_breakdown_steps))
    profile_agg: Optional[BreakdownAggregator] = BreakdownAggregator() if profile_breakdown else None
    sw: Optional[StepStopwatch] = StepStopwatch(device) if profile_breakdown else None

    with contextlib.ExitStack() as _prof_stack:
        if torch_profiler_trace is not None:
            prof = build_torch_profiler(torch_profiler_trace, device)
            _prof_stack.enter_context(prof)

        for epoch_idx in range(epochs):
            epoch_sums = {
                "loss_g": 0.0,
                "loss_mel": 0.0,
                "loss_spk": 0.0,
                "loss_adv_g": 0.0,
                "loss_fm": 0.0,
                "loss_d": 0.0,
            }
            epoch_count = 0
            batches_seen = 0

            # Manual tqdm updates: wrapping ``for batch in tqdm(dataloader)`` and then ``break`` skips the
            # generator's post-yield update, so the bar can freeze at (N-1)/N.
            pbar_ctx: Any = contextlib.nullcontext()
            use_tqdm_pbar = False
            if _tqdm_cls is not None and expected_batches_per_epoch is not None:
                eff_steps = _effective_optimizer_steps_per_epoch(
                    expected_batches_per_epoch, cfg.grad_accum_steps,
                )
                n_tqdm = eff_steps
                if max_steps is not None:
                    n_tqdm = min(eff_steps, max(0, max_steps - global_step))
                if n_tqdm > 0:
                    use_tqdm_pbar = True
                    pbar_ctx = _tqdm_cls(
                        total=n_tqdm,
                        desc=f"train epoch {epoch_idx + 1}/{epochs}",
                        leave=False,
                        dynamic_ncols=True,
                    )

            with pbar_ctx as pbar:
                batch_iter = iter(dataloader)
                epoch_done = False
                inv_accum = 1.0 / cfg.grad_accum_steps

                while not epoch_done:
                    if sw is not None:
                        sw.begin_step()

                    opt_g.zero_grad(set_to_none=True)
                    opt_d.zero_grad(set_to_none=True)
                    include_gan = (global_step >= cfg.disc_start_step)

                    micro_sums: Dict[str, float] = {
                        "loss_g": 0.0,
                        "loss_mel": 0.0,
                        "loss_spk": 0.0,
                        "loss_adv_g": 0.0,
                        "loss_fm": 0.0,
                        "loss_d": 0.0,
                    }
                    micros_done = 0
                    nan_detected = False

                    for _micro in range(cfg.grad_accum_steps):
                        if sw is not None:
                            sw.start("dataloader")
                        try:
                            batch = next(batch_iter)
                        except StopIteration:
                            if sw is not None:
                                sw.discard_active()
                            epoch_done = True
                            break
                        if sw is not None:
                            sw.end("dataloader", sync_cuda=False)

                        micros_done += 1
                        batches_seen += 1

                        if sw is not None:
                            sw.start("h2d")
                        ref_16 = _ensure_batch_time(batch["ref_wav_16k"].to(device))
                        tgt_24 = _ensure_batch_time(batch["target_wav_24k"].to(device))
                        input_ids = batch["input_ids"].to(device)
                        input_ids_lengths = batch["input_ids_lengths"].to(device)
                        ref_lengths = batch["ref_lengths"].to(device)
                        target_lengths = batch["target_lengths"].to(device)
                        B = ref_16.size(0)
                        if sw is not None:
                            sw.end("h2d", sync_cuda=True)
                        if input_ids.dim() != 2:
                            raise ValueError("input_ids must be (batch, seq_len)")

                        # mHuBERT conditioning (frozen) -> GST -> ref_s.
                        with _cuda_amp_context(cfg.use_amp, ref_16):
                            if sw is not None:
                                sw.start("mhubert_ref")
                            ref_attn_mask = (
                                torch.arange(ref_16.size(1), device=device).unsqueeze(0)
                                < ref_lengths.unsqueeze(1)
                            ).long()
                            mhubert_out = mhubert(ref_16, attention_mask=ref_attn_mask)
                            gst_out, _ = gst(mhubert_out.hidden_states, mhubert_out.frame_mask)
                            ref_s = gst_out.ref_s
                            if sw is not None:
                                sw.end("mhubert_ref", sync_cuda=True)

                            # WeSpeaker ref embedding for speaker loss (frozen, detached).
                            if sw is not None:
                                sw.start("sv_ref")
                            with torch.no_grad():
                                wv_ref: WeSpeakerSVOutput = sv_model(
                                    ref_16,
                                    sampling_rate=16_000,
                                    grad_through_input=False,
                                    waveform_lengths=ref_lengths,
                                    return_frame_features=False,
                                )
                            if sw is not None:
                                sw.end("sv_ref", sync_cuda=True)

                            # Kokoro generator forward (24 kHz waveform) -- per-item loop
                            # because KModel.forward_with_tokens is hardcoded for B=1.
                            if sw is not None:
                                sw.start("kokoro_fwd")
                            pred_wavs_list: List[torch.Tensor] = []
                            for _bi in range(B):
                                ids_i = input_ids[_bi, :input_ids_lengths[_bi]].unsqueeze(0)
                                ref_s_i = ref_s[_bi].unsqueeze(0)
                                wav_i, _ = kmodel.forward_with_tokens(ids_i, ref_s_i, speed=cfg.speed)
                                wav_i = _ensure_batch_time(wav_i).clamp(-1.0, 1.0)
                                pred_wavs_list.append(wav_i.squeeze(0))
                            max_gen_len = max(w.shape[0] for w in pred_wavs_list)
                            pred_wav = torch.stack(
                                [torch.nn.functional.pad(w, (0, max_gen_len - w.shape[0]))
                                 for w in pred_wavs_list],
                                dim=0,
                            )
                            pred_wav_lengths = torch.tensor(
                                [w.shape[0] for w in pred_wavs_list],
                                dtype=torch.long, device=device,
                            )
                            if sw is not None:
                                sw.end("kokoro_fwd", sync_cuda=True)

                        # Generated audio SSL forward (with grad) for speaker loss.
                        if sw is not None:
                            sw.start("sv_gen")
                        mel_gen = speaker_input_mel_from_waveform(
                            pred_wav,
                            mel_transform=sv_model._waveforms_to_mel,
                            amp_enabled=cfg.use_amp,
                            disable_amp_for_stft=cfg.disable_amp_for_stft,
                        )
                        wv_gen: WeSpeakerSVOutput = sv_model.forward_from_mel(
                            mel_gen,
                            grad_through_input=True,
                            return_frame_features=False,
                        )
                        if sw is not None:
                            sw.end("sv_gen", sync_cuda=True)
                        
                        # Per-item random crop for discriminator.
                        segment_size = 16384
                        valid_lens = torch.minimum(pred_wav_lengths, target_lengths)
                        slice_len = int(valid_lens.min().clamp(max=segment_size).item())
                        pred_segs: List[torch.Tensor] = []
                        tgt_segs: List[torch.Tensor] = []
                        for _bi in range(B):
                            max_start = int(valid_lens[_bi].item()) - slice_len
                            start = (
                                int(torch.randint(0, max(1, max_start + 1), (1,)).item())
                                if max_start > 0 else 0
                            )
                            pred_segs.append(pred_wav[_bi, start : start + slice_len])
                            tgt_segs.append(tgt_24[_bi, start : start + slice_len])
                        pred_wav_seg = torch.stack(pred_segs, dim=0)
                        tgt_24_seg = torch.stack(tgt_segs, dim=0)

                        # Waveform discriminator on real vs fake (D-step).
                        if sw is not None:
                            sw.start("disc")
                        ld = torch.zeros((), device=device, dtype=torch.float32)
                        real_disc_features: Optional[List[List[torch.Tensor]]] = None
                        if include_gan:
                            # Wrap the D-step forward passes in AMP
                            with _cuda_amp_context(cfg.use_amp, tgt_24):
                                # Real path.
                                real_logits, real_feats = disc(tgt_24_seg)
                                # Fake (detached) path for D-step only.
                                fake_logits_detached, _ = disc(pred_wav_seg.detach())
                                ld = discriminator_loss_lsgan(real_logits, fake_logits_detached)
                            scaled_ld = ld * inv_accum
                            if scaler_d is not None and cfg.use_amp:
                                scaler_d.scale(scaled_ld).backward()
                            else:
                                scaled_ld.backward()
                            real_disc_features = real_feats
                        if sw is not None:
                            sw.end("disc", sync_cuda=True)

                        # Generator backward: mel + speaker + optional GAN (adv + FM).
                        if sw is not None:
                            sw.start("gen_backward")
                        g_metrics = generator_loss_backward(
                            kmodel,
                            gst,
                            mel_loss_mod,
                            opt_g,
                            pred_wav,
                            pred_wav_seg,
                            wv_ref.pooled_embedding.detach(),
                            wv_gen.pooled_embedding,
                            tgt_24,
                            disc_waveform=disc,
                            real_disc_features=real_disc_features,
                            weights=cfg.loss_weights,
                            scaler=scaler_g,
                            use_amp=cfg.use_amp,
                            disable_amp_for_stft=cfg.disable_amp_for_stft,
                            scale_factor=inv_accum,
                            pred_lengths=pred_wav_lengths,
                            target_lengths=target_lengths,
                        )
                        if sw is not None:
                            sw.end("gen_backward", sync_cuda=True)

                        loss_values = list(g_metrics.values())
                        loss_values.append(float(ld.detach()))
                        if any(not np.isfinite(v) for v in loss_values):
                            nan_detected = True
                            warnings.warn(
                                f"[step {global_step}] NaN/Inf in generator/discriminator loss "
                                f"(micro-step {_micro}); skipping this optimizer step.",
                                RuntimeWarning,
                                stacklevel=1,
                            )
                            break

                        for k in g_metrics:
                            micro_sums[k] += g_metrics[k]
                        micro_sums["loss_d"] += float(ld.detach()) if isinstance(ld, torch.Tensor) else float(ld)

                    if micros_done == 0:
                        break

                    if not nan_detected and micros_done != cfg.grad_accum_steps:
                        rescale = float(cfg.grad_accum_steps) / float(micros_done)
                        _rescale_optimizer_grads(opt_g, rescale)
                        if include_gan:
                            _rescale_optimizer_grads(opt_d, rescale)

                    if nan_detected:
                        opt_g.zero_grad(set_to_none=True)
                        opt_d.zero_grad(set_to_none=True)
                        # Important: GradScaler.update() expects inf checks recorded via scaler.step(...).
                        # When we skip the optimizer step entirely, there are no inf checks to consume.
                        if wandb_run is not None:
                            wandb_run.log({"train/nan_skips": 1}, step=global_step)
                        global_step += 1
                        if pbar is not None:
                            pbar.update(1)
                        if max_steps is not None and global_step >= max_steps:
                            break
                        continue

                    # -- Deferred optimizer steps: unscale, step --
                    d_stepped = include_gan
                    if include_gan:
                        if scaler_d is not None and cfg.use_amp:
                            scaler_d.unscale_(opt_d)
                        torch.nn.utils.clip_grad_norm_(disc.parameters(), cfg.grad_clip_norm_d)
                        if scaler_d is not None and cfg.use_amp:
                            scaler_d.step(opt_d)
                        else:
                            opt_d.step()

                    if scaler_g is not None and cfg.use_amp:
                        scaler_g.unscale_(opt_g)
                    torch.nn.utils.clip_grad_norm_(params_g, cfg.grad_clip_norm_g)
                    if scaler_g is not None and cfg.use_amp:
                        scaler_g.step(opt_g)
                    else:
                        opt_g.step()

                    g_stepped = True
                    if scaler_g is not None and cfg.use_amp:
                        _old_scale_g = float(scaler_g.get_scale())
                        scaler_g.update()
                        if float(scaler_g.get_scale()) < _old_scale_g:
                            g_stepped = False
                    elif scaler_g is not None:
                        scaler_g.update()

                    if include_gan:
                        if scaler_d is not None and cfg.use_amp:
                            _old_scale_d = float(scaler_d.get_scale())
                            scaler_d.update()
                            if float(scaler_d.get_scale()) < _old_scale_d:
                                d_stepped = False
                        elif scaler_d is not None:
                            scaler_d.update()

                    global_step += 1

                    metrics = {k: v / micros_done for k, v in micro_sums.items()}

                    if sched_g is not None and g_stepped:
                        generator_updates += 1
                        sched_g.step()
                    elif g_stepped:
                        generator_updates += 1
                    if d_stepped:
                        discriminator_updates += 1
                    if sched_d is not None and global_step > cfg.disc_start_step and d_stepped:
                        sched_d.step()

                    epoch_count += 1
                    for k in epoch_sums:
                        epoch_sums[k] += metrics[k]

                    if report_callback is not None:
                        report_callback(global_step, metrics["loss_mel"])

                    if pbar is not None:
                        pbar.update(1)

                    at_log_interval = global_step % cfg.log_interval == 0
                    at_max_stop = max_steps is not None and global_step >= max_steps
                    should_save_and_validate = _should_save_and_validate(
                        global_step=global_step,
                        cfg=cfg,
                        max_steps=max_steps,
                        epoch_idx=epoch_idx,
                        epochs=epochs,
                        epoch_done=epoch_done,
                        expected_batches_per_epoch=expected_batches_per_epoch,
                        batches_seen=batches_seen,
                    )

                    if at_log_interval or at_max_stop:
                        if pbar is not None:
                            pbar.set_postfix(step=global_step, loss_g=round(float(metrics["loss_g"]), 4), refresh=True)
                        else:
                            sys.stdout.write("\r" + _terminal_metrics_cr_line(global_step, metrics))
                            sys.stdout.flush()
                        if wandb_run is not None:
                            if sw is not None:
                                sw.start("wandb_log")
                            log_payload = {
                                    "train/loss_g": metrics["loss_g"],
                                    "train/loss_mel": metrics["loss_mel"],
                                    "train/loss_spk": metrics["loss_spk"],
                                    "train/loss_adv_g": metrics["loss_adv_g"],
                                    "train/loss_fm": metrics["loss_fm"],
                                    "train/loss_d": metrics["loss_d"],
                                    "train/epoch": float(epoch_idx + 1),
                                    "train/lr_g": opt_g.param_groups[0]["lr"],
                                    "train/lr_d": opt_d.param_groups[0]["lr"],
                            }
                            wandb_run.log(log_payload, step=global_step)
                            if sw is not None:
                                sw.end("wandb_log", sync_cuda=False)

                    if profile_agg is not None and sw is not None:
                        profile_agg.add_step(sw.finish_step())
                        if len(profile_agg) >= profile_breakdown_k:
                            profile_agg.print_summary(skip_first_step=True)
                            sw = None

                    if should_save_and_validate:
                        if ckpt_dir is not None:
                            save_checkpoint(
                                ckpt_dir / f"checkpoint_{global_step}.pt",
                                gst=gst,
                                disc=disc,
                                kmodel=kmodel,
                                opt_g=opt_g,
                                opt_d=opt_d,
                                step=global_step,
                                cfg=cfg,
                                scaler_g=scaler_g,
                                scaler_d=scaler_d,
                                sched_g=sched_g,
                                sched_d=sched_d,
                                generator_updates=generator_updates,
                                discriminator_updates=discriminator_updates,
                            )
                        if wandb_run is not None:
                            _log_wandb_validation(
                                wandb_run,
                                dataset=dataloader.dataset,
                                kmodel=kmodel,
                                gst=gst,
                                mhubert=mhubert,
                                sv_model=sv_model,
                                mel_loss_mod=mel_loss_mod,
                                cfg=cfg,
                                device=device,
                                global_step=global_step,
                                num_samples=wandb_num_samples,
                                vocab=vocab,
                                val_dataset=val_dataset,
                            )

                    if prof is not None:
                        prof.step()

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

            if max_steps is not None and global_step >= max_steps:
                return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train voice-clone GST + adapters + waveform discriminator.")
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
    p.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Optional JSONL validation manifest (same format as --manifest). When provided, "
        "validation metrics and audio samples are computed from this set instead of the training set.",
    )
    p.add_argument(
        "--val-manifest-root",
        type=Path,
        default=None,
        help="Directory for resolving relative paths in the validation manifest (default: val manifest's parent).",
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
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (default: from TrainConfig, which is 1).",
    )
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation micro-steps per optimizer step (default: from TrainConfig).",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="LR warmup steps (default: from TrainConfig).",
    )
    p.add_argument(
        "--disc-start-step",
        type=int,
        default=None,
        help="Step at which discriminator training begins (default: from TrainConfig).",
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save a checkpoint every N optimizer steps (default: from TrainConfig).",
    )
    p.add_argument(
        "--save-final-checkpoint",
        dest="save_final_checkpoint",
        action="store_true",
        default=None,
        help="Force a checkpoint at the end of training even if the last step is off-interval.",
    )
    p.add_argument(
        "--no-save-final-checkpoint",
        dest="save_final_checkpoint",
        action="store_false",
        help="Only save checkpoints at the configured interval.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop manifest training after this many optimizer steps (across epochs).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0). On Windows, spawn overhead applies; try 4–8 or use 0 if workers error.",
    )
    p.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="Coarse CUDA-synchronized timers per phase; prints aggregate after --profile-breakdown-steps.",
    )
    p.add_argument(
        "--profile-breakdown-steps",
        type=int,
        default=3,
        help="Steps to record before printing [profile-breakdown] summary; timing then stops (default: 3).",
    )
    p.add_argument(
        "--torch-profiler-trace",
        type=Path,
        default=None,
        metavar="PATH",
        help="Export torch.profiler Chrome trace JSON (open in chrome://tracing). Use max-steps>=4 for default schedule.",
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
    from dataclasses import replace as _replace
    _cli_overrides: Dict[str, Any] = {}
    if args.batch_size is not None:
        _cli_overrides["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        _cli_overrides["grad_accum_steps"] = args.grad_accum_steps
    if args.warmup_steps is not None:
        _cli_overrides["warmup_steps"] = args.warmup_steps
    if args.disc_start_step is not None:
        _cli_overrides["disc_start_step"] = args.disc_start_step
    if args.checkpoint_interval is not None:
        _cli_overrides["checkpoint_interval"] = args.checkpoint_interval
    if args.save_final_checkpoint is not None:
        _cli_overrides["save_final_checkpoint"] = args.save_final_checkpoint
    if _cli_overrides:
        cfg = _replace(cfg, **_cli_overrides)

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
                "num_workers": args.num_workers,
                "resume": str(args.resume) if args.resume else None,
                "ckpt_dir": str(args.ckpt_dir) if args.ckpt_dir else None,
                "profile_breakdown": args.profile_breakdown,
                "profile_breakdown_steps": args.profile_breakdown_steps,
                "torch_profiler_trace": str(args.torch_profiler_trace) if args.torch_profiler_trace else None,
                "val_manifest": str(args.val_manifest) if args.val_manifest else None,
            }
        )
        init_kw: Dict[str, Any] = {
            "project": args.wandb_project,
            "config": wandb_config,
            "settings": wb.Settings(quiet=True),
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

    val_ds: Optional[Dataset] = None
    if args.val_manifest is not None:
        val_ds = VoiceCloneManifestDataset(
            args.val_manifest,
            kokoro_repo_id=cfg.kokoro_repo_id,
            vocab=vocab,
            context_length=ctx,
            manifest_root=args.val_manifest_root,
        )

    nw = args.num_workers
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_voice_clone_batch,
        num_workers=nw,
        persistent_workers=nw > 0,
        pin_memory=device.type == "cuda",
    )
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
        profile_breakdown=args.profile_breakdown,
        profile_breakdown_steps=args.profile_breakdown_steps,
        torch_profiler_trace=args.torch_profiler_trace,
        val_dataset=val_ds,
    )


if __name__ == "__main__":
    main()
