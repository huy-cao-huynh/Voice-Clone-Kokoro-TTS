"""Train SegmentGST + Kokoro L-adapters + SLM discriminator (Kokoro and WavLM frozen)."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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


def _cuda_amp_context(use_amp: bool, reference_tensor: torch.Tensor):
    if use_amp and reference_tensor.is_cuda:
        return autocast("cuda")
    return contextlib.nullcontext()


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


class DummyVoiceCloneDataset(Dataset):
    """Random waveforms and token ids for shape / gradient smoke tests (not for quality)."""

    def __init__(
        self,
        n: int,
        *,
        device_for_tensors: bool = False,
        ref_samples_16k: int = 32_000,
        tgt_samples_24k: int = 48_000,
        seq_len: int = 32,
        vocab_max: int = 177,
    ) -> None:
        self.n = n
        self.ref_samples_16k = ref_samples_16k
        self.tgt_samples_24k = tgt_samples_24k
        self.seq_len = seq_len
        self.vocab_max = vocab_max
        self.device_for_tensors = device_for_tensors

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(idx + 13)
        ref = torch.randn(self.ref_samples_16k, generator=g) * 0.02
        tgt = torch.randn(self.tgt_samples_24k, generator=g) * 0.02
        # Valid Kokoro boundaries: 0; interior from vocab (skip 0).
        interior = torch.randint(1, self.vocab_max + 1, (self.seq_len - 2,), generator=g)
        ids = torch.cat([torch.tensor([0]), interior, torch.tensor([0])]).long()
        batch = {
            "ref_wav_16k": ref,
            "target_wav_24k": tgt,
            "input_ids": ids,
        }
        if self.device_for_tensors:
            raise NotImplementedError
        return batch


def train_loop(
    dataloader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    *,
    epochs: int = 1,
    max_steps: Optional[int] = None,
    ckpt_dir: Optional[Path] = None,
    resume: Optional[Path] = None,
) -> None:
    kmodel, gst, wavlm, disc, mel_loss_mod, _kokoro_cfg = build_models(cfg, device)
    params_g = generator_trainable_parameters(kmodel, gst)
    opt_g = torch.optim.AdamW(params_g, lr=cfg.lr_g, weight_decay=cfg.weight_decay_g)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg.lr_d, weight_decay=cfg.weight_decay_d)
    scaler: Optional[GradScaler] = GradScaler("cuda", enabled=cfg.use_amp) if device.type == "cuda" else None

    step = 0
    if resume is not None:
        step = load_checkpoint(resume, gst=gst, disc=disc, kmodel=kmodel, opt_g=opt_g, opt_d=opt_d, device=device)
        print(f"Resumed step {step} from {resume}")

    global_step = step
    for _epoch in range(epochs):
        for batch in dataloader:
            global_step += 1
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

            if global_step % cfg.log_interval == 0:
                print(f"step {global_step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

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
                return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train voice-clone GST + adapters + SLM discriminator.")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu | mps (default: auto)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--dummy-steps", type=int, default=2, help="Run N steps on random data (smoke test).")
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--amp", action="store_true", help="CUDA autocast + GradScaler")
    p.add_argument("--kokoro-repo", type=str, default="hexgrad/Kokoro-82M")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL manifest (see voice_clone/DATASET.md). If omitted, uses dummy random data.",
    )
    p.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help="Directory for resolving relative paths in the manifest (default: manifest file's parent).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cfg = TrainConfig(kokoro_repo_id=args.kokoro_repo, use_amp=args.amp)
    if args.manifest is not None:
        vocab, ctx = kokoro_vocab_and_context_length(cfg.kokoro_repo_id)
        ds = VoiceCloneManifestDataset(
            args.manifest,
            kokoro_repo_id=cfg.kokoro_repo_id,
            vocab=vocab,
            context_length=ctx,
            manifest_root=args.manifest_root,
        )
        dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_voice_clone_batch)
    else:
        ds = DummyVoiceCloneDataset(args.dummy_steps * 4)
        dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_voice_clone_batch)

    kmodel, gst, wavlm, disc, mel_loss_mod, _ = build_models(cfg, device)
    params_g = generator_trainable_parameters(kmodel, gst)
    opt_g = torch.optim.AdamW(params_g, lr=cfg.lr_g, weight_decay=cfg.weight_decay_g)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg.lr_d, weight_decay=cfg.weight_decay_d)
    scaler: Optional[GradScaler] = GradScaler("cuda", enabled=cfg.use_amp) if device.type == "cuda" else None

    if args.resume:
        load_checkpoint(args.resume, gst=gst, disc=disc, kmodel=kmodel, opt_g=opt_g, opt_d=opt_d, device=device)

    step = 0
    dl_iter = iter(dl)
    for _ in range(args.dummy_steps):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)
        step += 1
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
        print(f"step {step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if args.ckpt_dir:
        save_checkpoint(
            args.ckpt_dir / "checkpoint_last.pt",
            gst=gst,
            disc=disc,
            kmodel=kmodel,
            opt_g=opt_g,
            opt_d=opt_d,
            step=step,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
