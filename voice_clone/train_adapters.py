"""Train SegmentGST against a frozen Kokoro backbone with cached supervision."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, fields, replace as dc_replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from kokoro.model import KModel

from .config import LossWeights, MelLossConfig, TrainConfig, kokoro_vocab_and_context_length, load_kokoro_config
from .dataset import VoiceCloneManifestDataset, collate_voice_clone_batch
from .losses import (
    MelReconstructionLoss,
    discriminator_loss_lsgan,
    duration_loss_log_space,
    feature_matching_loss,
    generator_loss_lsgan,
    masked_l1_loss,
    speaker_contrastive_loss,
)
from .mhubert_encoder import MHuBERTEncoder
from .segment_gst import SegmentGST
from .wespeaker_sv import WeSpeakerSV
from .discriminators.hifigan import HiFiGANMPDMSDDiscriminator

_LEGACY_CHECKPOINT_KEYS = {"kokoro_lora", "duration_adapters", "decoder_adapters", "generator_adapters"}


def _cuda_amp_context(use_amp: bool, reference_tensor: torch.Tensor):
    if use_amp and reference_tensor.is_cuda:
        return autocast("cuda", enabled=True)
    return contextlib.nullcontext()


def freeze_kokoro_backbone(kmodel: nn.Module) -> None:
    for p in kmodel.parameters():
        p.requires_grad_(False)
    kmodel.train(True)


def resample_mono(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    import torchaudio

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() != 2:
        raise ValueError(f"waveform must be (time,) or (batch, time), got {tuple(wav.shape)}")
    if orig_sr == new_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=new_sr)


def build_mel_loss(kokoro_cfg: Dict[str, Any], mel_cfg: MelLossConfig, device: torch.device) -> MelReconstructionLoss:
    return MelReconstructionLoss(
        sample_rate=mel_cfg.sample_rate,
        n_mels=int(kokoro_cfg["n_mels"]),
        n_fft=mel_cfg.n_fft,
        hop_length=mel_cfg.hop_length,
        win_length=mel_cfg.win_length,
        f_min=mel_cfg.f_min,
        f_max=mel_cfg.f_max,
    ).to(device)


def load_universal_style_vector(path: str, *, ref_dim: int) -> torch.Tensor:
    p = Path(path).expanduser()
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)
    if obj.dim() != 1 or int(obj.numel()) != ref_dim:
        raise ValueError(f"Universal style vector at {p} must have shape ({ref_dim},)")
    return obj.detach().to(dtype=torch.float32).contiguous()


def build_kokoro_model(cfg: TrainConfig, device: torch.device) -> Tuple[KModel, Dict[str, Any]]:
    kokoro_cfg = load_kokoro_config(cfg.kokoro_repo_id)
    kmodel = KModel(repo_id=cfg.kokoro_repo_id, config=kokoro_cfg).to(device)
    freeze_kokoro_backbone(kmodel)
    return kmodel, kokoro_cfg


def build_models(
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[KModel, SegmentGST, MHuBERTEncoder, WeSpeakerSV, HiFiGANMPDMSDDiscriminator, MelReconstructionLoss, Dict[str, Any]]:
    kmodel, kokoro_cfg = build_kokoro_model(cfg, device)
    mhubert = MHuBERTEncoder(repo_id=cfg.mhubert_repo_id, extract_layer=cfg.mhubert_extract_layer).to(device)
    sv_model = WeSpeakerSV.from_checkpoint(
        cfg.wespeaker_checkpoint_path,
        embedding_dim=cfg.wespeaker_embedding_dim,
        sample_rate=cfg.wespeaker_sample_rate,
        device=device,
        dtype=None,
    )
    gst = SegmentGST(
        frame_dim=mhubert.hidden_size,
        embed_dim=cfg.gst_embed_dim,
        num_bases=1024,
        num_heads=4,
        ref_dim=256,
        style_dec_dim=128,
        dropout=cfg.gst_dropout,
        universal_style_vector=load_universal_style_vector(cfg.universal_style_vector_path, ref_dim=256),
    ).to(device)
    disc = HiFiGANMPDMSDDiscriminator().to(device)
    mel_loss = build_mel_loss(kokoro_cfg, cfg.mel, device)
    return kmodel, gst, mhubert, sv_model, disc, mel_loss, kokoro_cfg


def generator_trainable_parameters(kmodel: KModel, gst: SegmentGST) -> List[nn.Parameter]:
    del kmodel
    return list(gst.parameters())


def _assert_new_checkpoint_schema(ckpt: Dict[str, Any]) -> None:
    legacy = sorted(k for k in _LEGACY_CHECKPOINT_KEYS if k in ckpt)
    if legacy:
        raise ValueError(
            "Legacy adapter/LoRA checkpoint detected; this training stack only supports the new SegmentGST-only schema. "
            f"Found keys: {legacy}"
        )


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
    del kmodel
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "train_config": asdict(cfg),
            "segment_gst": gst.state_dict(),
            "waveform_discriminator": disc.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict(),
            "scaler_g": scaler_g.state_dict() if scaler_g is not None else None,
            "scaler_d": scaler_d.state_dict() if scaler_d is not None else None,
            "scheduler_g": sched_g.state_dict() if sched_g is not None else None,
            "scheduler_d": sched_d.state_dict() if sched_d is not None else None,
            "generator_updates": int(generator_updates),
            "discriminator_updates": int(discriminator_updates),
        },
        path,
    )


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
    del kmodel
    ckpt = torch.load(path, map_location=device, weights_only=False)
    _assert_new_checkpoint_schema(ckpt)
    gst.load_state_dict(ckpt["segment_gst"])
    disc_state = ckpt.get("waveform_discriminator") or ckpt.get("slm_discriminator")
    if disc_state is not None:
        disc.load_state_dict(disc_state)
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


def _effective_optimizer_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int:
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    return (int(num_batches) + int(grad_accum_steps) - 1) // int(grad_accum_steps)


def _build_scheduler(
    opt: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: Optional[int],
    lr_min: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    wu = int(warmup_steps)
    if wu <= 0:
        if total_steps is not None and int(total_steps) > 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(total_steps), eta_min=lr_min)
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda _step: 1.0)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=wu)
    if total_steps is not None and int(total_steps) > wu:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(total_steps) - wu, eta_min=lr_min)
        return torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[wu])
    return warmup


class LanguageHomogeneousUniqueSpeakerBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
        min_language_speakers: int = 2,
    ) -> None:
        if batch_size < 2:
            raise ValueError("Contrastive training requires batch_size >= 2")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.generator = generator
        self.min_language_speakers = int(min_language_speakers)
        self.skipped_languages: Dict[str, int] = {}
        self._batches = self._build_batches()

    def _dataset_rows(self) -> Sequence[Dict[str, Any]]:
        rows = getattr(self.dataset, "rows", None)
        if rows is None:
            raise ValueError("Dataset must expose .rows for language-aware batching")
        return rows

    def _build_batches(self) -> List[List[int]]:
        rows = self._dataset_rows()
        by_lang: Dict[str, List[int]] = {}
        for idx, row in enumerate(rows):
            by_lang.setdefault(str(row["lang_code"]).lower(), []).append(idx)

        gen = self.generator
        batches: List[List[int]] = []
        skipped: Dict[str, int] = {}
        for lang in sorted(by_lang):
            indices = list(by_lang[lang])
            if gen is not None:
                order = torch.randperm(len(indices), generator=gen).tolist()
                indices = [indices[i] for i in order]
            speaker_to_indices: Dict[str, List[int]] = {}
            for idx in indices:
                speaker = str(rows[idx].get("speaker_id", f"row-{idx}"))
                speaker_to_indices.setdefault(speaker, []).append(idx)
            if len(speaker_to_indices) < self.min_language_speakers:
                skipped[lang] = len(speaker_to_indices)
                continue

            pool = {spk: list(spk_indices) for spk, spk_indices in speaker_to_indices.items()}
            while True:
                available = [spk for spk, items in pool.items() if items]
                if len(available) < 2:
                    break
                batch: List[int] = []
                for spk in list(available):
                    if len(batch) >= self.batch_size:
                        break
                    batch.append(pool[spk].pop(0))
                if len(batch) >= 2:
                    batches.append(batch)
                else:
                    break
            leftover = [items[0] for items in pool.values() if items]
            if not self.drop_last and len(leftover) >= 2:
                batches.append(leftover[: self.batch_size])

        self.skipped_languages = skipped
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._batches:
            yield batch

    def __len__(self) -> int:
        return len(self._batches)


def _apply_config_overrides(cfg: TrainConfig, overrides: Dict[str, Any]) -> TrainConfig:
    top_names = {f.name for f in fields(TrainConfig)}
    loss_names = {f.name for f in fields(LossWeights)}
    top_kw: Dict[str, Any] = {}
    loss_kw: Dict[str, Any] = {}
    for k, v in overrides.items():
        if k in top_names:
            top_kw[k] = v
        elif k in loss_names:
            loss_kw[k] = v
        else:
            raise ValueError(f"Unknown config override key: {k!r}")
    if loss_kw:
        top_kw["loss_weights"] = dc_replace(cfg.loss_weights, **loss_kw)
    return dc_replace(cfg, **top_kw)


def build_manifest_dataset(
    manifest: Path,
    *,
    cfg: TrainConfig,
    manifest_root: Optional[Path] = None,
) -> VoiceCloneManifestDataset:
    vocab, context_length = kokoro_vocab_and_context_length(cfg.kokoro_repo_id)
    return VoiceCloneManifestDataset(
        manifest,
        kokoro_repo_id=cfg.kokoro_repo_id,
        vocab=vocab,
        context_length=context_length,
        manifest_root=manifest_root,
        feature_cache_root=cfg.feature_cache_root,
        validate_cache_freshness=cfg.validate_cache_freshness,
    )


def create_train_dataloader(
    dataset: VoiceCloneManifestDataset,
    *,
    cfg: TrainConfig,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = LanguageHomogeneousUniqueSpeakerBatchSampler(
        dataset,
        batch_size=cfg.batch_size,
        generator=generator,
        min_language_speakers=cfg.min_language_speakers,
    )
    if sampler.skipped_languages:
        summary = ", ".join(f"{lang}({count})" for lang, count in sorted(sampler.skipped_languages.items()))
        print(f"Skipping languages with fewer than {cfg.min_language_speakers} speakers: {summary}")
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_voice_clone_batch)


def create_val_dataloader(dataset: VoiceCloneManifestDataset, *, batch_size: int, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_voice_clone_batch)


def _trimmed_input_ids(batch_input_ids: torch.Tensor, lengths: torch.Tensor, i: int) -> torch.Tensor:
    return batch_input_ids[i, : int(lengths[i].item())].unsqueeze(0)


def _forward_batch_outputs(
    kmodel: KModel,
    gst: SegmentGST,
    batch: Dict[str, Any],
    device: torch.device,
    *,
    speed: float,
) -> Tuple[List[KModel.TrainingOutputs], torch.Tensor]:
    ref_hidden_states = batch["ref_hidden_states"].to(device)
    ref_frame_mask = batch["ref_frame_mask"].to(device)
    gst_out, _ = gst(ref_hidden_states, ref_frame_mask)
    outputs: List[KModel.TrainingOutputs] = []
    for i in range(ref_hidden_states.size(0)):
        outputs.append(
            kmodel.forward_with_tokens(
                _trimmed_input_ids(batch["input_ids"].to(device), batch["input_ids_lengths"].to(device), i),
                gst_out.ref_s[i : i + 1],
                speed=speed,
                return_training_outputs=True,
            )
        )
    return outputs, gst_out.ref_s


def _predicted_audio_batch(outputs: Sequence[KModel.TrainingOutputs], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(out.audio.size(-1)) for out in outputs)
    audio = torch.zeros(len(outputs), max_len, device=device)
    lengths = torch.zeros(len(outputs), dtype=torch.long, device=device)
    for i, out in enumerate(outputs):
        n = int(out.audio.size(-1))
        audio[i, :n] = out.audio.squeeze(0)
        lengths[i] = n
    return audio, lengths


def _duration_tensor(outputs: Sequence[KModel.TrainingOutputs], device: torch.device) -> torch.Tensor:
    max_len = max(int(out.duration_logits.size(-1)) for out in outputs)
    durations = torch.zeros(len(outputs), max_len, device=device)
    for i, out in enumerate(outputs):
        n = int(out.duration_logits.size(-1))
        durations[i, :n] = out.duration_logits.squeeze(0)
    return durations


def _f0_tensor(outputs: Sequence[KModel.TrainingOutputs], device: torch.device) -> torch.Tensor:
    max_len = max(int(out.f0_pred.size(-1)) for out in outputs)
    f0 = torch.zeros(len(outputs), max_len, device=device)
    for i, out in enumerate(outputs):
        n = int(out.f0_pred.size(-1))
        f0[i, :n] = out.f0_pred.squeeze(0)
    return f0


def _compute_generator_losses(
    *,
    cfg: TrainConfig,
    mel_loss_mod: MelReconstructionLoss,
    sv_model: WeSpeakerSV,
    outputs: Sequence[KModel.TrainingOutputs],
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    pred_wav, pred_lengths = _predicted_audio_batch(outputs, device)
    tgt_wav = batch["target_wav_24k"].to(device)
    target_lengths = batch["target_lengths"].to(device)
    duration_pred = _duration_tensor(outputs, device)
    f0_pred = _f0_tensor(outputs, device)

    mel_out = mel_loss_mod(pred_wav, tgt_wav, pred_lengths=pred_lengths, target_lengths=target_lengths)
    spk_pred = sv_model(
        pred_wav,
        sampling_rate=24_000,
        waveform_lengths=pred_lengths,
        grad_through_input=True,
        return_frame_features=False,
    ).pooled_embedding
    spk_loss = speaker_contrastive_loss(
        spk_pred,
        batch["target_wespeaker_embedding"].to(device),
        temperature=cfg.contrastive_temperature,
        detach_targets=True,
    )
    dur_loss = duration_loss_log_space(duration_pred, batch["duration_targets"].to(device), batch["duration_mask"].to(device))
    f0_loss = masked_l1_loss(f0_pred, batch["f0_targets"].to(device), batch["f0_mask"].to(device))
    weights = cfg.loss_weights
    total = (
        weights.lambda_mel * mel_out.loss
        + weights.lambda_spk_contrastive * spk_loss
        + weights.lambda_dur * dur_loss
        + weights.lambda_f0 * f0_loss
    )
    metrics = {
        "loss_g": float(total.detach()),
        "loss_mel": float(mel_out.loss.detach()),
        "loss_spk_contrastive": float(spk_loss.detach()),
        "loss_dur": float(dur_loss.detach()),
        "loss_f0": float(f0_loss.detach()),
    }
    return total, metrics, pred_wav


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
    del wandb_run, wandb_num_samples, profile_breakdown, profile_breakdown_steps, torch_profiler_trace, val_dataset
    if config_overrides:
        cfg = _apply_config_overrides(cfg, config_overrides)
    if cfg.batch_size < 2:
        raise ValueError("Contrastive training requires batch_size >= 2")
    if cfg.grad_accum_steps < 1:
        raise ValueError("cfg.grad_accum_steps must be >= 1")
    if cfg.log_interval < 1:
        raise ValueError("cfg.log_interval must be >= 1")
    if cfg.checkpoint_interval < 1:
        raise ValueError("cfg.checkpoint_interval must be >= 1")

    kmodel, gst, _mhubert, sv_model, disc, mel_loss_mod, _kokoro_cfg = build_models(cfg, device)
    params_g = generator_trainable_parameters(kmodel, gst)
    adam_betas = (cfg.adam_b1, cfg.adam_b2)
    opt_g = torch.optim.AdamW(params_g, lr=cfg.lr_g, betas=adam_betas, weight_decay=cfg.weight_decay_g)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg.lr_d, betas=adam_betas, weight_decay=cfg.weight_decay_d)
    scaler_g = GradScaler("cuda", enabled=cfg.use_amp) if device.type == "cuda" else None
    scaler_d = GradScaler("cuda", enabled=cfg.use_amp) if device.type == "cuda" else None

    try:
        total_steps = _effective_optimizer_steps_per_epoch(len(dataloader), cfg.grad_accum_steps) * int(epochs)
    except TypeError:
        total_steps = max_steps
    sched_g = _build_scheduler(opt_g, cfg.warmup_steps, total_steps, cfg.lr_min_g)
    sched_d = _build_scheduler(opt_d, cfg.warmup_steps, total_steps, cfg.lr_min_d)

    start_step = 0
    generator_updates = 0
    discriminator_updates = 0
    if resume is not None:
        resume_state: Dict[str, Any] = {}
        start_step = load_checkpoint(
            resume,
            gst=gst,
            disc=disc,
            kmodel=kmodel,
            opt_g=opt_g,
            opt_d=opt_d,
            device=device,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
            sched_g=sched_g,
            sched_d=sched_d,
            resume_state=resume_state,
        )
        generator_updates = int(resume_state.get("generator_updates", start_step))
        discriminator_updates = int(resume_state.get("discriminator_updates", 0))

    step = start_step
    accum = 0
    opt_g.zero_grad(set_to_none=True)
    opt_d.zero_grad(set_to_none=True)
    for _epoch in range(int(epochs)):
        for batch in dataloader:
            outputs, _ = _forward_batch_outputs(kmodel, gst, batch, device, speed=cfg.speed)
            with _cuda_amp_context(cfg.use_amp, batch["target_wav_24k"].to(device)):
                total_g, metrics, pred_wav = _compute_generator_losses(
                    cfg=cfg,
                    mel_loss_mod=mel_loss_mod,
                    sv_model=sv_model,
                    outputs=outputs,
                    batch=batch,
                    device=device,
                )
            scaled_g = total_g / float(cfg.grad_accum_steps)
            if scaler_g is not None:
                scaler_g.scale(scaled_g).backward()
            else:
                scaled_g.backward()
            accum += 1

            if step >= cfg.disc_start_step:
                real_logits, _ = disc(batch["target_wav_24k"].to(device))
                fake_logits, _ = disc(pred_wav.detach())
                loss_d = discriminator_loss_lsgan(real_logits, fake_logits) / float(cfg.grad_accum_steps)
                if scaler_d is not None:
                    scaler_d.scale(loss_d).backward()
                else:
                    loss_d.backward()
                metrics["loss_d"] = float(loss_d.detach()) * float(cfg.grad_accum_steps)
            else:
                metrics["loss_d"] = 0.0

            if accum < cfg.grad_accum_steps:
                continue

            if scaler_g is not None:
                scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(params_g, cfg.grad_clip_norm_g)
            if scaler_g is not None:
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                opt_g.step()
            sched_g.step()
            opt_g.zero_grad(set_to_none=True)
            generator_updates += 1

            if step >= cfg.disc_start_step:
                if scaler_d is not None:
                    scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), cfg.grad_clip_norm_d)
                if scaler_d is not None:
                    scaler_d.step(opt_d)
                    scaler_d.update()
                else:
                    opt_d.step()
                sched_d.step()
                opt_d.zero_grad(set_to_none=True)
                discriminator_updates += 1

            step += 1
            accum = 0
            if report_callback is not None:
                report_callback(step, metrics["loss_g"])
            if ckpt_dir is not None and step > 0 and step % cfg.checkpoint_interval == 0:
                save_checkpoint(
                    ckpt_dir / f"checkpoint_{step}.pt",
                    gst=gst,
                    disc=disc,
                    kmodel=kmodel,
                    opt_g=opt_g,
                    opt_d=opt_d,
                    step=step,
                    cfg=cfg,
                    scaler_g=scaler_g,
                    scaler_d=scaler_d,
                    sched_g=sched_g,
                    sched_d=sched_d,
                    generator_updates=generator_updates,
                    discriminator_updates=discriminator_updates,
                )
            if max_steps is not None and step >= max_steps:
                if ckpt_dir is not None and cfg.save_final_checkpoint:
                    save_checkpoint(
                        ckpt_dir / f"checkpoint_{step}.pt",
                        gst=gst,
                        disc=disc,
                        kmodel=kmodel,
                        opt_g=opt_g,
                        opt_d=opt_d,
                        step=step,
                        cfg=cfg,
                        scaler_g=scaler_g,
                        scaler_d=scaler_d,
                        sched_g=sched_g,
                        sched_d=sched_d,
                        generator_updates=generator_updates,
                        discriminator_updates=discriminator_updates,
                    )
                return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SegmentGST with cached supervision.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, default=None)
    p.add_argument("--val-manifest", type=Path, default=None)
    p.add_argument("--val-manifest-root", type=Path, default=None)
    p.add_argument("--kokoro-repo", type=str, default=TrainConfig().kokoro_repo_id)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--disc-start-step", type=int, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--save-final-checkpoint", action="store_true")
    p.add_argument("--no-save-final-checkpoint", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--profile-breakdown", action="store_true")
    p.add_argument("--profile-breakdown-steps", type=int, default=3)
    p.add_argument("--torch-profiler-trace", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(kokoro_repo_id=args.kokoro_repo)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps
    if args.warmup_steps is not None:
        cfg.warmup_steps = args.warmup_steps
    if args.disc_start_step is not None:
        cfg.disc_start_step = args.disc_start_step
    if args.checkpoint_interval is not None:
        cfg.checkpoint_interval = args.checkpoint_interval
    if args.amp:
        cfg.use_amp = True
    if args.save_final_checkpoint:
        cfg.save_final_checkpoint = True
    if args.no_save_final_checkpoint:
        cfg.save_final_checkpoint = False

    dataset = build_manifest_dataset(args.manifest, cfg=cfg, manifest_root=args.manifest_root)
    dataloader = create_train_dataloader(dataset, cfg=cfg, num_workers=args.num_workers)
    train_loop(
        dataloader,
        cfg,
        torch.device(args.device),
        epochs=args.epochs,
        max_steps=args.max_steps,
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
        profile_breakdown=args.profile_breakdown,
        profile_breakdown_steps=args.profile_breakdown_steps,
        torch_profiler_trace=args.torch_profiler_trace,
    )


if __name__ == "__main__":
    main()
