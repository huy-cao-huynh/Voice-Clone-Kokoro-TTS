"""Train SegmentGST against a frozen Kokoro backbone with cached supervision."""

from __future__ import annotations

import argparse
import csv
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
_WANDB_LOCAL_CSV_METRICS = {
    "train/loss_mel": "train_loss_mel.csv",
    "train/loss_spk_contrastive": "train_loss_spk_contrastive.csv",
    "val_free/loss_spk_contrastive": "val_free_loss_spk_contrastive.csv",
    "val_tf/loss_spk_contrastive": "val_tf_loss_spk_contrastive.csv",
}
_KOKORO_DURATION_FRAME_SAMPLES = 600


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
        conv_kernel_size=cfg.gst_conv_kernel_size,
        conv_stride=cfg.gst_conv_stride,
        conv_padding=cfg.gst_conv_padding,
        universal_style_vector=load_universal_style_vector(cfg.universal_style_vector_path, ref_dim=256),
    ).to(device)
    disc = HiFiGANMPDMSDDiscriminator().to(device)
    mel_loss = build_mel_loss(kokoro_cfg, cfg.mel, device)
    return kmodel, gst, mhubert, sv_model, disc, mel_loss, kokoro_cfg


def build_training_models(
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[KModel, SegmentGST, WeSpeakerSV, HiFiGANMPDMSDDiscriminator, MelReconstructionLoss, Dict[str, Any]]:
    kmodel, kokoro_cfg = build_kokoro_model(cfg, device)
    sv_model = WeSpeakerSV.from_checkpoint(
        cfg.wespeaker_checkpoint_path,
        embedding_dim=cfg.wespeaker_embedding_dim,
        sample_rate=cfg.wespeaker_sample_rate,
        device=device,
        dtype=None,
    )
    gst = SegmentGST(
        frame_dim=768,
        embed_dim=cfg.gst_embed_dim,
        num_bases=1024,
        num_heads=4,
        ref_dim=256,
        style_dec_dim=128,
        dropout=cfg.gst_dropout,
        conv_kernel_size=cfg.gst_conv_kernel_size,
        conv_stride=cfg.gst_conv_stride,
        conv_padding=cfg.gst_conv_padding,
        universal_style_vector=load_universal_style_vector(cfg.universal_style_vector_path, ref_dim=256),
    ).to(device)
    disc = HiFiGANMPDMSDDiscriminator().to(device)
    mel_loss = build_mel_loss(kokoro_cfg, cfg.mel, device)
    return kmodel, gst, sv_model, disc, mel_loss, kokoro_cfg


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
    max_rows: Optional[int] = None,
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
        max_rows=max_rows,
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


def _trimmed_gt_durations(batch: Dict[str, Any], lengths: torch.Tensor, i: int, device: torch.device) -> Optional[torch.Tensor]:
    prosody_enabled = batch.get("prosody_enabled")
    if prosody_enabled is None or not bool(prosody_enabled[i]):
        return None
    gt_dur_frames = batch.get("gt_dur_frames")
    if gt_dur_frames is None:
        return None
    token_count = int(lengths[i].item())
    return gt_dur_frames[i, :token_count].to(device=device, dtype=torch.long).unsqueeze(0)


def _gst_projection_diagnostics(gst_out: Any, universal_style_vector: torch.Tensor) -> Dict[str, float]:
    base = universal_style_vector.to(device=gst_out.style_dec.device, dtype=gst_out.style_dec.dtype)
    split = gst_out.style_dec.size(-1)
    dec_proj = gst_out.style_dec.detach() - base[:split].unsqueeze(0)
    pred_proj = gst_out.style_pred.detach() - base[split:].unsqueeze(0)
    return {
        "gst/proj_dec_norm_mean": float(dec_proj.norm(dim=-1).mean().detach()),
        "gst/proj_pred_norm_mean": float(pred_proj.norm(dim=-1).mean().detach()),
    }


def _gst_internals_diagnostics(
    pooled_style: torch.Tensor,
    style_dec: torch.Tensor,
    style_pred: torch.Tensor,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["gst/pooled_style_norm_mean"] = float(pooled_style.norm(dim=-1).mean().detach())
    if pooled_style.size(0) >= 2:
        def _offdiag_cos_mean(x: torch.Tensor) -> float:
            sim = torch.nn.functional.cosine_similarity(x[:, None, :], x[None, :, :], dim=-1)
            mask = ~torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
            return float(sim[mask].mean().detach())
        metrics["gst/pooled_style_pairwise_cos_mean"] = _offdiag_cos_mean(pooled_style)
        metrics["gst/style_dec_pairwise_cos_mean"] = _offdiag_cos_mean(style_dec)
        metrics["gst/style_pred_pairwise_cos_mean"] = _offdiag_cos_mean(style_pred)
    return metrics


def _gst_attn_diagnostics(attn_weights: torch.Tensor) -> Dict[str, float]:
    # attn_weights: (B, T_query, num_bases), softmax output from MHA with average_attn_weights=True
    metrics: Dict[str, float] = {}
    eps = 1e-8
    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1).mean()
    metrics["gst/attn_entropy_mean"] = float(entropy.detach())
    b = attn_weights.size(0)
    top1 = attn_weights.argmax(dim=-1)  # (B, T_query)
    modal_per_item: List[int] = []
    for i in range(b):
        vals, cnts = top1[i].unique(return_counts=True)
        modal_per_item.append(int(vals[cnts.argmax()].item()))
    modal_tensor = torch.tensor(modal_per_item, dtype=torch.long)
    _, modal_counts = modal_tensor.unique(return_counts=True)
    metrics["gst/attn_top1_index_mode_count"] = float(int(modal_counts.max().item()))
    return metrics


def _forward_batch_outputs(
    kmodel: KModel,
    gst: SegmentGST,
    batch: Dict[str, Any],
    device: torch.device,
    *,
    speed: float,
    force_target_total_frames: bool = False,
    style_decoder_only: bool = False,
) -> Tuple[List[KModel.TrainingOutputs], Any]:
    ref_hidden_states = batch["ref_hidden_states"].to(device)
    ref_frame_mask = batch["ref_frame_mask"].to(device)
    input_ids = batch["input_ids"].to(device)
    input_ids_lengths = batch["input_ids_lengths"].to(device)
    target_lengths = batch["target_lengths"].to(device)
    gst_out = gst(
        ref_hidden_states,
        ref_frame_mask,
        need_weights=True,
        use_universal_style_pred=style_decoder_only,
    )
    outputs: List[KModel.TrainingOutputs] = []
    for i in range(ref_hidden_states.size(0)):
        gt_dur_frames = _trimmed_gt_durations(batch, input_ids_lengths, i, device)
        force_total_frames = None
        if force_target_total_frames and gt_dur_frames is None:
            token_count = int(input_ids_lengths[i].item())
            target_samples = int(target_lengths[i].item())
            force_total_frames = max(token_count, int(round(target_samples / float(_KOKORO_DURATION_FRAME_SAMPLES))))
        outputs.append(
            kmodel.forward_with_tokens(
                _trimmed_input_ids(input_ids, input_ids_lengths, i),
                gst_out.ref_s[i : i + 1],
                speed=speed,
                gt_dur_frames=gt_dur_frames,
                force_total_frames=force_total_frames,
                return_training_outputs=True,
            )
        )
    return outputs, gst_out


def _predicted_audio_batch(outputs: Sequence[KModel.TrainingOutputs], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(out.audio.size(-1)) for out in outputs)
    audio = torch.zeros(len(outputs), max_len, device=device)
    lengths = torch.zeros(len(outputs), dtype=torch.long, device=device)
    for i, out in enumerate(outputs):
        n = int(out.audio.size(-1))
        audio[i, :n] = out.audio.squeeze(0)
        lengths[i] = n
    return audio, lengths


def _target_audio_lengths(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if "target_lengths" in batch:
        return batch["target_lengths"].to(device)
    target_wav = batch["target_wav_24k"].to(device)
    return torch.full((target_wav.size(0),), target_wav.size(1), dtype=torch.long, device=device)


def _effective_target_audio_lengths(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    match_teacher_forced_span: bool,
) -> torch.Tensor:
    target_lengths = _target_audio_lengths(batch, device)
    if not match_teacher_forced_span:
        return target_lengths
    prosody_enabled = batch.get("prosody_enabled")
    gt_total_duration_samples = batch.get("gt_total_duration_samples")
    if prosody_enabled is None or gt_total_duration_samples is None:
        return target_lengths
    prosody_enabled = prosody_enabled.to(device)
    forced_lengths = gt_total_duration_samples.to(device=device, dtype=torch.long)
    forced_lengths = torch.minimum(forced_lengths, target_lengths)
    return torch.where(prosody_enabled, forced_lengths, target_lengths)


def _speaker_target_embeddings(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    match_teacher_forced_span: bool,
) -> torch.Tensor:
    targets = batch["target_wespeaker_embedding"].to(device)
    if not match_teacher_forced_span:
        return targets
    prosody_enabled = batch.get("prosody_enabled")
    span_targets = batch.get("target_wespeaker_embedding_tf_span")
    if prosody_enabled is None or span_targets is None:
        return targets
    prosody_mask = prosody_enabled.to(device=device, dtype=torch.bool).unsqueeze(1)
    return torch.where(prosody_mask, span_targets.to(device), targets)


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


def _duration_frame_lengths(outputs: Sequence[KModel.TrainingOutputs], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [int(out.rounded_durations.sum().item()) for out in outputs],
        dtype=torch.long,
        device=device,
    )


def _mean_length_ratio(pred_lengths: torch.Tensor, target_lengths: torch.Tensor) -> float:
    pred = pred_lengths.to(dtype=torch.float32)
    target = target_lengths.to(dtype=torch.float32).clamp_min(1.0)
    return float((pred / target).mean().detach())


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    norms: List[torch.Tensor] = []
    for param in parameters:
        if param.grad is None:
            continue
        norms.append(param.grad.detach().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).detach())


def _style_decoder_only_active(cfg: TrainConfig, step: int) -> bool:
    return int(cfg.style_decoder_only_steps) > 0 and int(step) < int(cfg.style_decoder_only_steps)


def _collapse_diagnostics(ref_s: torch.Tensor, universal_style_vector: torch.Tensor) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if ref_s.dim() != 2:
        raise ValueError(f"ref_s must be shaped (batch, dim), got {tuple(ref_s.shape)}")
    base = universal_style_vector.to(device=ref_s.device, dtype=ref_s.dtype).unsqueeze(0)
    delta = ref_s - base
    ref_s_std = ref_s.std(dim=0, unbiased=False)
    metrics["collapse/ref_s_std_mean"] = float(ref_s_std.mean().detach())
    metrics["collapse/ref_s_delta_norm_mean"] = float(delta.norm(dim=-1).mean().detach())
    if ref_s.size(0) >= 2:
        sim = torch.nn.functional.cosine_similarity(ref_s[:, None, :], ref_s[None, :, :], dim=-1)
        offdiag = sim[~torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)]
        metrics["collapse/ref_s_pairwise_cos_mean"] = float(offdiag.mean().detach())
    else:
        metrics["collapse/ref_s_pairwise_cos_mean"] = 1.0
    return metrics


def _maybe_train_diagnostics(gst_out: Any, gst: SegmentGST) -> Dict[str, float]:
    if not hasattr(gst_out, "ref_s") or not hasattr(gst, "universal_style_vector"):
        return {}
    logs = {f"train/{key}": value for key, value in _collapse_diagnostics(gst_out.ref_s.detach(), gst.universal_style_vector).items()}
    if hasattr(gst_out, "pooled_style") and hasattr(gst_out, "style_dec") and hasattr(gst_out, "style_pred"):
        logs.update({f"train/{key}": value for key, value in _gst_projection_diagnostics(gst_out, gst.universal_style_vector).items()})
        logs.update({f"train/{key}": value for key, value in _gst_internals_diagnostics(
            gst_out.pooled_style.detach(), gst_out.style_dec.detach(), gst_out.style_pred.detach()
        ).items()})
    if hasattr(gst_out, "attn_weights") and gst_out.attn_weights is not None:
        logs.update({f"train/{key}": value for key, value in _gst_attn_diagnostics(gst_out.attn_weights.detach()).items()})
    return logs


def _maybe_val_diagnostics(prefix: str, gst_out: Any, gst: SegmentGST) -> Dict[str, float]:
    if not hasattr(gst_out, "ref_s") or not hasattr(gst, "universal_style_vector"):
        return {}
    logs = {f"{prefix}/{key}": value for key, value in _collapse_diagnostics(gst_out.ref_s, gst.universal_style_vector).items()}
    if prefix == "val_tf" and hasattr(gst_out, "pooled_style") and hasattr(gst_out, "style_dec") and hasattr(gst_out, "style_pred"):
        logs.update({f"{prefix}/{key}": value for key, value in _gst_projection_diagnostics(gst_out, gst.universal_style_vector).items()})
    if hasattr(gst_out, "pooled_style") and hasattr(gst_out, "style_dec") and hasattr(gst_out, "style_pred"):
        logs.update({f"{prefix}/{key}": value for key, value in _gst_internals_diagnostics(
            gst_out.pooled_style, gst_out.style_dec, gst_out.style_pred
        ).items()})
    if hasattr(gst_out, "attn_weights") and gst_out.attn_weights is not None:
        logs.update({f"{prefix}/{key}": value for key, value in _gst_attn_diagnostics(gst_out.attn_weights).items()})
    return logs


def _compute_generator_losses(
    *,
    cfg: TrainConfig,
    mel_loss_mod: MelReconstructionLoss,
    sv_model: WeSpeakerSV,
    outputs: Sequence[KModel.TrainingOutputs],
    batch: Dict[str, Any],
    device: torch.device,
    match_teacher_forced_span: bool,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    pred_wav, pred_lengths = _predicted_audio_batch(outputs, device)
    tgt_wav = batch["target_wav_24k"].to(device)
    target_lengths = _effective_target_audio_lengths(
        batch,
        device,
        match_teacher_forced_span=match_teacher_forced_span,
    )
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
        _speaker_target_embeddings(batch, device, match_teacher_forced_span=match_teacher_forced_span),
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


@contextlib.contextmanager
def _temporary_eval(modules: Sequence[nn.Module]) -> Iterator[None]:
    modes = [m.training for m in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, was_training in zip(modules, modes):
            module.train(was_training)


def _wandb_audio_table(
    *,
    pred_wav_free: torch.Tensor,
    pred_wav_tf: torch.Tensor,
    pred_lengths_free: torch.Tensor,
    pred_lengths_tf: torch.Tensor,
    batch: Dict[str, Any],
    sample_rate: int,
    max_items: int,
    step: int,
) -> Dict[str, Any]:
    try:
        import wandb
    except ImportError:
        return {}
    table = wandb.Table(
        columns=[
            "step",
            "row_index",
            "speaker_id",
            "text",
            "coverage_ratio",
            "gt_audio",
            "pred_audio_free",
            "pred_audio_tf",
            "gt_samples_full",
            "gt_samples_tf",
            "pred_samples_free",
            "pred_samples_tf",
        ]
    )
    texts = batch.get("texts") or []
    speaker_ids = batch.get("speaker_ids") or []
    target_wav = batch["target_wav_24k"]
    target_lengths_free = _target_audio_lengths(batch, pred_lengths_free.device).cpu()
    target_lengths_tf = _effective_target_audio_lengths(
        batch,
        pred_lengths_tf.device,
        match_teacher_forced_span=True,
    ).cpu()
    coverage_ratios = batch.get("duration_coverage_ratio")
    count = min(int(pred_wav_free.size(0)), int(pred_wav_tf.size(0)), int(target_wav.size(0)), int(max_items))
    row_indices = batch.get("row_indices")
    for i in range(count):
        text = texts[i] if i < len(texts) else f"validation_example_{i}"
        speaker_id = speaker_ids[i] if i < len(speaker_ids) else ""
        row_index = int(row_indices[i]) if row_indices is not None else i
        free_len = int(pred_lengths_free[i].item())
        tf_len = int(pred_lengths_tf[i].item())
        gt_free_len = int(target_lengths_free[i].item())
        gt_tf_len = int(target_lengths_tf[i].item())
        coverage = float(coverage_ratios[i]) if coverage_ratios is not None else None
        table.add_data(
            int(step),
            row_index,
            speaker_id,
            text,
            coverage,
            wandb.Audio(target_wav[i, :gt_free_len].detach().float().cpu().numpy(), sample_rate=sample_rate),
            wandb.Audio(pred_wav_free[i, :free_len].detach().float().cpu().numpy(), sample_rate=sample_rate),
            wandb.Audio(pred_wav_tf[i, :tf_len].detach().float().cpu().numpy(), sample_rate=sample_rate),
            gt_free_len,
            gt_tf_len,
            free_len,
            tf_len,
        )
    return {"val/audio_table": table}


def _wandb_run_dir(wandb_run: Any) -> Optional[Path]:
    run_dir = getattr(wandb_run, "dir", None)
    if not run_dir:
        return None
    return Path(run_dir)


def _append_metric_csv(csv_path: Path, *, step: int, value: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("step", "value"))
        if not file_exists:
            writer.writeheader()
        writer.writerow({"step": int(step), "value": float(value)})


def _write_wandb_metric_csvs(wandb_run: Optional[Any], logs: Dict[str, Any], *, step: int) -> None:
    if wandb_run is None:
        return
    run_dir = _wandb_run_dir(wandb_run)
    if run_dir is None:
        return
    for metric_name, filename in _WANDB_LOCAL_CSV_METRICS.items():
        value = logs.get(metric_name)
        if value is None:
            continue
        _append_metric_csv(run_dir / filename, step=step, value=float(value))


def _run_validation_snapshot(
    *,
    cfg: TrainConfig,
    kmodel: KModel,
    gst: SegmentGST,
    sv_model: WeSpeakerSV,
    mel_loss_mod: MelReconstructionLoss,
    batch: Dict[str, Any],
    device: torch.device,
    wandb_num_samples: int,
    step: int,
) -> Dict[str, Any]:
    style_decoder_only = _style_decoder_only_active(cfg, step)
    with _temporary_eval((kmodel, gst, sv_model)):
        with torch.no_grad():
            outputs_free, gst_out_free = _forward_batch_outputs(
                kmodel,
                gst,
                batch,
                device,
                speed=cfg.speed,
                force_target_total_frames=False,
                style_decoder_only=style_decoder_only,
            )
            total_free, metrics_free, pred_wav_free = _compute_generator_losses(
                cfg=cfg,
                mel_loss_mod=mel_loss_mod,
                sv_model=sv_model,
                outputs=outputs_free,
                batch=batch,
                device=device,
                match_teacher_forced_span=False,
            )
            outputs_tf, gst_out_tf = _forward_batch_outputs(
                kmodel,
                gst,
                batch,
                device,
                speed=cfg.speed,
                force_target_total_frames=True,
                style_decoder_only=style_decoder_only,
            )
            total_tf, metrics_tf, pred_wav_tf = _compute_generator_losses(
                cfg=cfg,
                mel_loss_mod=mel_loss_mod,
                sv_model=sv_model,
                outputs=outputs_tf,
                batch=batch,
                device=device,
                match_teacher_forced_span=True,
            )
    pred_lengths_free = _predicted_audio_batch(outputs_free, device)[1]
    pred_lengths_tf = _predicted_audio_batch(outputs_tf, device)[1]
    target_lengths_free = _target_audio_lengths(batch, device)
    target_lengths_tf = _effective_target_audio_lengths(batch, device, match_teacher_forced_span=True)
    logs = {f"val_free/{key}": value for key, value in metrics_free.items()}
    logs["val_free/loss_total"] = float(total_free.detach())
    logs.update({f"val_tf/{key}": value for key, value in metrics_tf.items()})
    logs["val_tf/loss_total"] = float(total_tf.detach())
    logs["val_free/len_ratio"] = _mean_length_ratio(pred_lengths_free, target_lengths_free)
    logs["val_tf/len_ratio"] = _mean_length_ratio(pred_lengths_tf, target_lengths_tf)
    logs["val_gap/loss_mel"] = logs["val_free/loss_mel"] - logs["val_tf/loss_mel"]
    logs["val_gap/loss_total"] = logs["val_free/loss_total"] - logs["val_tf/loss_total"]
    logs["val_gap/len_ratio"] = logs["val_free/len_ratio"] - logs["val_tf/len_ratio"]
    logs.update(_maybe_val_diagnostics("val_free", gst_out_free, gst))
    logs.update(_maybe_val_diagnostics("val_tf", gst_out_tf, gst))
    logs.update(
        _wandb_audio_table(
            pred_wav_free=pred_wav_free,
            pred_wav_tf=pred_wav_tf,
            pred_lengths_free=pred_lengths_free,
            pred_lengths_tf=pred_lengths_tf,
            batch=batch,
            sample_rate=cfg.mel.sample_rate,
            max_items=wandb_num_samples,
            step=step,
        )
    )
    return logs


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
    del profile_breakdown, profile_breakdown_steps, torch_profiler_trace
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

    kmodel, gst, sv_model, disc, mel_loss_mod, _kokoro_cfg = build_training_models(cfg, device)
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
        print(f"Resumed step {start_step} from {resume}")

    fixed_val_batch: Optional[Dict[str, Any]] = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_batch_size = min(int(wandb_num_samples), len(val_dataset))
        fixed_val_batch = next(iter(create_val_dataloader(val_dataset, batch_size=val_batch_size, num_workers=0)))

    try:
        from tqdm.auto import tqdm as _tqdm_cls
    except ImportError:
        _tqdm_cls = None  # type: ignore[misc, assignment]

    step = start_step
    accum = 0
    opt_g.zero_grad(set_to_none=True)
    opt_d.zero_grad(set_to_none=True)
    try:
        expected_steps_per_epoch = _effective_optimizer_steps_per_epoch(len(dataloader), cfg.grad_accum_steps)
    except TypeError:
        expected_steps_per_epoch = None
    for epoch_idx in range(int(epochs)):
        pbar_ctx: Any = contextlib.nullcontext()
        use_tqdm_pbar = False
        if _tqdm_cls is not None and expected_steps_per_epoch is not None:
            remaining_steps = expected_steps_per_epoch
            if max_steps is not None:
                remaining_steps = min(remaining_steps, max(0, int(max_steps) - step))
            if remaining_steps > 0:
                use_tqdm_pbar = True
                pbar_ctx = _tqdm_cls(
                    total=remaining_steps,
                    desc=f"train epoch {epoch_idx + 1}/{int(epochs)}",
                    leave=False,
                    dynamic_ncols=True,
                )

        with pbar_ctx as pbar:
            for batch in dataloader:
                style_decoder_only = _style_decoder_only_active(cfg, step)
                outputs, gst_out = _forward_batch_outputs(
                    kmodel,
                    gst,
                    batch,
                    device,
                    speed=cfg.speed,
                    force_target_total_frames=True,
                    style_decoder_only=style_decoder_only,
                )
                with _cuda_amp_context(cfg.use_amp, batch["target_wav_24k"].to(device)):
                    total_g, metrics, pred_wav = _compute_generator_losses(
                        cfg=cfg,
                        mel_loss_mod=mel_loss_mod,
                        sv_model=sv_model,
                        outputs=outputs,
                        batch=batch,
                        device=device,
                        match_teacher_forced_span=True,
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
                grad_norm_g = _grad_norm(params_g)
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
                if use_tqdm_pbar:
                    postfix = {
                        "loss_g": f"{metrics['loss_g']:.4f}",
                        "mel": f"{metrics['loss_mel']:.4f}",
                    }
                    if "loss_d" in metrics:
                        postfix["d"] = f"{metrics['loss_d']:.4f}"
                    pbar.set_postfix(postfix, refresh=False)
                    pbar.update(1)
                if wandb_run is not None:
                    train_logs = {f"train/{key}": value for key, value in metrics.items()}
                    train_logs["train/grad_norm_g"] = grad_norm_g
                    train_logs["train/lr_g"] = float(opt_g.param_groups[0]["lr"])
                    train_logs.update(_maybe_train_diagnostics(gst_out, gst))
                    wandb_run.log(train_logs, step=step)
                    _write_wandb_metric_csvs(wandb_run, train_logs, step=step)
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
                if fixed_val_batch is not None and step > 0 and step % cfg.checkpoint_interval == 0:
                    val_logs = _run_validation_snapshot(
                        cfg=cfg,
                        kmodel=kmodel,
                        gst=gst,
                        sv_model=sv_model,
                        mel_loss_mod=mel_loss_mod,
                        batch=fixed_val_batch,
                        device=device,
                        wandb_num_samples=wandb_num_samples,
                        step=step,
                    )
                    if wandb_run is not None:
                        wandb_run.log(val_logs, step=step)
                        _write_wandb_metric_csvs(wandb_run, val_logs, step=step)
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
    p.add_argument("--style-decoder-only-steps", type=int, default=None)
    p.add_argument("--gst-conv-kernel-size", type=int, default=None)
    p.add_argument("--gst-conv-stride", type=int, default=None)
    p.add_argument("--gst-conv-padding", type=int, default=None)
    p.add_argument("--lambda-mel", type=float, default=None)
    p.add_argument("--lambda-spk-contrastive", type=float, default=None)
    p.add_argument("--contrastive-temperature", type=float, default=None)
    p.add_argument("--grad-clip-norm-g", type=float, default=None)
    p.add_argument("--save-final-checkpoint", action="store_true")
    p.add_argument("--no-save-final-checkpoint", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
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
    if args.style_decoder_only_steps is not None:
        cfg.style_decoder_only_steps = args.style_decoder_only_steps
    if args.gst_conv_kernel_size is not None:
        cfg.gst_conv_kernel_size = args.gst_conv_kernel_size
    if args.gst_conv_stride is not None:
        cfg.gst_conv_stride = args.gst_conv_stride
    if args.gst_conv_padding is not None:
        cfg.gst_conv_padding = args.gst_conv_padding
    if args.lambda_mel is not None:
        cfg.loss_weights.lambda_mel = args.lambda_mel
    if args.lambda_spk_contrastive is not None:
        cfg.loss_weights.lambda_spk_contrastive = args.lambda_spk_contrastive
    if args.contrastive_temperature is not None:
        cfg.contrastive_temperature = args.contrastive_temperature
    if args.grad_clip_norm_g is not None:
        cfg.grad_clip_norm_g = args.grad_clip_norm_g
    if args.amp:
        cfg.use_amp = True
    if args.save_final_checkpoint:
        cfg.save_final_checkpoint = True
    if args.no_save_final_checkpoint:
        cfg.save_final_checkpoint = False

    dataset = build_manifest_dataset(args.manifest, cfg=cfg, manifest_root=args.manifest_root)
    val_dataset = None
    if args.val_manifest is not None:
        val_dataset = build_manifest_dataset(args.val_manifest, cfg=cfg, manifest_root=args.val_manifest_root)
    dataloader = create_train_dataloader(dataset, cfg=cfg, num_workers=args.num_workers)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb logging requested but wandb is not installed") from exc
        init_kwargs: Dict[str, Any] = {"config": asdict(cfg)}
        if args.wandb_project:
            init_kwargs["project"] = args.wandb_project
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        wandb_run = wandb.init(**init_kwargs)
    train_loop(
        dataloader,
        cfg,
        torch.device(args.device),
        epochs=args.epochs,
        max_steps=args.max_steps,
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
        wandb_run=wandb_run,
        val_dataset=val_dataset,
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
