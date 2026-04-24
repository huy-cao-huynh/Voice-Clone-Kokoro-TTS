"""Mel, contrastive speaker, prosody, and GAN helper losses."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


@dataclass
class MelLossOutput:
    loss: torch.Tensor
    mel_pred: torch.Tensor
    mel_target: torch.Tensor


def _min_time_crop(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def speaker_input_mel_from_waveform(
    waveforms: torch.Tensor,
    *,
    mel_transform: nn.Module,
    amp_enabled: bool,
    disable_amp_for_stft: bool,
) -> torch.Tensor:
    if waveforms.dim() != 2:
        raise ValueError(f"waveforms must be `(batch, time)`, got {tuple(waveforms.shape)}")
    if disable_amp_for_stft and waveforms.is_cuda:
        with autocast("cuda", enabled=False):
            mel = mel_transform(waveforms.float())
        return mel.to(dtype=waveforms.dtype)
    ctx = autocast("cuda", enabled=amp_enabled) if waveforms.is_cuda else contextlib.nullcontext()
    with ctx:
        return mel_transform(waveforms)


class MelReconstructionLoss(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        log_floor: float = 1e-5,
        l1_weight: float = 1.0,
        l2_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.log_floor = log_floor
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            center=True,
            power=1.0,
        )

    def forward(
        self,
        pred_wav: torch.Tensor,
        target_wav: torch.Tensor,
        *,
        pred_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> MelLossOutput:
        if pred_wav.dim() != 2 or target_wav.dim() != 2:
            raise ValueError("expected pred_wav and target_wav shaped (batch, time)")

        if pred_lengths is not None and target_lengths is not None:
            items: List[torch.Tensor] = []
            first_p: Optional[torch.Tensor] = None
            first_t: Optional[torch.Tensor] = None
            for i in range(pred_wav.size(0)):
                valid = min(int(pred_lengths[i].item()), int(target_lengths[i].item()))
                p_i = pred_wav[i : i + 1, :valid]
                t_i = target_wav[i : i + 1, :valid]
                mp = torch.log(self.mel(p_i).clamp_min(self.log_floor))
                mt = torch.log(self.mel(t_i).clamp_min(self.log_floor))
                item = self.l1_weight * (mp - mt).abs().mean() + self.l2_weight * ((mp - mt) ** 2).mean()
                items.append(item)
                if first_p is None:
                    first_p, first_t = mp, mt
            return MelLossOutput(loss=sum(items) / len(items), mel_pred=first_p, mel_target=first_t)

        pred_wav, target_wav = _min_time_crop(pred_wav, target_wav)
        mel_p = torch.log(self.mel(pred_wav).clamp_min(self.log_floor))
        mel_t = torch.log(self.mel(target_wav).clamp_min(self.log_floor))
        loss = self.l1_weight * (mel_p - mel_t).abs().mean() + self.l2_weight * ((mel_p - mel_t) ** 2).mean()
        return MelLossOutput(loss=loss, mel_pred=mel_p, mel_target=mel_t)


def speaker_contrastive_loss(
    predicted_embeddings: torch.Tensor,
    cached_target_embeddings: torch.Tensor,
    *,
    temperature: float = 0.07,
    detach_targets: bool = True,
) -> torch.Tensor:
    if predicted_embeddings.dim() != 2 or cached_target_embeddings.dim() != 2:
        raise ValueError("predicted_embeddings and cached_target_embeddings must be `(batch, dim)`")
    if predicted_embeddings.shape != cached_target_embeddings.shape:
        raise ValueError("predicted_embeddings and cached_target_embeddings must have the same shape")
    if predicted_embeddings.size(0) < 2:
        raise ValueError("Contrastive speaker loss requires batch size >= 2")

    anchors = F.normalize(predicted_embeddings, dim=-1)
    targets = cached_target_embeddings.detach() if detach_targets else cached_target_embeddings
    targets = F.normalize(targets, dim=-1)
    logits = anchors @ targets.transpose(0, 1)
    logits = logits / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def duration_loss_log_space(
    duration_logits: torch.Tensor,
    duration_targets: torch.Tensor,
    duration_mask: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    pred = torch.log(duration_logits.clamp_min(0.0) + 1.0 + eps)
    target = torch.log(duration_targets.clamp_min(0.0) + 1.0 + eps)
    mask = duration_mask.to(dtype=pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=prediction.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return ((prediction - target).abs() * mask_f).sum() / denom


def _as_list(x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _crop_logits_to_min_last_dim(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.dim() != b.dim():
        raise ValueError(f"a and b must have the same dim, got {a.dim()} and {b.dim()}")
    if a.shape[:-1] != b.shape[:-1]:
        raise ValueError("a and b must match on all non-time dims before cropping")
    n = min(a.size(-1), b.size(-1))
    return a[..., :n], b[..., :n]


def _crop_to_min_shape(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.dim() != b.dim():
        raise ValueError(f"a and b must have the same dim, got {a.dim()} and {b.dim()}")
    if a.shape[:2] != b.shape[:2]:
        raise ValueError("a and b must match on batch/channel dims before cropping")
    out_a = a
    out_b = b
    for dim in range(2, a.dim()):
        n = min(out_a.size(dim), out_b.size(dim))
        out_a = out_a.narrow(dim, 0, n)
        out_b = out_b.narrow(dim, 0, n)
    return out_a, out_b


def discriminator_loss_lsgan(
    real_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
    fake_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
) -> torch.Tensor:
    r_list = _as_list(real_logits)
    f_list = _as_list(fake_logits)
    if len(r_list) != len(f_list):
        raise ValueError("real_logits and fake_logits must have the same number of discriminator outputs")
    losses = []
    for r, f in zip(r_list, f_list):
        r, f = _crop_logits_to_min_last_dim(r, f)
        losses.append(((r - 1.0) ** 2).mean() + (f**2).mean())
    return sum(losses) / len(losses)


def generator_loss_lsgan(fake_logits: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    f_list = _as_list(fake_logits)
    return sum(((f - 1.0) ** 2).mean() for f in f_list) / len(f_list)


def feature_matching_loss(
    real_features: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]],
    fake_features: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]],
) -> torch.Tensor:
    def _normalize(feats: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]]):
        if len(feats) == 0:
            return []
        return [list(feats)] if isinstance(feats[0], torch.Tensor) else [list(inner) for inner in feats]  # type: ignore[index]

    r_feats = _normalize(real_features)
    f_feats = _normalize(fake_features)
    if len(r_feats) != len(f_feats):
        raise ValueError("real_features and fake_features must have the same number of discriminators")
    losses: List[torch.Tensor] = []
    for r_layers, f_layers in zip(r_feats, f_feats):
        if len(r_layers) != len(f_layers):
            raise ValueError("real_features and fake_features must have the same number of layers")
        for r, f in zip(r_layers, f_layers):
            r, f = _crop_to_min_shape(r, f)
            losses.append((r - f).abs().mean())
    if not losses:
        raise ValueError("feature lists must contain at least one feature map")
    return sum(losses) / len(losses)
