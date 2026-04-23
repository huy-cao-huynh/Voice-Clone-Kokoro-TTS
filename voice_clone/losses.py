"""Mel reconstruction, speaker cosine, and waveform-GAN helper losses."""

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
    """Crop last dim to ``min(a.size(-1), b.size(-1))``."""
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def speaker_input_mel_from_waveform(
    waveforms: torch.Tensor,
    *,
    mel_transform: nn.Module,
    amp_enabled: bool,
    disable_amp_for_stft: bool,
) -> torch.Tensor:
    """Compute mel input for frozen speaker frontend while preserving gradients to waveform."""
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
    """L1 (and optional L2) on log mel spectrograms."""

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
        """``pred_wav`` / ``target_wav``: shape ``(batch, time)`` mono.

        When ``pred_lengths`` and ``target_lengths`` are supplied (variable-length
        batches), the loss is computed per-item on valid (unpadded) slices and
        averaged across the batch.
        """
        if pred_wav.dim() != 2 or target_wav.dim() != 2:
            raise ValueError("expected pred_wav and target_wav shaped (batch, time)")

        if pred_lengths is not None and target_lengths is not None:
            B = pred_wav.size(0)
            item_losses: List[torch.Tensor] = []
            first_mel_p: Optional[torch.Tensor] = None
            first_mel_t: Optional[torch.Tensor] = None
            for i in range(B):
                vlen = min(int(pred_lengths[i].item()), int(target_lengths[i].item()))
                p_i = pred_wav[i : i + 1, :vlen]
                t_i = target_wav[i : i + 1, :vlen]
                mp = torch.log(self.mel(p_i).clamp_min(self.log_floor))
                mt = torch.log(self.mel(t_i).clamp_min(self.log_floor))
                l1 = (mp - mt).abs().mean()
                l2 = ((mp - mt) ** 2).mean()
                item_losses.append(self.l1_weight * l1 + self.l2_weight * l2)
                if first_mel_p is None:
                    first_mel_p, first_mel_t = mp, mt
            loss = sum(item_losses) / B
            return MelLossOutput(loss=loss, mel_pred=first_mel_p, mel_target=first_mel_t)

        pred_wav, target_wav = _min_time_crop(pred_wav, target_wav)
        mel_p = torch.log(self.mel(pred_wav).clamp_min(self.log_floor))
        mel_t = torch.log(self.mel(target_wav).clamp_min(self.log_floor))
        l1 = (mel_p - mel_t).abs().mean()
        l2 = ((mel_p - mel_t) ** 2).mean()
        loss = self.l1_weight * l1 + self.l2_weight * l2
        return MelLossOutput(loss=loss, mel_pred=mel_p, mel_target=mel_t)


def speaker_cosine_loss(
    emb_ref: torch.Tensor,
    emb_gen: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """``1 - cos(a, b)`` with optional L2 re-normalization (HF embeddings are usually pre-normalized)."""
    a = F.normalize(emb_ref, dim=-1, eps=eps)
    b = F.normalize(emb_gen, dim=-1, eps=eps)
    cos = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean()


def _as_list(x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _crop_logits_to_min_last_dim(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crop `(B, N)` logits tensors to the minimum last-dim length.
    """
    if a.dim() != b.dim():
        raise ValueError(f"a and b must have the same dim, got {a.dim()} and {b.dim()}")
    if a.shape[:-1] != b.shape[:-1]:
        raise ValueError(
            "a and b must match on all non-time dims before cropping, "
            f"got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    n = min(a.size(-1), b.size(-1))
    if a.size(-1) != n:
        a = a[..., :n]
    if b.size(-1) != n:
        b = b[..., :n]
    return a, b


def _crop_to_min_shape(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crop `a` and `b` to the minimum size along each spatial dimension.
    Keeps batch/channel dims intact.
    """
    if a.dim() != b.dim():
        raise ValueError(f"a and b must have the same dim, got {a.dim()} and {b.dim()}")
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(
            "a and b must match on batch/channel dims before cropping, "
            f"got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    out_a = a
    out_b = b
    for dim in range(2, a.dim()):
        n = min(out_a.size(dim), out_b.size(dim))
        if out_a.size(dim) != n:
            out_a = out_a.narrow(dim, 0, n)
        if out_b.size(dim) != n:
            out_b = out_b.narrow(dim, 0, n)
    return out_a, out_b


def discriminator_loss_lsgan(
    real_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
    fake_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
) -> torch.Tensor:
    """
    LSGAN discriminator loss for HiFi-GAN style discriminators.

    Supports either a single logits tensor or a list/tuple of logits tensors.
    """
    r_list = _as_list(real_logits)
    f_list = _as_list(fake_logits)
    if len(r_list) == 0 or len(f_list) == 0:
        raise ValueError("real_logits and fake_logits must be non-empty")
    if len(r_list) != len(f_list):
        raise ValueError(
            "real_logits and fake_logits must have the same number of discriminator outputs, "
            f"got {len(r_list)} and {len(f_list)}"
        )

    losses: List[torch.Tensor] = []
    for i in range(len(r_list)):
        r, f = _crop_logits_to_min_last_dim(r_list[i], f_list[i])
        losses.append(((r - 1.0) ** 2).mean() + (f**2).mean())
    return sum(losses) / len(losses)


def generator_loss_lsgan(
    fake_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
) -> torch.Tensor:
    """
    LSGAN generator loss for HiFi-GAN style discriminators.
    Encourages fake logits to match the "real" target (= 1).
    """
    f_list = _as_list(fake_logits)
    if len(f_list) == 0:
        raise ValueError("fake_logits must be non-empty")
    losses: List[torch.Tensor] = []
    for f in f_list:
        # If logits come in mismatched shapes, we only have one tensor here.
        losses.append(((f - 1.0) ** 2).mean())
    return sum(losses) / len(losses)


def feature_matching_loss(
    real_features: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]],
    fake_features: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]],
) -> torch.Tensor:
    """
    Feature matching (HiFi-GAN style) using intermediate discriminator feature maps.

    Expected structure:
      - `real_features` and `fake_features` are lists over discriminators/sub-discriminators
      - each item is a list of layer feature maps.

    The function also tolerates a "single discriminator" case where you pass a flat list
    of tensors instead of a nested list (it will be treated as a 1-element outer list).
    """

    def _normalize_feats(feats: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]]):
        if len(feats) == 0:
            return []
        first = feats[0]
        if isinstance(first, torch.Tensor):
            # Flat list: [T1, T2, ...] -> [[T1, T2, ...]]
            return [list(feats)]  # type: ignore[list-item]
        # Nested list: [[...], [...], ...]
        return [list(inner) for inner in feats]  # type: ignore[arg-type]

    r_feats = _normalize_feats(real_features)
    f_feats = _normalize_feats(fake_features)
    if len(r_feats) == 0 or len(f_feats) == 0:
        raise ValueError("real_features and fake_features must be non-empty")
    if len(r_feats) != len(f_feats):
        raise ValueError(
            "real_features and fake_features must have the same number of discriminators, "
            f"got {len(r_feats)} and {len(f_feats)}"
        )

    losses: List[torch.Tensor] = []
    for di in range(len(r_feats)):
        r_layers = r_feats[di]
        f_layers = f_feats[di]
        if len(r_layers) != len(f_layers):
            raise ValueError(
                "real_features and fake_features must have the same number of layers per discriminator, "
                f"got {len(r_layers)} and {len(f_layers)} for discriminator index {di}"
            )
        for li in range(len(r_layers)):
            r, f = _crop_to_min_shape(r_layers[li], f_layers[li])
            losses.append((r - f).abs().mean())
    if len(losses) == 0:
        raise ValueError("feature lists must contain at least one feature map")
    return sum(losses) / len(losses)
