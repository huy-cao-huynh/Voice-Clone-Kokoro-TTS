"""Mel reconstruction, speaker cosine (WavLM-SV), and SLM feature GAN (D on encoder frames only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MelLossOutput:
    loss: torch.Tensor
    mel_pred: torch.Tensor
    mel_target: torch.Tensor


def _min_time_crop(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop last dim to ``min(a.size(-1), b.size(-1))``."""
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


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

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> MelLossOutput:
        """``pred_wav`` / ``target_wav``: shape ``(batch, time)`` mono."""
        if pred_wav.dim() != 2 or target_wav.dim() != 2:
            raise ValueError("expected pred_wav and target_wav shaped (batch, time)")
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


class SLMFeatureDiscriminator(nn.Module):
    """Temporal conv stack on frozen WavLM frame features ``(B, T, C)`` (StyleTTS2-style SLM on features)."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_channels: int = 256,
        num_layers: int = 4,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length conv")
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            nn.LayerNorm(in_dim),
            nn.Conv1d(in_dim, hidden_channels, kernel_size=kernel_size, padding=pad),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=pad),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers.append(nn.Conv1d(hidden_channels, 1, kernel_size=kernel_size, padding=pad))
        self.net = nn.Sequential(*layers)

    def forward(self, frame_features: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """Returns a **scalar logit per batch item** (masked mean over time).

        Args:
            frame_features: ``(batch, time, in_dim)``
            frame_mask: ``(batch, time)`` with ``1`` / ``0`` (or float weights).
        """
        if frame_features.dim() != 3:
            raise ValueError("frame_features must be (batch, time, dim)")
        x = frame_features.transpose(1, 2)
        logits_t = self.net(x).squeeze(1)
        m = frame_mask.to(dtype=logits_t.dtype, device=logits_t.device)
        if m.dim() != 2:
            raise ValueError("frame_mask must be (batch, time)")
        w = m.sum(dim=1).clamp(min=1e-6)
        pooled = (logits_t * m).sum(dim=1) / w
        return pooled


def slm_discriminator_loss_hinge(
    disc: SLMFeatureDiscriminator,
    real_feats: torch.Tensor,
    fake_feats: torch.Tensor,
    real_mask: torch.Tensor,
    fake_mask: torch.Tensor,
) -> torch.Tensor:
    """Hinge loss training **D** only on WavLM frame tensors (already detached from G when needed)."""
    d_real = disc(real_feats, real_mask)
    d_fake = disc(fake_feats, fake_mask)
    loss_real = F.relu(1.0 - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    return loss_real + loss_fake


def slm_generator_loss_hinge(
    disc: SLMFeatureDiscriminator,
    fake_feats: torch.Tensor,
    fake_mask: torch.Tensor,
) -> torch.Tensor:
    """Generator side: encourage **D** to output positive on generated features."""
    d_fake = disc(fake_feats, fake_mask)
    return (-d_fake).mean()
