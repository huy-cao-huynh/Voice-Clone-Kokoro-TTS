from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_mono_1ch(wav: torch.Tensor) -> torch.Tensor:
    """
    Accept waveform shaped `(B, T)` or `(B, 1, T)` and return `(B, 1, T)`.
    """
    if wav.dim() == 2:
        return wav.unsqueeze(1)
    if wav.dim() == 3 and wav.size(1) == 1:
        return wav
    raise ValueError(f"expected wav shape (B, T) or (B, 1, T), got {tuple(wav.shape)}")


def _crop_to_min_shape(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crop `a` and `b` to the minimum size along each spatial dimension.

    Used for adversarial / feature-matching losses where real and fake audio
    might differ by a few samples.
    """
    if a.dim() != b.dim():
        raise ValueError(f"a and b must have the same dim, got {a.dim()} and {b.dim()}")
    # Keep batch and channel dims intact, crop remaining dims.
    out_a = a
    out_b = b
    for dim in range(2, a.dim()):
        n = min(out_a.size(dim), out_b.size(dim))
        if out_a.size(dim) != n:
            out_a = out_a.narrow(dim, 0, n)
        if out_b.size(dim) != n:
            out_b = out_b.narrow(dim, 0, n)
    return out_a, out_b


class MPDiscriminator(nn.Module):
    """
    HiFi-GAN MPD sub-discriminator.

    Returns:
      - logits: tensor `(B, *)`
      - features: list of intermediate feature maps (for feature matching)
    """

    def __init__(self, period: int, *, use_spectral_norm: bool = False) -> None:
        super().__init__()
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = int(period)

        def maybe_sn(m: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(m) if use_spectral_norm else m

        # Kernel/stride choices follow the original HiFi-GAN discriminator family,
        # but we keep it lightweight (enough for adversarial + FM losses).
        chs = [32, 32, 64, 64, 128, 128, 256, 256]
        layers: List[nn.Module] = []
        in_ch = 1
        for idx, ch in enumerate(chs):
            layers.append(
                maybe_sn(
                    nn.Conv2d(
                        in_ch,
                        ch,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                )
            )
            in_ch = ch
        self.convs = nn.ModuleList(layers)
        self.final_conv = maybe_sn(
            nn.Conv2d(in_ch, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        )

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        wav: `(B, T)` or `(B, 1, T)`
        """
        x = _ensure_mono_1ch(wav)
        b, _, t = x.shape

        # Pad to multiple of period and reshape into 2D patches.
        pad_len = (-t) % self.period
        if pad_len:
            x = F.pad(x, (0, pad_len))
        t = x.size(-1)
        x = x.view(b, 1, t // self.period, self.period)

        feats: List[torch.Tensor] = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2, inplace=True)
            feats.append(x)

        x = self.final_conv(x)
        # Match HiFi-GAN convention: logits are the final conv output (flattened).
        logits = x.flatten(1)
        return logits, feats


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) wrapper.

    Returns:
      - logits: list of tensors, one per period
      - features: list[list[Tensor]], one per period, then per conv layer
    """

    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11), *, use_spectral_norm: bool = False) -> None:
        super().__init__()
        if len(periods) == 0:
            raise ValueError("periods must be non-empty")
        self.periods = list(map(int, periods))
        self.sub_discriminators = nn.ModuleList(
            [MPDiscriminator(p, use_spectral_norm=use_spectral_norm) for p in self.periods]
        )

    def forward(self, wav: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        logits: List[torch.Tensor] = []
        feats: List[List[torch.Tensor]] = []
        for d in self.sub_discriminators:
            l, f = d(wav)
            logits.append(l)
            feats.append(f)
        return logits, feats


class _SubMSDiscriminator(nn.Module):
    """
    HiFi-GAN MSD sub-discriminator (single scale).

    Returns:
      - logits: tensor `(B, *)`
      - features: list of intermediate feature maps (for feature matching)
    """

    def __init__(self, *, use_spectral_norm: bool = False) -> None:
        super().__init__()

        def maybe_sn(m: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(m) if use_spectral_norm else m

        # A compact discriminator stack.
        chs = [16, 64, 256, 1024, 1024]
        convs: List[nn.Module] = []
        in_ch = 1
        for i, ch in enumerate(chs):
            # First conv reduces resolution more aggressively.
            stride = 1 if i == 0 else 2
            convs.append(maybe_sn(nn.Conv1d(in_ch, ch, kernel_size=5, stride=stride, padding=2)))
            in_ch = ch
        self.convs = nn.ModuleList(convs)
        self.final_conv = maybe_sn(nn.Conv1d(in_ch, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = _ensure_mono_1ch(wav)
        feats: List[torch.Tensor] = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2, inplace=True)
            feats.append(x)
        x = self.final_conv(x)
        logits = x.flatten(1)
        return logits, feats


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD) wrapper.

    Standard HiFi-GAN practice uses raw audio plus downsampled variants.
    Here we implement:
      - scale 1: original
      - scale 2: average pooled by 2
      - scale 4: average pooled by 4
    """

    def __init__(self, scales: Sequence[int] = (1, 2, 4), *, use_spectral_norm: bool = False) -> None:
        super().__init__()
        if len(scales) == 0:
            raise ValueError("scales must be non-empty")
        self.scales = list(map(int, scales))
        self.sub_discriminators = nn.ModuleList(
            [_SubMSDiscriminator(use_spectral_norm=use_spectral_norm) for _ in self.scales]
        )

    def forward(self, wav: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        x = _ensure_mono_1ch(wav)
        logits: List[torch.Tensor] = []
        feats: List[List[torch.Tensor]] = []
        for scale, d in zip(self.scales, self.sub_discriminators):
            if scale == 1:
                x_s = x
            else:
                # Downsample by exact factor using average pooling.
                x_s = F.avg_pool1d(x, kernel_size=scale, stride=scale, padding=0)
            l, f = d(x_s)
            logits.append(l)
            feats.append(f)
        return logits, feats


class HiFiGANMPDMSDDiscriminator(nn.Module):
    """
    Convenience wrapper that combines MPD + MSD.

    Forward returns:
      - logits: list of tensors (real/fake LSGAN targets)
      - features: list of feature-map lists for feature matching
        structure: features[subdisc_idx][layer_idx]
    """

    def __init__(
        self,
        *,
        mpd_periods: Sequence[int] = (2, 3, 5, 7, 11),
        msd_scales: Sequence[int] = (1, 2, 4),
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(mpd_periods, use_spectral_norm=use_spectral_norm)
        self.msd = MultiScaleDiscriminator(msd_scales, use_spectral_norm=use_spectral_norm)

    def forward(self, wav: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        mpd_logits, mpd_feats = self.mpd(wav)
        msd_logits, msd_feats = self.msd(wav)
        return mpd_logits + msd_logits, mpd_feats + msd_feats


@dataclass(frozen=True)
class HiFiGANDiscriminatorOutput:
    logits: List[torch.Tensor]
    features: List[List[torch.Tensor]]

