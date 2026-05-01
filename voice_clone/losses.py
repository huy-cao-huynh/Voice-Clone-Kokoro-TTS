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
    """Multi-resolution STFT loss with mel-scale magnitude warping.

    Wraps `auraloss.freq.MultiResolutionSTFTLoss` (Yamamoto et al., 2019) configured
    with `scale="mel"` so the per-resolution log-magnitude term is computed against a
    mel-warped magnitude spectrum at each FFT size. The class name and constructor
    keyword signature are preserved from the previous single-resolution log-mel L1
    implementation so existing call sites (`voice_clone/train_adapters.py`) and
    `MelLossOutput` consumers do not need to change. `n_fft / hop_length / win_length /
    f_min / f_max` are now consumed only by the kept viz spectrogram used to populate
    `MelLossOutput.mel_pred` / `mel_target`; the actual loss uses the multi-resolution
    schedule below. `l1_weight` / `l2_weight` are accepted for backwards compatibility
    but ignored: auraloss already uses an L1 distance internally.
    """

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
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (50, 120, 240),
        win_lengths: Sequence[int] = (240, 600, 1200),
        **_unused: object,
    ) -> None:
        super().__init__()
        self.log_floor = log_floor
        self.fft_sizes = tuple(int(x) for x in fft_sizes)
        self.hop_sizes = tuple(int(x) for x in hop_sizes)
        self.win_lengths = tuple(int(x) for x in win_lengths)
        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths)):
            raise ValueError(
                "fft_sizes, hop_sizes, and win_lengths must all have the same length"
            )
        if int(n_mels) > min(self.fft_sizes):
            raise ValueError(
                f"n_mels={n_mels} must be <= smallest fft_size={min(self.fft_sizes)}"
            )

        import auraloss
        import torchaudio

        self.mr_stft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=list(self.fft_sizes),
            hop_sizes=list(self.hop_sizes),
            win_lengths=list(self.win_lengths),
            scale="mel",
            n_bins=int(n_mels),
            sample_rate=int(sample_rate),
            perceptual_weighting=False,
        )

        self.mel_viz = torchaudio.transforms.MelSpectrogram(
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

    def _viz_log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        # STFT in fp16 is numerically unstable (inf/nan even on well-scaled audio),
        # so we force fp32 here regardless of the surrounding autocast region.
        ctx = autocast("cuda", enabled=False) if wav.is_cuda else contextlib.nullcontext()
        with ctx:
            mel = self.mel_viz(wav.float())
        return torch.log(mel.clamp_min(self.log_floor))

    def _mr_stft_pair(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # auraloss STFTLoss expects (batch, channels, time); waveforms are (batch, time).
        # Force fp32 to keep the STFT + mel-warp + log path numerically stable under AMP.
        ctx = autocast("cuda", enabled=False) if pred.is_cuda else contextlib.nullcontext()
        with ctx:
            return self.mr_stft(pred.float().unsqueeze(1), target.float().unsqueeze(1))

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
                items.append(self._mr_stft_pair(p_i, t_i))
                if first_p is None:
                    first_p = self._viz_log_mel(p_i)
                    first_t = self._viz_log_mel(t_i)
            return MelLossOutput(loss=sum(items) / len(items), mel_pred=first_p, mel_target=first_t)

        pred_wav, target_wav = _min_time_crop(pred_wav, target_wav)
        loss = self._mr_stft_pair(pred_wav, target_wav)
        mel_p = self._viz_log_mel(pred_wav)
        mel_t = self._viz_log_mel(target_wav)
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
    if prediction.dim() != target.dim() or prediction.dim() != mask.dim():
        raise ValueError("prediction, target, and mask must have the same rank")
    if prediction.shape[:-1] != target.shape[:-1] or prediction.shape[:-1] != mask.shape[:-1]:
        raise ValueError("prediction, target, and mask must match on all non-time dimensions")
    n = min(prediction.size(-1), target.size(-1), mask.size(-1))
    prediction = prediction[..., :n]
    target = target[..., :n]
    mask = mask[..., :n]
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
