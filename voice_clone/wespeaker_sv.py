"""Frozen WeSpeaker-style speaker frontend with differentiable mel path."""

from __future__ import annotations

import sys
import yaml
from unittest import mock
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Prevent crash from s3prl calling removed torchaudio backend function
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None

if "torchaudio.sox_effects" not in sys.modules:
    sys.modules["torchaudio.sox_effects"] = mock.MagicMock()

try:
    from wespeaker.models.speaker_model import get_speaker_model
    from wespeaker.utils.checkpoint import load_checkpoint
except ImportError:
    raise ImportError(
        "The 'wespeaker' package is required for WeSpeakerSV but not found. "
        "Please install it with: `pip install wespeaker`"
    )

Tensor = torch.Tensor

@dataclass
class WeSpeakerSVOutput:
    """Outputs from `WeSpeakerSV.forward` / `forward_from_mel`."""

    pooled_embedding: Tensor
    """L2-normalized utterance embedding, shape `(batch, embedding_dim)`."""

    frame_features: Optional[Tensor]
    """Optional frame-level features, shape `(batch, frames, channels)`."""

class WeSpeakerToolkitEncoder(nn.Module):
    """
    Wraps a ``wespeaker`` toolkit speaker backbone (e.g. ``ResNet34``).

    Input is ``(batch, 1, n_mels, time)``; internally converted to the toolkit
    layout ``(batch, time, n_mels)``.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"expected x shaped (B, 1, n_mels, T), got {tuple(x.shape)}")
        # WeSpeaker expects (batch, time, freq)
        x_bt_f = x.squeeze(1).permute(0, 2, 1).contiguous()
        
        if not hasattr(self.backbone, "get_frame_level_feat"):
            raise TypeError("backbone must define get_frame_level_feat (WeSpeaker ResNet family)")
        
        # WeSpeaker ResNet implementation: 
        # get_frame_level_feat returns the frame features (B, T, C)
        # _get_frame_level_feat returns the output before pooling (B, C, F, T or similar)
        frame = self.backbone.get_frame_level_feat(x_bt_f)
        out = self.backbone._get_frame_level_feat(x_bt_f)
        stats = self.backbone.pool(out)
        pooled = self.backbone.seg_1(stats)
        return pooled, frame

def load_wespeaker_toolkit_encoder(
    model_dir: Path,
    *,
    embedding_dim: int,
) -> WeSpeakerToolkitEncoder:
    """
    Loads a WeSpeaker backbone from a directory containing ``config.yaml`` and ``avg_model.pt``.
    
    Throws FileNotFoundError if files are missing.
    """
    cfg_path = model_dir / "config.yaml"
    pt_path = model_dir / "avg_model.pt"
    
    if not cfg_path.is_file():
        raise FileNotFoundError(f"WeSpeaker config not found at {cfg_path}")
    if not pt_path.is_file():
        raise FileNotFoundError(f"WeSpeaker weights not found at {pt_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Simplified loading logic relying on get_speaker_model
    model_name = config.get("model", "ResNet34")
    model_args = config.get("model_args", {})
    
    model = get_speaker_model(model_name)(**model_args)
    
    cfg_embed = int(model_args.get("embed_dim", embedding_dim))
    if cfg_embed != int(embedding_dim):
        raise ValueError(
            f"config.yaml embed_dim={cfg_embed} does not match requested embedding_dim={embedding_dim}"
        )

    load_checkpoint(model, str(pt_path))
    model.eval()
    return WeSpeakerToolkitEncoder(model)

def _resolve_wespeaker_model_dir(checkpoint_path: str) -> Path:
    """Directory that should contain ``config.yaml`` / ``avg_model.pt``."""
    p = Path(checkpoint_path).expanduser()
    if p.is_dir():
        return p
    return p.parent

class WeSpeakerSV(nn.Module):
    """Frozen WeSpeaker wrapper with differentiable mel frontend."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 20.0,
        f_max: float = 7600.0,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.sample_rate = int(sample_rate)
        self.embedding_dim = int(embedding_dim)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            center=True,
            power=2.0,
            normalized=False,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    def train(self, mode: bool = True) -> "WeSpeakerSV":
        super().train(mode)
        # Keep speaker encoder frozen regardless of external train/eval toggles.
        self.encoder.eval()
        return self

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        embedding_dim: int = 256,
        sample_rate: int = 16_000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "WeSpeakerSV":
        """
        Creates a WeSpeakerSV from a checkpoint path or directory.
        Strictly requires the 'wespeaker' package and valid toolkit files.
        """
        model_dir = _resolve_wespeaker_model_dir(checkpoint_path)
        encoder = load_wespeaker_toolkit_encoder(model_dir, embedding_dim=embedding_dim)

        model = cls(
            encoder,
            sample_rate=sample_rate,
            embedding_dim=embedding_dim,
        )
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        model.encoder.requires_grad_(False)
        model.encoder.eval()
        return model

    def _prepare_waveforms(
        self,
        waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
    ) -> Tensor:
        if isinstance(waveforms, Tensor):
            w = waveforms
            if w.dim() == 1:
                w = w.unsqueeze(0)
            if w.dim() != 2:
                raise ValueError("waveforms must be `(batch, samples)` or `(samples,)`")
            return w.to(device=self.device, dtype=self.dtype)

        rows = []
        max_len = 0
        for i, x in enumerate(waveforms):
            row = x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=torch.float32)
            if row.dim() != 1:
                raise ValueError(f"waveforms[{i}] must be 1-D, got {tuple(row.shape)}")
            row = row.to(device=self.device, dtype=self.dtype)
            rows.append(row)
            max_len = max(max_len, int(row.numel()))
        if not rows:
            raise ValueError("Empty waveform batch")
        padded = torch.zeros(len(rows), max_len, device=self.device, dtype=self.dtype)
        for i, row in enumerate(rows):
            padded[i, : row.numel()] = row
        return padded

    def _waveforms_to_mel(self, waveforms: Tensor) -> Tensor:
        mel_tf = self.mel_transform.to(device=waveforms.device, dtype=waveforms.dtype)
        db_tf = self.amplitude_to_db.to(device=waveforms.device, dtype=waveforms.dtype)
        mel = mel_tf(waveforms)  # (B, n_mels, T)
        mel = db_tf(mel)
        # CMS
        mel = (mel - mel.mean(dim=-1, keepdim=True)) / (mel.std(dim=-1, keepdim=True) + 1e-5)
        return mel

    def forward_from_mel(
        self,
        mel: Tensor,
        *,
        normalize_embeddings: bool = True,
        grad_through_input: bool = True,
        return_frame_features: bool = True,
    ) -> WeSpeakerSVOutput:
        if mel.dim() != 3:
            raise ValueError(f"mel must be `(batch, n_mels, frames)`, got {tuple(mel.shape)}")
        mel = mel.to(device=self.device, dtype=self.dtype)
        x = mel.unsqueeze(1)  # (B, 1, n_mels, frames)
        if grad_through_input:
            pooled, frame = self.encoder(x)
        else:
            with torch.no_grad():
                pooled, frame = self.encoder(x)
        emb = F.normalize(pooled, dim=-1) if normalize_embeddings else pooled
        return WeSpeakerSVOutput(
            pooled_embedding=emb,
            frame_features=frame if return_frame_features else None,
        )

    def forward(
        self,
        waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
        *,
        sampling_rate: int = 16_000,
        normalize_embeddings: bool = True,
        grad_through_input: bool = False,
        return_frame_features: bool = True,
    ) -> WeSpeakerSVOutput:
        if sampling_rate != self.sample_rate:
            # Differentiable resampling for generated waveforms.
            wav = self._prepare_waveforms(waveforms)
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=sampling_rate,
                new_freq=self.sample_rate,
            )
        else:
            wav = self._prepare_waveforms(waveforms)

        mel = self._waveforms_to_mel(wav)
        return self.forward_from_mel(
            mel,
            normalize_embeddings=normalize_embeddings,
            grad_through_input=grad_through_input,
            return_frame_features=return_frame_features,
        )
