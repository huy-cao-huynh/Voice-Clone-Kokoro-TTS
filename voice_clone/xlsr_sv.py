"""Frozen Hugging Face `facebook/wav2vec2-xls-r-300m` for SV-like embeddings and encoder frames.

Exposes intermediate-layer frame hidden states (configurable layer index), a frame mask, and a
normalized pooled embedding, all at 16 kHz input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


Tensor = torch.Tensor


def _waveforms_use_torch_preprocess(
    waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
) -> bool:
    """True for tensor batches; keeps preprocessing on-device and autograd-safe."""
    if isinstance(waveforms, Tensor):
        return True
    if not isinstance(waveforms, (list, tuple)) or not waveforms:
        return False
    if not all(isinstance(x, Tensor) for x in waveforms):
        return False
    dev0 = waveforms[0].device
    return all(x.device == dev0 for x in waveforms)


def _zero_mean_unit_var_norm_torch(
    input_values: Tensor,
    attention_mask: Tensor,
    padding_value: float,
    eps: float = 1e-7,
) -> Tensor:
    """Match Hugging Face `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm` (batched)."""
    mask = attention_mask.to(dtype=input_values.dtype)
    lengths = attention_mask.sum(dim=-1).clamp(min=1)
    sum_v = (input_values * mask).sum(dim=-1)
    mean = sum_v / lengths.to(dtype=input_values.dtype)
    centered = input_values - mean.unsqueeze(-1)
    var = ((centered * centered) * mask).sum(dim=-1) / lengths.to(dtype=input_values.dtype)
    normed = centered / torch.sqrt(var.unsqueeze(-1) + eps)
    inv_mask = 1.0 - mask
    return normed * mask + inv_mask * padding_value


def wav2vec2_preprocess_batch(
    feature_extractor: Wav2Vec2FeatureExtractor,
    waveforms: Union[Tensor, Sequence[Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    sampling_rate: int,
) -> Tuple[Tensor, Tensor]:
    """Pad/normalize waveforms like `Wav2Vec2FeatureExtractor(..., return_tensors='pt')`."""
    if sampling_rate != feature_extractor.sampling_rate:
        raise ValueError(
            f"Expected sampling_rate={feature_extractor.sampling_rate} for this feature extractor; "
            f"got {sampling_rate}"
        )

    pad_val = float(feature_extractor.padding_value)
    side = feature_extractor.padding_side

    if isinstance(waveforms, Tensor):
        w = waveforms
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if w.dim() != 2:
            raise ValueError("waveforms tensor must be (batch, samples) or (samples,)")
        rows = [w[i].to(device=device, dtype=dtype) for i in range(w.size(0))]
    else:
        rows = []
        for i, x in enumerate(waveforms):
            if not isinstance(x, Tensor):
                raise TypeError(
                    f"wav2vec2_preprocess_batch expected Tensor entries; got {type(x).__name__} at index {i}"
                )
            if x.dim() != 1:
                raise ValueError(f"waveforms[{i}] must be 1-D, got shape {tuple(x.shape)}")
            rows.append(x.to(device=device, dtype=dtype))

    if not rows:
        raise ValueError("Empty waveform batch")

    lengths = [int(r.shape[0]) for r in rows]
    max_len = max(lengths)
    b = len(rows)
    padded = torch.full((b, max_len), pad_val, device=device, dtype=dtype)
    attention_mask = torch.zeros(b, max_len, device=device, dtype=torch.long)

    for i, r in enumerate(rows):
        n = lengths[i]
        if side == "right":
            padded[i, :n] = r
            attention_mask[i, :n] = 1
        elif side == "left":
            padded[i, -n:] = r
            attention_mask[i, -n:] = 1
        else:
            raise ValueError(f"Unsupported padding_side: {side!r}")

    if feature_extractor.do_normalize:
        padded = _zero_mean_unit_var_norm_torch(padded, attention_mask, pad_val)

    return padded, attention_mask


def frame_mask_from_sample_mask(attention_mask: Tensor, wav2vec2_feature_extractor: nn.Module) -> Tensor:
    """Build frame-level mask from sample-level attention mask."""
    if attention_mask.dtype != torch.long:
        attention_mask = attention_mask.long()
    lengths = attention_mask.sum(dim=-1).clamp(min=1)
    frame_lengths = wav2vec2_feature_extractor._get_feat_extract_output_lengths(lengths)
    max_frames = int(frame_lengths.max().item())
    b = frame_lengths.shape[0]
    idx = torch.arange(max_frames, device=frame_lengths.device, dtype=frame_lengths.dtype).expand(
        b, max_frames
    )
    return (idx < frame_lengths.unsqueeze(-1)).to(dtype=torch.float32)


def _waveform_rows_for_extractor(
    waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
) -> List[Tensor]:
    """Return a list of 1-D float CPU tensors (true lengths preserved for the feature extractor)."""
    if isinstance(waveforms, Tensor):
        w = waveforms
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if w.dim() != 2:
            raise ValueError("waveforms tensor must be (batch, samples) or (samples,)")
        return [w[i].detach().float().cpu() for i in range(w.size(0))]
    rows: List[Tensor] = []
    for i, x in enumerate(waveforms):
        t = torch.as_tensor(x, dtype=torch.float32).detach().cpu()
        if t.dim() != 1:
            raise ValueError(
                "waveforms[{}] must be 1-D, got shape {}".format(i, tuple(t.shape))
            )
        rows.append(t)
    return rows


@dataclass
class XLSRSVOutput:
    """Outputs from `XLSRSV.forward`."""

    pooled_embedding: Tensor
    """L2-normalized pooled embeddings from a chosen encoder layer, shape `(batch, 1024)`."""

    frame_hidden_states: Tensor
    """Hidden states from the chosen encoder layer, shape `(batch, time, 1024)`."""

    frame_mask: Tensor
    """Float mask with `1.0` for valid frames and `0.0` for padding, shape `(batch, time)`."""

    attention_mask: Tensor
    """Sample-level mask from the feature extractor (`1` = valid audio sample), shape `(batch, audio_len)`."""

    all_hidden_states: Optional[Tuple[Tensor, ...]] = None
    """If requested, tuple of encoder layer outputs (each `(batch, time, 1024)`), index `0` = first layer."""


class XLSRSV(nn.Module):
    """Frozen `Wav2Vec2Model` (XLS-R) + feature extractor.

    Provides:
    - frame_hidden_states from an intermediate encoder layer (configurable layer index)
    - frame_mask derived from the feature extractor lengths
    - pooled_embedding: masked mean over time from the same layer, L2-normalized
    """

    default_model_id = "facebook/wav2vec2-xls-r-300m"

    def __init__(
        self,
        model_id: str = default_model_id,
        *,
        layer_idx: int = 12,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.layer_idx = layer_idx

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        model = Wav2Vec2Model.from_pretrained(model_id)
        model.requires_grad_(False)
        model.eval()
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        self.wav2vec2 = model

    def train(self, mode: bool = True) -> "XLSRSV":
        super().train(mode)
        # Keep XLS-R frozen regardless of train/eval toggles.
        self.wav2vec2.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.wav2vec2.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.wav2vec2.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_id: Optional[str] = None,
        *,
        layer_idx: int = 12,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "XLSRSV":
        return cls(model_id or cls.default_model_id, layer_idx=layer_idx, device=device, dtype=dtype)

    def _extract_inputs(
        self,
        waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
        sampling_rate: int,
    ) -> Tuple[Tensor, Tensor]:
        rows = _waveform_rows_for_extractor(waveforms)
        np_rows = [r.numpy() for r in rows]
        enc = self.feature_extractor(
            np_rows,
            padding=True,
            return_tensors="pt",
            sampling_rate=sampling_rate,
        )
        input_values = enc["input_values"].to(device=self.device, dtype=self.dtype)
        attention_mask = enc["attention_mask"].to(device=self.device, dtype=torch.long)
        return input_values, attention_mask

    def _prepare_model_inputs(
        self,
        waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
        sampling_rate: int,
    ) -> Tuple[Tensor, Tensor]:
        """Hugging Face NumPy path for non-tensor inputs; torch path (config-matched padding/mask/norm) for tensors."""
        if _waveforms_use_torch_preprocess(waveforms):
            return wav2vec2_preprocess_batch(
                self.feature_extractor,
                waveforms,  # type: ignore[arg-type]
                device=self.device,
                dtype=self.dtype,
                sampling_rate=sampling_rate,
            )
        return self._extract_inputs(waveforms, sampling_rate)

    def forward(
        self,
        waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
        *,
        sampling_rate: int = 16_000,
        layer_idx: Optional[int] = None,
        normalize_embeddings: bool = True,
        grad_through_input: bool = False,
        output_hidden_states: bool = False,
    ) -> XLSRSVOutput:
        """Run frozen XLS-R on mono waveforms (16 kHz).

        Args:
            waveforms: Batch of shape `(batch, samples)`, or list of 1-D waveforms (variable length).
            sampling_rate: Input sample rate; must be 16_000 for XLS-R.
            layer_idx: Optional override for the encoder layer index to tap (default: `self.layer_idx`).
            normalize_embeddings: If True, L2-normalize `pooled_embedding`.
            grad_through_input: If True, run the forward *without* `torch.no_grad()` so gradients
                can flow into `waveforms` (weights stay frozen).
            output_hidden_states: If True, `all_hidden_states` is set on the output (tuple of every
                encoder layer).
        """
        if sampling_rate != 16_000:
            raise ValueError(
                f"XLS-R expects 16 kHz audio; got sampling_rate={sampling_rate}"
            )

        input_values, attention_mask = self._prepare_model_inputs(waveforms, sampling_rate)
        frame_mask = frame_mask_from_sample_mask(attention_mask, self.wav2vec2.feature_extractor)

        use_layer_idx = self.layer_idx if layer_idx is None else layer_idx

        if grad_through_input:
            out = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        else:
            with torch.no_grad():
                out = self.wav2vec2(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

        hidden_states: Tuple[Tensor, ...] = out.hidden_states  # type: ignore[assignment]
        num_layers = len(hidden_states)
        if not (0 <= use_layer_idx < num_layers):
            raise ValueError(
                f"layer_idx={use_layer_idx} is out of range for XLS-R hidden_states (len={num_layers})"
            )

        frames = hidden_states[use_layer_idx]

        m = frame_mask.to(dtype=frames.dtype, device=frames.device).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1e-6)
        pooled = (frames * m).sum(dim=1) / denom

        emb = F.normalize(pooled, dim=-1) if normalize_embeddings else pooled

        all_hs: Optional[Tuple[Tensor, ...]] = hidden_states if output_hidden_states else None

        return XLSRSVOutput(
            pooled_embedding=emb,
            frame_hidden_states=frames,
            frame_mask=frame_mask,
            attention_mask=attention_mask,
            all_hidden_states=all_hs,
        )

