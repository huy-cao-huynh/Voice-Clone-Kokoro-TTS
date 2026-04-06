"""Frozen Hugging Face `microsoft/wavlm-base-plus-sv` for SV embeddings and encoder frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


Tensor = torch.Tensor


def _waveforms_use_torch_preprocess(
    waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
) -> bool:
    """True for tensor batches: avoids NumPy/HF round-trip and keeps autograd through audio when requested.

    Nested Python lists / non-tensor sequences use the Hugging Face NumPy path instead.
    """
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
    eps: float = 1e-5,
) -> Tensor:
    """Match Hugging Face `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm` (batched, differentiable)."""
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
    """Pad and optionally normalize waveforms like ``Wav2Vec2FeatureExtractor(..., return_tensors='pt')``.

    Reads ``do_normalize``, ``padding_value``, ``padding_side``, and ``sampling_rate`` from the extractor.
    Output ``input_values`` uses ``dtype``; ``attention_mask`` is ``torch.long`` (1 = valid, 0 = pad).
    """
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
        L = lengths[i]
        if side == "right":
            padded[i, :L] = r
            attention_mask[i, :L] = 1
        elif side == "left":
            padded[i, -L:] = r
            attention_mask[i, -L:] = 1
        else:
            raise ValueError(f"Unsupported padding_side: {side!r}")

    if feature_extractor.do_normalize:
        padded = _zero_mean_unit_var_norm_torch(padded, attention_mask, pad_val)

    return padded, attention_mask


@dataclass
class WavLMSVOutput:
    """Outputs from `WavLMSV.forward`."""

    pooled_embedding: Tensor
    """L2-normalized x-vector embeddings, shape `(batch, 512)`."""

    frame_hidden_states: Tensor
    """Last encoder layer hidden states (before the x-vector head), shape `(batch, time, 768)`."""

    frame_mask: Tensor
    """Float mask with `1.0` for valid frames and `0.0` for padding, shape `(batch, time)`.

    `time` matches `frame_hidden_states.size(1)` (max frames in the batch after CNN subsampling).
    """

    attention_mask: Tensor
    """Sample-level mask from the feature extractor (`1` = valid audio sample), shape `(batch, audio_len)`."""

    all_hidden_states: Optional[Tuple[Tensor, ...]] = None
    """If requested, tuple of encoder layer outputs (each `(batch, time, 768)`), index `0` = first layer."""


def frame_mask_from_sample_mask(attention_mask: Tensor, wavlm_core: nn.Module) -> Tensor:
    """Build frame-level mask from feature-extractor sample `attention_mask`.

    Args:
        attention_mask: Long tensor `(batch, audio_len)`, `1` for valid samples.
        wavlm_core: `WavLMForXVector.wavlm` (provides `_get_feat_extract_output_lengths`).

    Returns:
        Float tensor `(batch, max_frames)` with `1.0` for valid frames.
    """
    if attention_mask.dtype != torch.long:
        attention_mask = attention_mask.long()
    lengths = attention_mask.sum(dim=-1).clamp(min=1)
    frame_lengths = wavlm_core._get_feat_extract_output_lengths(lengths)
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


class WavLMSV(nn.Module):
    """Frozen `WavLMForXVector` + feature extractor: pooled SV embedding and encoder frame states."""

    default_model_id = "microsoft/wavlm-base-plus-sv"

    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base-plus-sv",
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        model = WavLMForXVector.from_pretrained(model_id)
        model.requires_grad_(False)
        model.eval()
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        self.wavlm_xv = model

    def train(self, mode: bool = True) -> "WavLMSV":
        super().train(mode)
        self.wavlm_xv.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.wavlm_xv.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.wavlm_xv.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_id: Optional[str] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "WavLMSV":
        return cls(model_id or cls.default_model_id, device=device, dtype=dtype)

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
        normalize_embeddings: bool = True,
        output_hidden_states: bool = False,
        grad_through_input: bool = False,
    ) -> WavLMSVOutput:
        """Run frozen WavLM-SV on mono waveforms (default **16 kHz** per model card).

        Args:
            waveforms: Batch of shape `(batch, samples)`, or list of 1-D waveforms (variable length).
            sampling_rate: Input sample rate; must be 16_000 for this checkpoint.
            normalize_embeddings: If True, L2-normalize `pooled_embedding` (speaker cosine loss).
            output_hidden_states: If True, `all_hidden_states` is set on the output (tuple of every
                encoder layer) for multi-layer SLM / analysis. Frames always use the last layer.
            grad_through_input: If True, run the forward **without** `torch.no_grad()` so gradients
                can flow into `waveforms` (weights stay frozen). Use for generated audio in speaker
                / SLM generator losses.

        Returns:
            `WavLMSVOutput` with pooled embedding, last-layer frame features, and masks.
        """
        if sampling_rate != 16_000:
            raise ValueError(
                f"microsoft/wavlm-base-plus-sv expects 16 kHz audio; got sampling_rate={sampling_rate}"
            )

        input_values, attention_mask = self._prepare_model_inputs(waveforms, sampling_rate)
        frame_mask = frame_mask_from_sample_mask(attention_mask, self.wavlm_xv.wavlm)

        if grad_through_input:
            out = self.wavlm_xv(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        else:
            with torch.no_grad():
                out = self.wavlm_xv(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

        frames = out.hidden_states[-1]
        emb = out.embeddings
        if normalize_embeddings:
            emb = F.normalize(emb, dim=-1)

        all_hs: Optional[Tuple[Tensor, ...]] = out.hidden_states if output_hidden_states else None

        return WavLMSVOutput(
            pooled_embedding=emb,
            frame_hidden_states=frames,
            frame_mask=frame_mask,
            attention_mask=attention_mask,
            all_hidden_states=all_hs,
        )

    def pooled_embedding(
        self,
        waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
        *,
        sampling_rate: int = 16_000,
        normalize: bool = True,
        grad_through_input: bool = False,
    ) -> Tensor:
        """X-vector embeddings only, shape `(batch, 512)`."""
        if sampling_rate != 16_000:
            raise ValueError(
                f"microsoft/wavlm-base-plus-sv expects 16 kHz audio; got sampling_rate={sampling_rate}"
            )
        input_values, attention_mask = self._prepare_model_inputs(waveforms, sampling_rate)
        if grad_through_input:
            out = self.wavlm_xv(input_values, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                out = self.wavlm_xv(input_values, attention_mask=attention_mask)
        emb = out.embeddings
        return F.normalize(emb, dim=-1) if normalize else emb

    def frame_hidden_states(
        self,
        waveforms: Union[Tensor, Sequence[Tensor], Sequence[Sequence[float]]],
        *,
        sampling_rate: int = 16_000,
        output_all_hidden_states: bool = False,
        grad_through_input: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Tensor, ...]]]:
        """Encoder frame states and frame mask (and optionally all layer states).

        Returns:
            `frame_hidden` `(batch, time, hidden)`, `frame_mask` `(batch, time)`, and optionally
            `all_hidden_states` (tuple of `(batch, time, hidden)` per layer) if
            `output_all_hidden_states` is True.
        """
        if sampling_rate != 16_000:
            raise ValueError(
                f"microsoft/wavlm-base-plus-sv expects 16 kHz audio; got sampling_rate={sampling_rate}"
            )
        input_values, attention_mask = self._prepare_model_inputs(waveforms, sampling_rate)
        frame_mask = frame_mask_from_sample_mask(attention_mask, self.wavlm_xv.wavlm)

        if output_all_hidden_states:
            if grad_through_input:
                out = self.wavlm_xv(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            else:
                with torch.no_grad():
                    out = self.wavlm_xv(
                        input_values,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
            return out.hidden_states[-1], frame_mask, out.hidden_states

        # Optimized: base encoder only, skip TDNN x-vector head and
        # avoid allocating all 13 layer hidden-state tensors.
        base = self.wavlm_xv.wavlm
        if grad_through_input:
            out = base(input_values, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                out = base(input_values, attention_mask=attention_mask)
        return out.last_hidden_state, frame_mask, None
