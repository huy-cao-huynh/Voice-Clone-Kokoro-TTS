"""Frozen mHuBERT encoder wrapper with hidden-state layer extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers import HubertModel


@dataclass
class MHuBERTOutput:
    """Output of :meth:`MHuBERTEncoder.forward`."""

    hidden_states: torch.Tensor
    """Hidden states from the selected transformer layer, shape ``(batch, frames, 768)``."""

    frame_mask: torch.Tensor
    """Boolean mask with ``True`` for valid frames, shape ``(batch, frames)``."""


class MHuBERTEncoder(nn.Module):
    """Frozen mHuBERT wrapper that extracts hidden states from a chosen transformer layer.

    Parameters
    ----------
    repo_id:
        Hugging Face Hub model identifier.
    extract_layer:
        Which transformer layer's hidden state to return (0-indexed into the
        ``output_hidden_states`` tuple where index 0 is the CNN feature projection
        and indices 1..N are the transformer layers).
    """

    def __init__(
        self,
        repo_id: str = "utter-project/mHuBERT-147-base-3rd-iter",
        extract_layer: int = 9,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.extract_layer = extract_layer

        self.model: HubertModel = HubertModel.from_pretrained(repo_id)
        self.hidden_size: int = self.model.config.hidden_size  # 768 for mHuBERT-base

        num_layers = self.model.config.num_hidden_layers
        if not 0 <= extract_layer <= num_layers:
            raise ValueError(
                f"extract_layer={extract_layer} out of range for a model with "
                f"{num_layers} transformer layers (valid: 0..{num_layers})"
            )

        self.requires_grad_(False)
        self.eval()

    # ------------------------------------------------------------------
    # Keep encoder frozen regardless of external train/eval toggles.
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "MHuBERTEncoder":
        super().train(mode)
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Frame-length computation
    # ------------------------------------------------------------------

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Replicate HuBERT's CNN feature-extractor length reduction.

        The model's conv feature extractor applies 7 conv layers with
        ``kernel_sizes = [10, 3, 3, 3, 3, 2, 2]`` and
        ``strides = [5, 2, 2, 2, 2, 2, 2]``.  Each layer reduces the
        sequence length as ``floor((L - kernel) / stride) + 1``.

        We delegate to the model's own helper when available (it handles
        edge cases around ``conv_bias``).
        """
        return self.model._get_feat_extract_output_lengths(input_lengths)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        waveforms_16k: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> MHuBERTOutput:
        """Run frozen mHuBERT and return intermediate hidden states + frame mask.

        Parameters
        ----------
        waveforms_16k:
            Raw 16 kHz mono waveforms, shape ``(batch, samples)``.
        attention_mask:
            Optional ``(batch, samples)`` mask with ``1`` for valid samples and
            ``0`` for padding.  When *None*, all samples are treated as valid.

        Returns
        -------
        MHuBERTOutput
            ``.hidden_states`` shaped ``(batch, frames, hidden_size)`` and
            ``.frame_mask`` shaped ``(batch, frames)`` (bool).
        """
        if waveforms_16k.dim() == 1:
            waveforms_16k = waveforms_16k.unsqueeze(0)
        if waveforms_16k.dim() != 2:
            raise ValueError(
                f"waveforms_16k must be (batch, samples); got {tuple(waveforms_16k.shape)}"
            )

        outputs = self.model(
            waveforms_16k,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = outputs.hidden_states[self.extract_layer]  # (B, T_frames, hidden_size)

        # Build frame-level boolean mask.
        B, T = hidden.shape[:2]
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=-1).long()
            frame_lengths = self._get_feat_extract_output_lengths(input_lengths)
            frame_ids = torch.arange(T, device=hidden.device).unsqueeze(0)  # (1, T)
            frame_mask = frame_ids < frame_lengths.unsqueeze(1)  # (B, T)
        else:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=hidden.device)

        return MHuBERTOutput(hidden_states=hidden, frame_mask=frame_mask)
