"""SegmentGST: MHA over a learnable bank with XLS-R frame queries → Kokoro `ref_s` (256 = 128|128)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SegmentGSTOutput:
    """Output of `SegmentGST.forward`."""

    ref_s: torch.Tensor
    """Kokoro style vector, shape `(batch, 256)`; `[:, :128]` → decoder, `[:, 128:]` → prosody."""

    style_dec: torch.Tensor
    """First 128 dims of `ref_s`, shape `(batch, 128)`."""

    style_pred: torch.Tensor
    """Last 128 dims of `ref_s`, shape `(batch, 128)`."""

    pooled_style: torch.Tensor
    """Pooled MHA output before `to_ref_s`, shape `(batch, embed_dim)`."""


class SegmentGST(nn.Module):
    """Multi-head attention from frozen XLS-R frames (queries) into a learnable bank `B` (keys/values).

    Matches the plan: bank `B ∈ R^{N×d}`, queries from the frame sequence (masked), pooled regularized
    style, then linear to 256 with Kokoro's split `ref_s = cat(style_dec_128, style_pred_128)`.
    """

    def __init__(
        self,
        *,
        num_bases: int = 512,
        embed_dim: int = 256,
        frame_dim: int = 1024,
        num_heads: int = 8,
        ref_dim: int = 256,
        style_dec_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if ref_dim != style_dec_dim * 2:
            raise ValueError(
                f"ref_dim must equal 2 * style_dec_dim for Kokoro; got ref_dim={ref_dim}, "
                f"style_dec_dim={style_dec_dim}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.num_bases = num_bases
        self.embed_dim = embed_dim
        self.frame_dim = frame_dim
        self.ref_dim = ref_dim
        self.style_dec_dim = style_dec_dim

        self.bank = nn.Parameter(torch.empty(num_bases, embed_dim))
        nn.init.normal_(self.bank, std=0.02)

        self.q_proj = nn.Linear(frame_dim, embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.to_ref_s = nn.Linear(embed_dim, ref_dim)
        nn.init.zeros_(self.to_ref_s.weight)
        nn.init.zeros_(self.to_ref_s.bias)

    def forward(
        self,
        frame_hidden_states: torch.Tensor,
        frame_mask: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> Tuple[SegmentGSTOutput, Optional[torch.Tensor]]:
        """Compute `ref_s` from XLS-R encoder frames.

        Args:
            frame_hidden_states: `(batch, time, frame_dim)` e.g. XLS-R layer hidden states.
            frame_mask: `(batch, time)` with `1.0` for valid frames and `0.0` for padding.
            need_weights: If True, also return average attention weights over heads `(batch, time, N)`.

        Returns:
            `(SegmentGSTOutput, attn_weights_or_none)`.
        """
        if frame_hidden_states.dim() != 3:
            raise ValueError(
                f"frame_hidden_states must be (batch, time, dim); got {tuple(frame_hidden_states.shape)}"
            )
        if frame_mask.dim() != 2:
            raise ValueError(f"frame_mask must be (batch, time); got {tuple(frame_mask.shape)}")
        b, t, fd = frame_hidden_states.shape
        if fd != self.frame_dim:
            raise ValueError(
                f"frame_hidden_states last dim {fd} != SegmentGST.frame_dim {self.frame_dim}"
            )
        if frame_mask.shape != (b, t):
            raise ValueError(
                f"frame_mask shape {tuple(frame_mask.shape)} != {(b, t)} for frame_hidden_states"
            )

        q = self.q_proj(frame_hidden_states)
        kv = self.bank.unsqueeze(0).expand(b, -1, -1).contiguous()

        key_padding_mask = None
        attn_mask = None

        attn_out, attn_w = self.mha(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        # attn_out: (B, T, embed_dim)
        attn_out = self.norm(attn_out)
        attn_out = self.dropout(attn_out)

        m = frame_mask.to(dtype=attn_out.dtype, device=attn_out.device).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1e-6)
        pooled = (attn_out * m).sum(dim=1) / denom

        ref_s = self.to_ref_s(pooled)
        style_dec = ref_s[:, : self.style_dec_dim]
        style_pred = ref_s[:, self.style_dec_dim :]

        out = SegmentGSTOutput(
            ref_s=ref_s,
            style_dec=style_dec,
            style_pred=style_pred,
            pooled_style=pooled,
        )
        return out, attn_w
