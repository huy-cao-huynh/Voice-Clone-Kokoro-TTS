"""SegmentGST: MHA over a learnable bank with frame queries -> Kokoro `ref_s`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SegmentGSTOutput:
    ref_s: torch.Tensor
    style_dec: torch.Tensor
    style_pred: torch.Tensor
    pooled_style: torch.Tensor


class SegmentGST(nn.Module):
    def __init__(
        self,
        *,
        num_bases: int = 1024,
        embed_dim: int = 1024,
        frame_dim: int = 768,
        num_heads: int = 4,
        ref_dim: int = 256,
        style_dec_dim: int = 128,
        dropout: float = 0.1,
        universal_style_vector: Optional[torch.Tensor] = None,
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
        self.style_pred_dim = ref_dim - style_dec_dim
        self.conv_kernel_size = 8
        self.conv_stride = 4
        self.conv_padding = 3

        self.bank = nn.Parameter(torch.empty(num_bases, embed_dim))
        nn.init.normal_(self.bank, std=0.02)
        self.q_proj = nn.Linear(frame_dim, embed_dim)
        self.pre_conv_norm = nn.LayerNorm(embed_dim)
        self.temporal_conv1 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
        )
        self.temporal_conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
        )
        self.post_conv_norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.to_style_dec = nn.Linear(embed_dim, self.style_dec_dim)
        self.to_style_pred = nn.Linear(embed_dim, self.style_pred_dim)
        nn.init.zeros_(self.to_style_dec.weight)
        nn.init.zeros_(self.to_style_dec.bias)
        nn.init.zeros_(self.to_style_pred.weight)
        nn.init.zeros_(self.to_style_pred.bias)

        if universal_style_vector is None:
            u = torch.zeros(ref_dim, dtype=self.to_style_dec.weight.dtype)
        else:
            if universal_style_vector.dim() != 1:
                raise ValueError("universal_style_vector must be 1-D")
            if int(universal_style_vector.numel()) != ref_dim:
                raise ValueError(
                    f"universal_style_vector length {int(universal_style_vector.numel())} != ref_dim {ref_dim}"
                )
            u = universal_style_vector.detach().to(dtype=self.to_style_dec.weight.dtype)
        self.register_buffer("universal_style_vector", u, persistent=True)

    def _reduce_mask_once(self, mask: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool1d(
            mask.to(dtype=torch.float32).unsqueeze(1),
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
        )
        return pooled.squeeze(1) > 0

    def _reduce_sequence(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pre_conv_norm(x)
        x = x * mask.to(dtype=x.dtype).unsqueeze(-1)
        x = x.transpose(1, 2)

        x = F.gelu(self.temporal_conv1(x))
        mask = self._reduce_mask_once(mask)
        x = x * mask.to(dtype=x.dtype).unsqueeze(1)

        x = F.gelu(self.temporal_conv2(x))
        mask = self._reduce_mask_once(mask)
        x = x.transpose(1, 2)
        x = self.post_conv_norm(x)
        x = x * mask.to(dtype=x.dtype).unsqueeze(-1)
        return x, mask

    def forward(
        self,
        frame_hidden_states: torch.Tensor,
        frame_mask: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[SegmentGSTOutput, Optional[torch.Tensor]]:
        if frame_hidden_states.dim() != 3:
            raise ValueError(
                f"frame_hidden_states must be (batch, time, dim); got {tuple(frame_hidden_states.shape)}"
            )
        if frame_mask.dim() != 2:
            raise ValueError(f"frame_mask must be (batch, time); got {tuple(frame_mask.shape)}")
        b, t, fd = frame_hidden_states.shape
        if fd != self.frame_dim:
            raise ValueError(f"frame_hidden_states last dim {fd} != SegmentGST.frame_dim {self.frame_dim}")
        if tuple(frame_mask.shape) != (b, t):
            raise ValueError(f"frame_mask shape {tuple(frame_mask.shape)} != {(b, t)} for frame_hidden_states")

        valid_mask = frame_mask.to(device=frame_hidden_states.device)
        if valid_mask.dtype is not torch.bool:
            valid_mask = valid_mask > 0

        q = self.q_proj(frame_hidden_states)
        q, reduced_mask = self._reduce_sequence(q, valid_mask)
        kv = self.bank.unsqueeze(0).expand(b, -1, -1).contiguous()
        attn_out, attn_w = self.mha(q, kv, kv, need_weights=need_weights, average_attn_weights=True)
        attn_out = self.dropout(self.norm(attn_out))

        m = reduced_mask.to(device=attn_out.device, dtype=attn_out.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1e-6)
        pooled = (attn_out * m).sum(dim=1) / denom

        base = self.universal_style_vector.to(device=pooled.device, dtype=pooled.dtype)
        u_dec = base[: self.style_dec_dim].unsqueeze(0).expand(b, -1)
        u_pred = base[self.style_dec_dim :].unsqueeze(0).expand(b, -1)
        style_dec = u_dec + self.to_style_dec(pooled)
        style_pred = u_pred + self.to_style_pred(pooled)
        ref_s = torch.cat([style_dec, style_pred], dim=-1)
        return SegmentGSTOutput(ref_s=ref_s, style_dec=style_dec, style_pred=style_pred, pooled_style=pooled), attn_w
