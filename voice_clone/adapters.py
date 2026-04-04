"""Trainable L-adapters: h' = h + W_up(ReLU(W_down([h || z_style])))."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAdapter(nn.Module):
    """Residual adapter on conv features. h is [B, C, T], z_style is [B, style_dim]."""

    def __init__(self, hidden_dim: int, style_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim + style_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(self, h: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
        b, _, t = h.shape
        z = z_style.unsqueeze(-1).expand(b, z_style.size(1), t)
        x = torch.cat([h, z], dim=1).transpose(1, 2)
        delta = self.up(F.relu(self.down(x))).transpose(1, 2)
        return h + delta


def build_duration_encoder_adapters(
    d_model: int,
    z_style_dim: int,
    nlayers: int,
    bottleneck_dim: int,
) -> nn.ModuleList:
    return nn.ModuleList(
        ResidualAdapter(d_model, z_style_dim, bottleneck_dim) for _ in range(nlayers)
    )


def build_decoder_adapters(
    z_style_dim: int,
    bottleneck_dim: int,
    upsample_initial_channel: int,
    num_upsamples: int,
) -> tuple[nn.ModuleList, nn.ModuleList]:
    from kokoro.istftnet import DECODER_L_ADAPTER_HIDDEN_DIMS, generator_l_adapter_hidden_dims

    dec = nn.ModuleList(
        ResidualAdapter(dim, z_style_dim, bottleneck_dim)
        for dim in DECODER_L_ADAPTER_HIDDEN_DIMS
    )
    gen_dims = generator_l_adapter_hidden_dims(upsample_initial_channel, num_upsamples)
    gen = nn.ModuleList(
        ResidualAdapter(dim, z_style_dim, bottleneck_dim) for dim in gen_dims
    )
    return dec, gen


class AdapterRegistry:
    """Holds optional ModuleLists for KModel(duration_encoder_adapters=..., ...)."""

    def __init__(
        self,
        duration_encoder: Optional[nn.ModuleList] = None,
        decoder: Optional[nn.ModuleList] = None,
        generator: Optional[nn.ModuleList] = None,
    ):
        self.duration_encoder = duration_encoder
        self.decoder = decoder
        self.generator = generator

    @staticmethod
    def from_dims(
        *,
        d_model: int,
        z_style_dim: int,
        duration_nlayers: int,
        adapter_bottleneck: int,
        upsample_initial_channel: int,
        num_upsamples: int,
    ) -> AdapterRegistry:
        dur = build_duration_encoder_adapters(
            d_model, z_style_dim, duration_nlayers, adapter_bottleneck
        )
        dec, gen = build_decoder_adapters(
            z_style_dim,
            adapter_bottleneck,
            upsample_initial_channel,
            num_upsamples,
        )
        return AdapterRegistry(duration_encoder=dur, decoder=dec, generator=gen)
