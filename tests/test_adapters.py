"""Tests for ResidualAdapter and AdapterRegistry (voice_clone/adapters.py)."""

from __future__ import annotations

import torch
import pytest


# Architecture constants from kokoro/kokoro/istftnet.py (documented in architecture.md).
# Hardcoded here to avoid pulling the full kokoro dep chain (loguru, etc.).
_DECODER_L_ADAPTER_HIDDEN_DIMS = (1024, 1024, 1024, 1024, 512)


def _generator_l_adapter_hidden_dims(upsample_initial_channel: int, num_upsamples: int) -> tuple[int, ...]:
    return tuple(upsample_initial_channel // (2 ** (i + 1)) for i in range(num_upsamples))


# ---------------------------------------------------------------------------
# ResidualAdapter
# ---------------------------------------------------------------------------

class TestResidualAdapter:
    def test_zero_init_identity(self, adapters_mod):
        """Freshly constructed adapter returns h unchanged (W_up is zero-init)."""
        adapter = adapters_mod.ResidualAdapter(hidden_dim=512, style_dim=256, bottleneck_dim=64)
        h = torch.randn(2, 512, 10)
        z = torch.randn(2, 256)
        out = adapter(h, z)
        torch.testing.assert_close(out, h)

    def test_output_shape(self, adapters_mod):
        adapter = adapters_mod.ResidualAdapter(hidden_dim=512, style_dim=256, bottleneck_dim=64)
        B, C, T = 3, 512, 20
        h = torch.randn(B, C, T)
        z = torch.randn(B, 256)
        out = adapter(h, z)
        assert out.shape == (B, C, T)

    def test_output_shape_different_dims(self, adapters_mod):
        """Works with non-default channel widths."""
        adapter = adapters_mod.ResidualAdapter(hidden_dim=1024, style_dim=256, bottleneck_dim=32)
        h = torch.randn(1, 1024, 5)
        z = torch.randn(1, 256)
        assert adapter(h, z).shape == (1, 1024, 5)

    def test_gradient_flow_to_down_weight(self, adapters_mod):
        """Backward through adapter produces non-zero grad on down.weight."""
        adapter = adapters_mod.ResidualAdapter(hidden_dim=512, style_dim=256, bottleneck_dim=64)
        # Perturb up so the adapter is no longer identity
        with torch.no_grad():
            adapter.up.weight.fill_(0.01)
        h = torch.randn(2, 512, 10, requires_grad=True)
        z = torch.randn(2, 256, requires_grad=True)
        out = adapter(h, z)
        out.sum().backward()
        assert adapter.down.weight.grad is not None
        assert adapter.down.weight.grad.abs().sum() > 0

    def test_residual_structure(self, adapters_mod):
        """After perturbing W_up, output differs from h but stays close (residual)."""
        adapter = adapters_mod.ResidualAdapter(hidden_dim=512, style_dim=256, bottleneck_dim=64)
        with torch.no_grad():
            adapter.up.weight.fill_(1e-3)
        h = torch.randn(1, 512, 5)
        z = torch.randn(1, 256)
        out = adapter(h, z)
        assert not torch.equal(out, h)
        # Delta should be small relative to h
        delta = (out - h).abs().mean()
        assert delta < h.abs().mean()

    def test_rejects_wrong_style_shape(self, adapters_mod):
        adapter = adapters_mod.ResidualAdapter(hidden_dim=512, style_dim=256, bottleneck_dim=64)
        h = torch.randn(2, 512, 10)
        with pytest.raises(ValueError, match="z_style"):
            adapter(h, torch.randn(2, 128))

    def test_rejects_wrong_hidden_shape(self, adapters_mod):
        adapter = adapters_mod.ResidualAdapter(hidden_dim=512, style_dim=256, bottleneck_dim=64)
        with pytest.raises(ValueError, match="hidden_dim"):
            adapter(torch.randn(2, 256, 10), torch.randn(2, 256))


# ---------------------------------------------------------------------------
# build_duration_encoder_adapters
# ---------------------------------------------------------------------------

class TestBuildDurationEncoderAdapters:
    def test_count(self, adapters_mod):
        adapters = adapters_mod.build_duration_encoder_adapters(
            d_model=512, z_style_dim=256, nlayers=3, bottleneck_dim=64,
        )
        assert len(adapters) == 3

    def test_all_hidden_dim_512(self, adapters_mod):
        adapters = adapters_mod.build_duration_encoder_adapters(
            d_model=512, z_style_dim=256, nlayers=3, bottleneck_dim=64,
        )
        for a in adapters:
            assert a.up.out_features == 512


# ---------------------------------------------------------------------------
# AdapterRegistry.from_dims (uses Kokoro istftnet constants)
# ---------------------------------------------------------------------------

class TestAdapterRegistry:
    @pytest.fixture
    def registry(self, adapters_mod):
        """Build registry manually using architecture constants (avoids kokoro dep chain)."""
        dur = adapters_mod.build_duration_encoder_adapters(
            d_model=512, z_style_dim=256, nlayers=3, bottleneck_dim=64,
        )
        dec = torch.nn.ModuleList(
            adapters_mod.ResidualAdapter(dim, 256, 64) for dim in _DECODER_L_ADAPTER_HIDDEN_DIMS
        )
        gen_dims = _generator_l_adapter_hidden_dims(512, 2)
        gen = torch.nn.ModuleList(
            adapters_mod.ResidualAdapter(dim, 256, 64) for dim in gen_dims
        )
        return adapters_mod.AdapterRegistry(duration_encoder=dur, decoder=dec, generator=gen)

    def test_duration_count(self, registry):
        assert len(registry.duration_encoder) == 3

    def test_decoder_count(self, registry):
        assert len(registry.decoder) == 5

    def test_generator_count(self, registry):
        assert len(registry.generator) == 2

    def test_total_adapter_count(self, registry):
        total = len(registry.duration_encoder) + len(registry.decoder) + len(registry.generator)
        assert total == 10

    def test_duration_channel_dims(self, registry):
        for a in registry.duration_encoder:
            assert a.up.out_features == 512

    def test_decoder_channel_dims(self, registry):
        expected = (1024, 1024, 1024, 1024, 512)
        actual = tuple(a.up.out_features for a in registry.decoder)
        assert actual == expected

    def test_generator_channel_dims(self, registry):
        expected = (256, 128)
        actual = tuple(a.up.out_features for a in registry.generator)
        assert actual == expected

    def test_all_adapters_start_as_identity(self, registry):
        """Every adapter in the registry should produce identity at init."""
        for group in (registry.duration_encoder, registry.decoder, registry.generator):
            for adapter in group:
                C = adapter.up.out_features
                h = torch.randn(1, C, 8)
                z = torch.randn(1, 256)
                torch.testing.assert_close(adapter(h, z), h)
