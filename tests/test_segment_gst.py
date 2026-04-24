"""Tests for SegmentGST split-head behavior."""

from __future__ import annotations

import pytest
import torch

B, T, FRAME_DIM = 2, 30, 768


def test_ref_s_matches_universal_style_at_init(segment_gst_mod):
    u = torch.linspace(-1.0, 1.0, 256)
    gst = segment_gst_mod.SegmentGST(universal_style_vector=u)
    out, _ = gst(torch.randn(B, T, FRAME_DIM), torch.ones(B, T))
    torch.testing.assert_close(out.ref_s, u.unsqueeze(0).expand(B, -1))
    torch.testing.assert_close(out.style_dec, u[:128].unsqueeze(0).expand(B, -1))
    torch.testing.assert_close(out.style_pred, u[128:].unsqueeze(0).expand(B, -1))


def test_split_heads_zero_initialized(segment_gst_mod):
    gst = segment_gst_mod.SegmentGST()
    assert torch.count_nonzero(gst.to_style_dec.weight) == 0
    assert torch.count_nonzero(gst.to_style_pred.weight) == 0


def test_decoder_head_only_receives_decoder_loss_grad(segment_gst_mod):
    gst = segment_gst_mod.SegmentGST()
    out, _ = gst(torch.randn(B, T, FRAME_DIM), torch.ones(B, T))
    out.style_dec.sum().backward()
    assert gst.to_style_dec.weight.grad is not None
    assert gst.to_style_dec.weight.grad.abs().sum() > 0
    assert gst.to_style_pred.weight.grad is None


def test_prosody_head_only_receives_prosody_loss_grad(segment_gst_mod):
    gst = segment_gst_mod.SegmentGST()
    out, _ = gst(torch.randn(B, T, FRAME_DIM), torch.ones(B, T))
    out.style_pred.sum().backward()
    assert gst.to_style_pred.weight.grad is not None
    assert gst.to_style_pred.weight.grad.abs().sum() > 0
    assert gst.to_style_dec.weight.grad is None


def test_rejects_invalid_universal_vector_length(segment_gst_mod):
    with pytest.raises(ValueError, match="length"):
        segment_gst_mod.SegmentGST(universal_style_vector=torch.zeros(255))
