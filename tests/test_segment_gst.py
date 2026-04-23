"""Tests for SegmentGST (voice_clone/segment_gst.py)."""

from __future__ import annotations

import pytest
import torch


B, T, FRAME_DIM = 2, 30, 768


class TestSegmentGSTOutputShapes:
    def test_ref_s_shape(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        out, _ = gst(frames, mask)
        assert out.ref_s.shape == (B, 256)

    def test_style_dec_shape(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        out, _ = gst(frames, mask)
        assert out.style_dec.shape == (B, 128)

    def test_style_pred_shape(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        out, _ = gst(frames, mask)
        assert out.style_pred.shape == (B, 128)

    def test_pooled_style_shape(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        out, _ = gst(frames, mask)
        assert out.pooled_style.shape == (B, 256)


class TestSegmentGSTZeroInit:
    def test_to_ref_s_weight_is_zero(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        assert (gst.to_ref_s.weight == 0).all()
        assert (gst.to_ref_s.bias == 0).all()

    def test_ref_s_equals_universal_style_at_init(self, segment_gst_mod):
        """With zero-init readout, ref_s equals the universal style vector."""
        u = torch.linspace(-1.0, 1.0, 256)
        gst = segment_gst_mod.SegmentGST(universal_style_vector=u)
        gst.eval()
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        with torch.no_grad():
            out, _ = gst(frames, mask)
        expected = u.unsqueeze(0).expand(B, -1)
        torch.testing.assert_close(out.ref_s, expected)


class TestSegmentGSTKokoroSplit:
    def test_style_dec_is_first_half(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        # Perturb readout so ref_s is non-zero
        with torch.no_grad():
            gst.to_ref_s.weight.fill_(0.01)
            gst.to_ref_s.bias.fill_(0.1)
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        out, _ = gst(frames, mask)
        torch.testing.assert_close(out.style_dec, out.ref_s[:, :128])

    def test_style_pred_is_second_half(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        with torch.no_grad():
            gst.to_ref_s.weight.fill_(0.01)
            gst.to_ref_s.bias.fill_(0.1)
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        out, _ = gst(frames, mask)
        torch.testing.assert_close(out.style_pred, out.ref_s[:, 128:])


class TestSegmentGSTMaskHandling:
    def test_masked_frames_excluded_from_pooling(self, segment_gst_mod):
        """Different masks that zero-out different frames should give different outputs."""
        gst = segment_gst_mod.SegmentGST()
        gst.eval()
        with torch.no_grad():
            gst.to_ref_s.weight.fill_(0.01)
        frames = torch.randn(1, 20, FRAME_DIM)
        mask_full = torch.ones(1, 20)
        mask_half = torch.ones(1, 20)
        mask_half[0, 10:] = 0.0
        with torch.no_grad():
            out_full, _ = gst(frames, mask_full)
            out_half, _ = gst(frames, mask_half)
        assert not torch.allclose(out_full.pooled_style, out_half.pooled_style, atol=1e-6)

    def test_treats_positive_mask_values_as_binary_valid_frames(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        gst.eval()
        with torch.no_grad():
            gst.to_ref_s.weight.fill_(0.01)
        frames = torch.randn(1, 20, FRAME_DIM)
        weighted_mask = torch.tensor([[1.0] * 10 + [0.25] * 10])
        binary_mask = torch.ones(1, 20)
        with torch.no_grad():
            out_weighted, _ = gst(frames, weighted_mask)
            out_binary, _ = gst(frames, binary_mask)
        torch.testing.assert_close(out_weighted.pooled_style, out_binary.pooled_style)


class TestSegmentGSTValidation:
    def test_universal_style_vector_length_mismatch_raises(self, segment_gst_mod):
        with pytest.raises(ValueError, match="length .* != ref_dim"):
            segment_gst_mod.SegmentGST(universal_style_vector=torch.zeros(255))

    def test_ref_dim_mismatch_raises(self, segment_gst_mod):
        with pytest.raises(ValueError, match="ref_dim must equal 2"):
            segment_gst_mod.SegmentGST(ref_dim=300, style_dec_dim=128)

    def test_embed_dim_not_divisible_by_heads_raises(self, segment_gst_mod):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            segment_gst_mod.SegmentGST(embed_dim=255, num_heads=8)

    def test_wrong_frame_ndim_raises(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        with pytest.raises(ValueError, match="must be.*batch, time, dim"):
            gst(torch.randn(FRAME_DIM), torch.ones(T))

    def test_wrong_mask_ndim_raises(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        with pytest.raises(ValueError, match="must be.*batch, time"):
            gst(torch.randn(B, T, FRAME_DIM), torch.ones(B, T, 1))

    def test_frame_dim_mismatch_raises(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST(frame_dim=768)
        with pytest.raises(ValueError, match="last dim"):
            gst(torch.randn(B, T, 512), torch.ones(B, T))

    def test_mask_shape_mismatch_raises(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        with pytest.raises(ValueError, match="frame_mask shape"):
            gst(torch.randn(B, T, FRAME_DIM), torch.ones(B, T + 5))


class TestSegmentGSTAttentionWeights:
    def test_returns_weights_when_requested(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        _, attn_w = gst(frames, mask, need_weights=True)
        assert attn_w is not None
        # average_attn_weights=True → (B, T_q, T_kv=num_bases)
        assert attn_w.shape == (B, T, 512)

    def test_weights_sum_to_one_over_keys(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        frames = torch.randn(1, 10, FRAME_DIM)
        mask = torch.ones(1, 10)
        _, attn_w = gst(frames, mask, need_weights=True)
        sums = attn_w.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=0.02, rtol=0.02)

    def test_no_weights_when_not_requested(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        _, attn_w = gst(frames, mask, need_weights=False)
        assert attn_w is None


class TestSegmentGSTGradientFlow:
    def test_grad_flows_to_to_ref_s_through_residual_path(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST(universal_style_vector=torch.randn(256))
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        out, _ = gst(frames, mask)
        out.ref_s.sum().backward()
        assert gst.to_ref_s.weight.grad is not None
        assert gst.to_ref_s.weight.grad.abs().sum() > 0

    def test_grad_flows_to_bank(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        with torch.no_grad():
            gst.to_ref_s.weight.fill_(0.01)
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        out, _ = gst(frames, mask)
        out.ref_s.sum().backward()
        assert gst.bank.grad is not None
        assert gst.bank.grad.abs().sum() > 0

    def test_grad_flows_to_q_proj(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        with torch.no_grad():
            gst.to_ref_s.weight.fill_(0.01)
        frames = torch.randn(B, T, FRAME_DIM)
        mask = torch.ones(B, T)
        out, _ = gst(frames, mask)
        out.ref_s.sum().backward()
        assert gst.q_proj.weight.grad is not None
        assert gst.q_proj.weight.grad.abs().sum() > 0
