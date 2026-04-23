"""Tests for MHuBERTEncoder shape contract and mask derivation.

The real encoder (``voice_clone.mhubert_encoder.MHuBERTEncoder``) requires
downloading the mHuBERT checkpoint from Hugging Face, so that test is
``@pytest.mark.slow``.  The fast tests here validate the output contract
against a lightweight stub that mirrors the real encoder's interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn


HIDDEN_SIZE = 768
B, T_SAMPLES = 2, 16_000  # 1 second @16 kHz


# ---------------------------------------------------------------------------
# Lightweight stub matching MHuBERTEncoder's output contract
# ---------------------------------------------------------------------------


@dataclass
class _StubOutput:
    hidden_states: torch.Tensor
    frame_mask: torch.Tensor


class _StubMHuBERTEncoder(nn.Module):
    """Mimics MHuBERTEncoder's forward signature and output shapes."""

    hidden_size: int = HIDDEN_SIZE

    def __init__(self, frames_per_sample: int = 49):
        super().__init__()
        self._frames_per_sample = frames_per_sample

    def forward(self, waveforms_16k: torch.Tensor, attention_mask=None):
        if waveforms_16k.dim() == 1:
            waveforms_16k = waveforms_16k.unsqueeze(0)
        B, S = waveforms_16k.shape
        T = self._frames_per_sample
        hidden = torch.zeros(B, T, self.hidden_size, device=waveforms_16k.device)
        if attention_mask is not None:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=waveforms_16k.device)
            valid = attention_mask.sum(dim=-1).long()
            for i in range(B):
                n_valid_frames = min(T, max(1, int(valid[i].item() * T / S)))
                frame_mask[i, n_valid_frames:] = False
        else:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=waveforms_16k.device)
        return _StubOutput(hidden_states=hidden, frame_mask=frame_mask)


# ---------------------------------------------------------------------------
# Tests against the stub (fast, no HF download)
# ---------------------------------------------------------------------------


class TestMHuBERTOutputShapes:
    def test_hidden_states_shape(self):
        enc = _StubMHuBERTEncoder()
        out = enc(torch.randn(B, T_SAMPLES))
        assert out.hidden_states.shape == (B, 49, HIDDEN_SIZE)

    def test_frame_mask_shape_matches_hidden(self):
        enc = _StubMHuBERTEncoder()
        out = enc(torch.randn(B, T_SAMPLES))
        assert out.frame_mask.shape == out.hidden_states.shape[:2]

    def test_frame_mask_dtype_is_bool(self):
        enc = _StubMHuBERTEncoder()
        out = enc(torch.randn(B, T_SAMPLES))
        assert out.frame_mask.dtype == torch.bool

    def test_all_true_mask_without_attention_mask(self):
        enc = _StubMHuBERTEncoder()
        out = enc(torch.randn(B, T_SAMPLES))
        assert out.frame_mask.all()

    def test_partial_mask_with_attention_mask(self):
        enc = _StubMHuBERTEncoder()
        attn_mask = torch.ones(B, T_SAMPLES, dtype=torch.long)
        attn_mask[1, T_SAMPLES // 2 :] = 0
        out = enc(torch.randn(B, T_SAMPLES), attention_mask=attn_mask)
        assert out.frame_mask[0].all()
        assert not out.frame_mask[1].all()

    def test_1d_input_promoted_to_batch(self):
        enc = _StubMHuBERTEncoder()
        out = enc(torch.randn(T_SAMPLES))
        assert out.hidden_states.shape[0] == 1
        assert out.frame_mask.shape[0] == 1

    def test_hidden_size_is_768(self):
        enc = _StubMHuBERTEncoder()
        assert enc.hidden_size == 768


# ---------------------------------------------------------------------------
# Integration test against the real encoder (requires HF download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMHuBERTEncoderReal:
    """Shape validation on the real mHuBERT encoder (downloads ~380 MB)."""

    def test_forward_shapes(self):
        from voice_clone.mhubert_encoder import MHuBERTEncoder

        enc = MHuBERTEncoder()
        wav = torch.randn(1, 16_000)
        out = enc(wav)
        assert out.hidden_states.dim() == 3
        assert out.hidden_states.shape[0] == 1
        assert out.hidden_states.shape[2] == 768
        assert out.frame_mask.shape == out.hidden_states.shape[:2]
        assert out.frame_mask.dtype == torch.bool

    def test_extract_layer_out_of_range_raises(self):
        from voice_clone.mhubert_encoder import MHuBERTEncoder

        with pytest.raises(ValueError, match="extract_layer"):
            MHuBERTEncoder(extract_layer=99)

    def test_batched_with_attention_mask(self):
        from voice_clone.mhubert_encoder import MHuBERTEncoder

        enc = MHuBERTEncoder()
        wav = torch.randn(2, 16_000)
        attn = torch.ones(2, 16_000, dtype=torch.long)
        attn[1, 8_000:] = 0
        out = enc(wav, attention_mask=attn)
        assert out.hidden_states.shape[0] == 2
        assert out.frame_mask[0].all()
        assert not out.frame_mask[1].all()
