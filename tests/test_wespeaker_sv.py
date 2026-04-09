"""Unit tests for WeSpeakerSV (dummy encoder + mocks; no real WeSpeaker checkpoint)."""

from __future__ import annotations

import builtins
import importlib
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

torchaudio = pytest.importorskip("torchaudio")
pytest.importorskip("wespeaker")

from voice_clone.wespeaker_sv import WeSpeakerSV, WeSpeakerSVOutput


class DummySpeakerEncoder(nn.Module):
    """Minimal encoder: differentiable map (B,1,n_mels,T) -> pooled (B,D) and frame (B,T,n_mels)."""

    def __init__(self, n_mels: int = 80, embedding_dim: int = 256) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"expected (B, 1, n_mels, T), got {tuple(x.shape)}")
        b = x.size(0)
        frame = x.squeeze(1).permute(0, 2, 1).contiguous()
        flat = x.reshape(b, -1)
        d = self.embedding_dim
        if flat.size(1) >= d:
            pooled = flat[:, :d]
        else:
            pooled = F.pad(flat, (0, d - flat.size(1)))
        return pooled, frame


def make_model(*, embedding_dim: int = 256, sample_rate: int = 16_000) -> WeSpeakerSV:
    enc = DummySpeakerEncoder(n_mels=80, embedding_dim=embedding_dim)
    return WeSpeakerSV(
        enc,
        sample_rate=sample_rate,
        embedding_dim=embedding_dim,
        n_mels=80,
    )


class TestStrictLoadingNoFallback:
    def test_from_checkpoint_raises_file_not_found_for_fake_pt_path(self, tmp_path) -> None:
        fake_pt = tmp_path / "nope.pt"
        fake_pt.parent.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="config.yaml"):
            WeSpeakerSV.from_checkpoint(str(fake_pt), embedding_dim=256)

    def test_from_checkpoint_raises_file_not_found_for_empty_model_dir(self, tmp_path) -> None:
        model_dir = tmp_path / "empty_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="config.yaml"):
            WeSpeakerSV.from_checkpoint(str(model_dir), embedding_dim=256)

    def test_reimport_raises_import_error_when_wespeaker_unavailable(self) -> None:
        """Simulate missing wespeaker at import time; ensures no silent fallback in this module."""
        mod_name = "voice_clone.wespeaker_sv"
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "wespeaker" or name.startswith("wespeaker."):
                raise ImportError("simulated: no module named 'wespeaker'")
            return real_import(name, globals, locals, fromlist, level)

        try:
            del sys.modules[mod_name]
            builtins.__import__ = guarded_import
            with pytest.raises(ImportError) as excinfo:
                importlib.import_module(mod_name)
            msg = str(excinfo.value).lower()
            assert "wespeaker" in msg
            assert "pip install" in msg
        finally:
            builtins.__import__ = real_import
            importlib.import_module(mod_name)


class TestMelPathDifferentiability:
    def test_waveform_grad_flows_through_forward(self) -> None:
        model = make_model()
        wav = torch.randn(1, 16_000, dtype=torch.float32, requires_grad=True)
        out = model.forward(
            wav,
            sampling_rate=16_000,
            grad_through_input=True,
            return_frame_features=False,
        )
        assert isinstance(out, WeSpeakerSVOutput)
        loss = out.pooled_embedding.sum()
        loss.backward()
        assert wav.grad is not None
        assert wav.grad.shape == wav.shape
        assert torch.isfinite(wav.grad).all()


class TestOutputShapesAndNormalization:
    def test_pooled_shape_and_l2_norm(self) -> None:
        model = make_model(embedding_dim=256)
        wav = torch.randn(2, 8000, dtype=torch.float32)
        out = model.forward(wav, sampling_rate=16_000, normalize_embeddings=True)
        assert out.pooled_embedding.shape == (2, 256)
        norms = out.pooled_embedding.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5, rtol=1e-5)


class TestResamplingLogic:
    def test_24khz_triggers_resample_to_16khz(self) -> None:
        model = make_model()
        wav24 = torch.randn(1, 24_000, dtype=torch.float32)
        with patch("torchaudio.functional.resample", wraps=torchaudio.functional.resample) as mock_resample:
            out = model.forward(
                wav24,
                sampling_rate=24_000,
                grad_through_input=False,
                return_frame_features=False,
            )
        mock_resample.assert_called_once()
        call_kw = mock_resample.call_args.kwargs
        assert call_kw.get("orig_freq") == 24_000
        assert call_kw.get("new_freq") == 16_000
        assert out.pooled_embedding.shape == (1, 256)


class TestFrameMaskOutput:
    def test_sequence_batch_returns_frame_mask(self) -> None:
        model = make_model()
        wavs = [torch.randn(16_000), torch.randn(8_000)]
        out = model.forward(wavs, sampling_rate=16_000, grad_through_input=False)
        assert out.frame_features is not None
        assert out.frame_mask is not None
        assert out.frame_mask.dtype == torch.bool
        assert out.frame_mask.shape == out.frame_features.shape[:2]
        assert out.frame_mask[0].all()
        assert not out.frame_mask[1].all()
        assert out.frame_mask[1].any()

    def test_explicit_waveform_lengths_override_padded_tensor(self) -> None:
        model = make_model()
        wav = torch.randn(2, 16_000)
        lengths = torch.tensor([16_000, 4_000])
        out = model.forward(
            wav,
            sampling_rate=16_000,
            waveform_lengths=lengths,
            grad_through_input=False,
        )
        assert out.frame_mask is not None
        assert out.frame_mask[0].all()
        assert not out.frame_mask[1].all()
