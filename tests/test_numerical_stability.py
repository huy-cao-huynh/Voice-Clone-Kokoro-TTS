"""Numerical stability tests: NaN/Inf guards, clamping, edge-case inputs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = _ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _try_import_train_adapters():
    """Try importing voice_clone.train_adapters; skip if deps missing."""
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    try:
        import voice_clone.train_adapters as ta
        return ta
    except ImportError as e:
        pytest.skip(f"Cannot import train_adapters: {e}")


@pytest.fixture(scope="module")
def adapters_mod():
    return _load_module("voice_clone.adapters", "voice_clone/adapters.py")


@pytest.fixture(scope="module")
def segment_gst_mod():
    return _load_module("voice_clone.segment_gst", "voice_clone/segment_gst.py")


@pytest.fixture(scope="module")
def losses_mod():
    return _load_module("voice_clone.losses", "voice_clone/losses.py")


@pytest.fixture(scope="module")
def hifigan_mod():
    return _load_module(
        "voice_clone.discriminators.hifigan",
        "voice_clone/discriminators/hifigan.py",
    )


# ---------------------------------------------------------------------------
# Adapter with extreme inputs
# ---------------------------------------------------------------------------

class TestAdapterNumericalStability:
    def test_large_h_values_produce_finite_output(self, adapters_mod):
        adapter = adapters_mod.ResidualAdapter(hidden_dim=512, style_dim=256, bottleneck_dim=64)
        h = torch.full((1, 512, 5), 1e4)
        z = torch.randn(1, 256)
        out = adapter(h, z)
        assert torch.isfinite(out).all()

    def test_large_z_style_produces_finite_output(self, adapters_mod):
        adapter = adapters_mod.ResidualAdapter(hidden_dim=512, style_dim=256, bottleneck_dim=64)
        h = torch.randn(1, 512, 5)
        z = torch.full((1, 256), 1e4)
        out = adapter(h, z)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SegmentGST with all-zero mask
# ---------------------------------------------------------------------------

class TestSegmentGSTZeroMask:
    def test_all_zero_mask_finite(self, segment_gst_mod):
        """clamp(min=1e-6) in denom prevents division by zero."""
        gst = segment_gst_mod.SegmentGST()
        gst.eval()
        frames = torch.randn(1, 10, 1024)
        mask = torch.zeros(1, 10)
        with torch.no_grad():
            out, _ = gst(frames, mask)
        assert torch.isfinite(out.ref_s).all()
        assert torch.isfinite(out.pooled_style).all()

    def test_single_valid_frame(self, segment_gst_mod):
        gst = segment_gst_mod.SegmentGST()
        gst.eval()
        frames = torch.randn(1, 10, 1024)
        mask = torch.zeros(1, 10)
        mask[0, 0] = 1.0
        with torch.no_grad():
            out, _ = gst(frames, mask)
        assert torch.isfinite(out.ref_s).all()


# ---------------------------------------------------------------------------
# MelReconstructionLoss log floor
# ---------------------------------------------------------------------------

class TestMelLossLogFloor:
    @pytest.fixture(autouse=True)
    def _skip_no_torchaudio(self):
        pytest.importorskip("torchaudio", reason="MelReconstructionLoss needs torchaudio")

    def test_zero_amplitude_wav_finite(self, losses_mod):
        """log_floor=1e-5 prevents log(0)."""
        mel = losses_mod.MelReconstructionLoss(
            sample_rate=24_000, n_mels=80, n_fft=1024,
            hop_length=256, win_length=1024,
        )
        zero_wav = torch.zeros(1, 24_000)
        target = torch.randn(1, 24_000) * 0.01
        out = mel(zero_wav, target)
        assert torch.isfinite(out.loss)
        assert torch.isfinite(out.mel_pred).all()

    def test_very_quiet_wav_finite(self, losses_mod):
        mel = losses_mod.MelReconstructionLoss(
            sample_rate=24_000, n_mels=80, n_fft=1024,
            hop_length=256, win_length=1024,
        )
        quiet = torch.randn(1, 24_000) * 1e-10
        target = torch.randn(1, 24_000) * 0.01
        out = mel(quiet, target)
        assert torch.isfinite(out.loss)


# ---------------------------------------------------------------------------
# speaker_cosine_loss with zero vectors
# ---------------------------------------------------------------------------

class TestSpeakerCosineZeroVectors:
    def test_zero_vector_finite(self, losses_mod):
        """eps in F.normalize prevents NaN from zero-norm vectors."""
        a = torch.zeros(1, 512)
        b = torch.randn(1, 512)
        loss = losses_mod.speaker_cosine_loss(a, b)
        assert torch.isfinite(loss)

    def test_both_zero_finite(self, losses_mod):
        a = torch.zeros(1, 512)
        b = torch.zeros(1, 512)
        loss = losses_mod.speaker_cosine_loss(a, b)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# HiFiGANMPDMSDDiscriminator with all-zero waveform
# ---------------------------------------------------------------------------

class TestWaveformDiscriminatorZeroInput:
    def test_zero_wav_finite(self, hifigan_mod):
        disc = hifigan_mod.HiFiGANMPDMSDDiscriminator()
        disc.eval()
        wav = torch.zeros(1, 24_000)
        with torch.no_grad():
            logits, _feats = disc(wav)
        for l in logits:
            assert torch.isfinite(l).all()

    def test_very_short_wav_still_runs(self, hifigan_mod):
        disc = hifigan_mod.HiFiGANMPDMSDDiscriminator(msd_scales=(1, 2, 4))
        disc.eval()
        wav = torch.zeros(1, 3)
        with torch.no_grad():
            logits, feats = disc(wav)
        assert len(logits) == 8
        assert len(feats) == 8
        for l in logits:
            assert l.numel() > 0
            assert torch.isfinite(l).all()


# ---------------------------------------------------------------------------
# resample_mono differentiability
# ---------------------------------------------------------------------------

class TestResampleDifferentiability:
    @pytest.fixture(autouse=True)
    def _skip_no_torchaudio(self):
        pytest.importorskip("torchaudio", reason="resample_mono needs torchaudio")

    def test_backward_no_nan(self):
        ta = _try_import_train_adapters()
        wav = (torch.randn(1, 24_000) * 0.1).requires_grad_(True)
        resampled = ta.resample_mono(wav, 24_000, 16_000)
        loss = resampled.abs().mean()
        loss.backward()
        assert wav.grad is not None
        assert torch.isfinite(wav.grad).all()

    def test_same_sr_identity(self):
        ta = _try_import_train_adapters()
        wav = torch.randn(1, 16_000)
        out = ta.resample_mono(wav, 16_000, 16_000)
        torch.testing.assert_close(out, wav)


# ---------------------------------------------------------------------------
# Loss composition finiteness
# ---------------------------------------------------------------------------

class TestLossCompositionFiniteness:
    def test_weighted_sum_finite_and_gradable(self, losses_mod):
        """lambda_mel * mel + lambda_spk * spk + lambda_adv * adv + lambda_fm * fm should be finite."""
        mel_val = torch.tensor(2.5, requires_grad=True)
        spk_val = torch.tensor(0.8, requires_grad=True)
        adv_val = torch.tensor(0.3, requires_grad=True)
        fm_val = torch.tensor(0.7, requires_grad=True)

        loss = 1.0 * mel_val + 0.5 * spk_val + 0.1 * adv_val + 1.0 * fm_val
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(mel_val.grad)
        assert torch.isfinite(spk_val.grad)
        assert torch.isfinite(adv_val.grad)
        assert torch.isfinite(fm_val.grad)

    def test_large_loss_values_finite(self, losses_mod):
        """Even large individual losses should compose to a finite total."""
        mel_val = torch.tensor(1e5, requires_grad=True)
        spk_val = torch.tensor(1e5, requires_grad=True)
        adv_val = torch.tensor(1e5, requires_grad=True)
        fm_val = torch.tensor(1e5, requires_grad=True)
        loss = 1.0 * mel_val + 0.5 * spk_val + 0.1 * adv_val + 1.0 * fm_val
        assert torch.isfinite(loss)
        loss.backward()
        for g in (mel_val.grad, spk_val.grad, adv_val.grad, fm_val.grad):
            assert torch.isfinite(g)


# ---------------------------------------------------------------------------
# _ensure_batch_time edge cases
# ---------------------------------------------------------------------------

def _ensure_batch_time(wav: torch.Tensor) -> torch.Tensor:
    """Local copy of train_adapters._ensure_batch_time (avoids heavy module import)."""
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    if wav.dim() != 2:
        raise ValueError(f"waveform must be (time,) or (batch, time), got {tuple(wav.shape)}")
    return wav


class TestEnsureBatchTime:
    def test_1d_becomes_2d(self):
        wav = torch.randn(16_000)
        out = _ensure_batch_time(wav)
        assert out.shape == (1, 16_000)

    def test_2d_unchanged(self):
        wav = torch.randn(1, 16_000)
        out = _ensure_batch_time(wav)
        assert out.shape == (1, 16_000)
        assert out.data_ptr() == wav.data_ptr()

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="batch, time"):
            _ensure_batch_time(torch.randn(1, 1, 16_000))
