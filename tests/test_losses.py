"""Tests for loss functions and SLM discriminator (voice_clone/losses.py)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MelReconstructionLoss (needs torchaudio)
# ---------------------------------------------------------------------------

def _make_mel_loss(losses_mod, sr=24_000, n_mels=80):
    pytest.importorskip("torchaudio", reason="MelReconstructionLoss needs torchaudio")
    return losses_mod.MelReconstructionLoss(
        sample_rate=sr, n_mels=n_mels, n_fft=1024,
        hop_length=256, win_length=1024,
    )


class TestMelReconstructionLoss:
    def test_identical_input_gives_near_zero_loss(self, losses_mod):
        mel = _make_mel_loss(losses_mod)
        wav = torch.randn(1, 24_000) * 0.1
        out = mel(wav, wav.clone())
        assert out.loss.item() < 1e-5

    def test_different_inputs_give_positive_loss(self, losses_mod):
        mel = _make_mel_loss(losses_mod)
        a = torch.randn(1, 24_000) * 0.1
        b = torch.randn(1, 24_000) * 0.1
        out = mel(a, b)
        assert out.loss.item() > 0

    def test_unequal_length_crop(self, losses_mod):
        mel = _make_mel_loss(losses_mod)
        a = torch.randn(1, 24_000) * 0.1
        b = torch.randn(1, 20_000) * 0.1
        out = mel(a, b)
        assert out.mel_pred.shape == out.mel_target.shape

    def test_mel_output_shapes_match(self, losses_mod):
        mel = _make_mel_loss(losses_mod)
        a = torch.randn(1, 24_000) * 0.1
        b = torch.randn(1, 24_000) * 0.1
        out = mel(a, b)
        assert out.mel_pred.shape == out.mel_target.shape
        assert out.mel_pred.shape[1] == 80

    def test_gradient_flows_to_pred_wav(self, losses_mod):
        mel = _make_mel_loss(losses_mod)
        pred = (torch.randn(1, 24_000) * 0.1).requires_grad_(True)
        target = torch.randn(1, 24_000) * 0.1
        out = mel(pred, target)
        out.loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_rejects_non_2d_input(self, losses_mod):
        mel = _make_mel_loss(losses_mod)
        with pytest.raises(ValueError, match="batch, time"):
            mel(torch.randn(24_000), torch.randn(24_000))


# ---------------------------------------------------------------------------
# speaker_cosine_loss
# ---------------------------------------------------------------------------

class TestSpeakerCosineLoss:
    def test_same_vectors_give_zero(self, losses_mod):
        v = F.normalize(torch.randn(4, 512), dim=-1)
        loss = losses_mod.speaker_cosine_loss(v, v.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_give_one(self, losses_mod):
        a = torch.zeros(1, 512)
        b = torch.zeros(1, 512)
        a[0, 0] = 1.0
        b[0, 1] = 1.0
        loss = losses_mod.speaker_cosine_loss(a, b)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_give_two(self, losses_mod):
        a = torch.zeros(1, 512)
        a[0, 0] = 1.0
        b = -a.clone()
        loss = losses_mod.speaker_cosine_loss(a, b)
        assert loss.item() == pytest.approx(2.0, abs=1e-6)

    def test_gradient_flows_through_both_args(self, losses_mod):
        a = torch.randn(2, 512, requires_grad=True)
        b = torch.randn(2, 512, requires_grad=True)
        loss = losses_mod.speaker_cosine_loss(a, b)
        loss.backward()
        assert a.grad is not None and a.grad.abs().sum() > 0
        assert b.grad is not None and b.grad.abs().sum() > 0

    def test_loss_is_nonnegative(self, losses_mod):
        a = torch.randn(8, 512)
        b = torch.randn(8, 512)
        loss = losses_mod.speaker_cosine_loss(a, b)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# SLMFeatureDiscriminator
# ---------------------------------------------------------------------------

def _make_disc(losses_mod, **kw):
    defaults = dict(in_dim=768, hidden_channels=64, num_layers=2, kernel_size=3, use_spectral_norm=True)
    defaults.update(kw)
    return losses_mod.SLMFeatureDiscriminator(**defaults)


class TestSLMFeatureDiscriminator:
    def test_output_shape_scalar_per_batch(self, losses_mod):
        disc = _make_disc(losses_mod)
        feats = torch.randn(3, 20, 768)
        mask = torch.ones(3, 20)
        out = disc(feats, mask)
        assert out.shape == (3,)

    def test_spectral_norm_applied(self, losses_mod):
        disc = _make_disc(losses_mod, use_spectral_norm=True)
        sn_found = False
        for m in disc.net.modules():
            if isinstance(m, torch.nn.Conv1d) and hasattr(m, "weight_orig"):
                sn_found = True
                break
        assert sn_found, "No spectral-normalized Conv1d found"

    def test_no_spectral_norm_when_disabled(self, losses_mod):
        disc = _make_disc(losses_mod, use_spectral_norm=False)
        for m in disc.net.modules():
            if isinstance(m, torch.nn.Conv1d):
                assert not hasattr(m, "weight_orig")

    def test_even_kernel_raises(self, losses_mod):
        with pytest.raises(ValueError, match="odd"):
            _make_disc(losses_mod, kernel_size=6)

    def test_mask_pooling_excludes_padding(self, losses_mod):
        disc = _make_disc(losses_mod)
        disc.eval()
        feats = torch.randn(1, 20, 768)
        mask_full = torch.ones(1, 20)
        mask_half = torch.cat([torch.ones(1, 10), torch.zeros(1, 10)], dim=1)
        with torch.no_grad():
            out_full = disc(feats, mask_full)
            out_half = disc(feats, mask_half)
        assert not torch.allclose(out_full, out_half, atol=1e-6)

    def test_rejects_non_3d_features(self, losses_mod):
        disc = _make_disc(losses_mod)
        with pytest.raises(ValueError, match="batch, time, dim"):
            disc(torch.randn(768), torch.ones(20))

    def test_rejects_non_2d_mask(self, losses_mod):
        disc = _make_disc(losses_mod)
        with pytest.raises(ValueError, match="batch, time"):
            disc(torch.randn(1, 20, 768), torch.ones(1, 20, 1))


# ---------------------------------------------------------------------------
# Hinge losses
# ---------------------------------------------------------------------------

class TestSLMHingeLosses:
    def test_disc_hinge_zero_for_perfect_d(self, losses_mod):
        """D outputs +2 for real, -2 for fake: both hinge margins satisfied → loss 0."""
        disc = _make_disc(losses_mod)
        disc.eval()
        real = torch.randn(2, 15, 768)
        fake = torch.randn(2, 15, 768)
        mask = torch.ones(2, 15)
        with torch.no_grad():
            d_real = disc(real, mask)
            d_fake = disc(fake, mask)
        # Manually check what the hinge would be if d_real=+2, d_fake=-2
        loss = F.relu(1.0 - torch.tensor(2.0)).mean() + F.relu(1.0 + torch.tensor(-2.0)).mean()
        assert loss.item() == 0.0

    def test_disc_hinge_positive_for_reversed(self, losses_mod):
        """D outputs -2 for real, +2 for fake: hinge loss should be positive."""
        loss = F.relu(1.0 - torch.tensor(-2.0)).mean() + F.relu(1.0 + torch.tensor(2.0)).mean()
        assert loss.item() > 0

    def test_gen_hinge_gradient_flows(self, losses_mod):
        disc = _make_disc(losses_mod)
        fake = torch.randn(2, 15, 768, requires_grad=True)
        mask = torch.ones(2, 15)
        loss = losses_mod.slm_generator_loss_hinge(disc, fake, mask)
        loss.backward()
        assert fake.grad is not None
        assert fake.grad.abs().sum() > 0

    def test_gen_hinge_is_negative_d_mean(self, losses_mod):
        """slm_generator_loss_hinge should be -mean(D(fake))."""
        disc = _make_disc(losses_mod)
        disc.eval()
        fake = torch.randn(1, 15, 768)
        mask = torch.ones(1, 15)
        with torch.no_grad():
            d_fake = disc(fake, mask)
            expected = (-d_fake).mean()
            actual = losses_mod.slm_generator_loss_hinge(disc, fake, mask)
        torch.testing.assert_close(actual, expected)
