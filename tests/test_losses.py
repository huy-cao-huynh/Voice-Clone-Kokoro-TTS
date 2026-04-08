"""Tests for loss functions in `voice_clone/losses.py`."""

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


class TestSpeakerBackpropPath:
    def test_speaker_path_backprops_to_pred_wav(self, losses_mod):
        pytest.importorskip("torchaudio", reason="speaker mel path requires torchaudio")
        import torchaudio

        class TinyFrozenSpeaker(torch.nn.Module):
            def __init__(self, n_mels: int, emb_dim: int) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(n_mels, emb_dim, bias=False)
                for p in self.parameters():
                    p.requires_grad_(False)

            def forward(self, mel: torch.Tensor) -> torch.Tensor:
                # mel: (B, n_mels, T)
                pooled = mel.mean(dim=-1)
                return self.proj(pooled)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24_000,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            power=1.0,
        )
        pred_wav = torch.randn(2, 24_000, requires_grad=True)
        ref_wav = torch.randn(2, 24_000)
        speaker = TinyFrozenSpeaker(n_mels=80, emb_dim=256)

        mel_pred = losses_mod.speaker_input_mel_from_waveform(
            pred_wav,
            mel_transform=mel_transform,
            amp_enabled=False,
            disable_amp_for_stft=True,
        )
        mel_ref = losses_mod.speaker_input_mel_from_waveform(
            ref_wav,
            mel_transform=mel_transform,
            amp_enabled=False,
            disable_amp_for_stft=True,
        )
        emb_pred = speaker(mel_pred)
        with torch.no_grad():
            emb_ref = speaker(mel_ref)

        loss = losses_mod.speaker_cosine_loss(emb_ref, emb_pred)
        loss.backward()

        assert pred_wav.grad is not None
        assert pred_wav.grad.abs().sum() > 0


