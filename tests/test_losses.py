"""Tests for loss functions in `voice_clone/losses.py`."""

from __future__ import annotations

import pytest
import torch


def _make_mel_loss(losses_mod):
    pytest.importorskip("torchaudio", reason="MelReconstructionLoss needs torchaudio")
    return losses_mod.MelReconstructionLoss(
        sample_rate=24_000,
        n_mels=80,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    )


def test_mel_loss_backprops(losses_mod):
    mel = _make_mel_loss(losses_mod)
    pred = torch.randn(1, 24_000, requires_grad=True)
    target = torch.randn(1, 24_000)
    out = mel(pred, target)
    out.loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_speaker_contrastive_loss_uses_diagonal_positives(losses_mod):
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    loss = losses_mod.speaker_contrastive_loss(anchors, targets, temperature=0.1)
    loss.backward()
    assert anchors.grad is not None
    assert anchors.grad.abs().sum() > 0
    assert targets.grad is None


def test_speaker_contrastive_loss_requires_batch_two(losses_mod):
    with pytest.raises(ValueError, match="batch size >= 2"):
        losses_mod.speaker_contrastive_loss(torch.randn(1, 4), torch.randn(1, 4))


def test_duration_loss_log_space_respects_mask(losses_mod):
    pred = torch.tensor([[1.0, 9.0]], requires_grad=True)
    target = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[True, False]])
    loss = losses_mod.duration_loss_log_space(pred, target, mask)
    loss.backward()
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert pred.grad is not None


def test_masked_l1_loss_ignores_masked_positions(losses_mod):
    pred = torch.tensor([[2.0, 10.0]])
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])
    loss = losses_mod.masked_l1_loss(pred, target, mask)
    assert loss.item() == pytest.approx(1.0, abs=1e-6)
