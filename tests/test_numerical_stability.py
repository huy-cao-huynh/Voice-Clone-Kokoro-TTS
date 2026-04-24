"""Numerical stability tests for the new loss and style stack."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

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


@pytest.fixture(scope="module")
def segment_gst_mod():
    return _load_module("voice_clone.segment_gst", "voice_clone/segment_gst.py")


@pytest.fixture(scope="module")
def losses_mod():
    return _load_module("voice_clone.losses", "voice_clone/losses.py")


def test_segment_gst_zero_mask_stays_finite(segment_gst_mod):
    gst = segment_gst_mod.SegmentGST()
    out, _ = gst(torch.randn(1, 10, 768), torch.zeros(1, 10))
    assert torch.isfinite(out.ref_s).all()
    assert torch.isfinite(out.pooled_style).all()


def test_contrastive_loss_zero_vectors_finite(losses_mod):
    anchors = torch.zeros(2, 8, requires_grad=True)
    targets = torch.zeros(2, 8)
    loss = losses_mod.speaker_contrastive_loss(anchors, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert anchors.grad is not None
    assert torch.isfinite(anchors.grad).all()


def test_duration_log_loss_large_values_finite(losses_mod):
    pred = torch.full((1, 4), 1e6, requires_grad=True)
    target = torch.ones(1, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)
    loss = losses_mod.duration_loss_log_space(pred, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
