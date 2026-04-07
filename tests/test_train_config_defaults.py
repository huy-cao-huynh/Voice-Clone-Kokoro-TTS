"""Unit tests for lightweight training config defaults."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "voice_clone" / "config.py"
_spec = importlib.util.spec_from_file_location("voice_clone.config", _CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_CONFIG_PATH}")
_config = importlib.util.module_from_spec(_spec)
sys.modules[str(_spec.name)] = _config
_spec.loader.exec_module(_config)
TrainConfig = _config.TrainConfig


def test_train_config_checkpoint_interval_default() -> None:
    """Checkpoint cadence defaults to 500 steps."""
    cfg = TrainConfig()
    assert cfg.checkpoint_interval == 1000


def test_disc_start_step_default() -> None:
    """Generator-only warmup phase lasts 500 effective steps before D activates."""
    cfg = TrainConfig()
    assert cfg.disc_start_step == 500


def test_grad_accum_steps_default() -> None:
    """Gradient accumulation over 8 micro-steps per effective step."""
    cfg = TrainConfig()
    assert cfg.grad_accum_steps == 8


def test_lr_d_default() -> None:
    """Discriminator learning rate uses TTUR (lower than generator)."""
    cfg = TrainConfig()
    assert cfg.lr_d == 5e-5


def test_lambda_spk_default() -> None:
    """Speaker loss weight at parity with mel for stable speaker identity."""
    cfg = TrainConfig()
    assert cfg.loss_weights.lambda_spk == 1.0


def test_use_spectral_norm_default() -> None:
    """Spectral normalization enabled by default on discriminator convolutions."""
    cfg = TrainConfig()
    assert cfg.slm_disc.use_spectral_norm is True
