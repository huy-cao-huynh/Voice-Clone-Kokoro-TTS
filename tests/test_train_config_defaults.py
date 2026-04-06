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
    """Checkpoint cadence defaults to 10k steps."""
    cfg = TrainConfig()
    assert cfg.checkpoint_interval == 10_000
