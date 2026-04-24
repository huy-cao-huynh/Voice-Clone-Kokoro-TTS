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


def test_train_config_defaults_match_contrastive_stack() -> None:
    cfg = TrainConfig()
    assert cfg.batch_size == 2
    assert cfg.feature_cache_root == "cache"
    assert cfg.contrastive_temperature == 0.07
    assert cfg.validate_cache_freshness is True
    assert cfg.min_language_speakers == 2
    assert cfg.loss_weights.lambda_spk_contrastive == 1.0
    assert cfg.loss_weights.lambda_dur == 1.0
    assert cfg.loss_weights.lambda_f0 == 1.0


def test_wespeaker_defaults_unchanged() -> None:
    cfg = TrainConfig()
    assert cfg.wespeaker_sample_rate == 16_000
    assert cfg.wespeaker_checkpoint_path.endswith("avg_model.pt")
