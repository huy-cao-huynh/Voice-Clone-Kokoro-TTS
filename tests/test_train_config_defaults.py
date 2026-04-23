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
    """Checkpoint cadence defaults to 100 steps."""
    cfg = TrainConfig()
    assert cfg.checkpoint_interval == 100


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
    assert cfg.loss_weights.lambda_spk == 5.0


def test_wespeaker_sample_rate_default() -> None:
    """WeSpeaker sample rate defaults to 16 kHz."""
    cfg = TrainConfig()
    assert cfg.wespeaker_sample_rate == 16_000


def test_wespeaker_checkpoint_path_default() -> None:
    """WeSpeaker toolkit weights default to the bundled encoder checkpoint layout."""
    cfg = TrainConfig()
    assert cfg.wespeaker_checkpoint_path == "voice_clone/encoder-ckpts/wespeaker-ckpt/models/avg_model.pt"


def test_universal_style_vector_path_default() -> None:
    """Universal style vector path points to the canonical checkpoint."""
    cfg = TrainConfig()
    assert cfg.universal_style_vector_path == "voice_clone/universal_style_vector.pt"


def test_grad_clip_norm_defaults() -> None:
    """Gradient clipping enabled by default for both generator and discriminator."""
    cfg = TrainConfig()
    assert cfg.grad_clip_norm_g == 5.0
    assert cfg.grad_clip_norm_d == 1.0


def test_weight_decay_g_default() -> None:
    """Generator weight decay is disabled by default."""
    cfg = TrainConfig()
    assert cfg.weight_decay_g == 0.0


def test_adam_beta_defaults() -> None:
    """AdamW defaults use HiFi-GAN-style betas."""
    cfg = TrainConfig()
    assert cfg.adam_b1 == 0.8
    assert cfg.adam_b2 == 0.99


def test_disable_amp_for_stft_default() -> None:
    """AMP for STFT enabled by default; disable manually if fp16 causes numerical issues."""
    cfg = TrainConfig()
    assert cfg.disable_amp_for_stft is False


def test_mhubert_repo_id_default() -> None:
    """mHuBERT conditioning encoder defaults to utter-project multilingual HuBERT."""
    cfg = TrainConfig()
    assert cfg.mhubert_repo_id == "utter-project/mHuBERT-147-base-3rd-iter"


def test_mhubert_extract_layer_default() -> None:
    """Hidden-state extraction from transformer layer 9 (0-indexed, includes CNN projection at 0)."""
    cfg = TrainConfig()
    assert cfg.mhubert_extract_layer == 9


def test_batch_size_default() -> None:
    """Batch size defaults to 4 for adapter training."""
    cfg = TrainConfig()
    assert cfg.batch_size == 4
