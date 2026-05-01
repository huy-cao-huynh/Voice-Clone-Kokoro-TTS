"""Unit tests for SegmentGST internal diagnostic helpers in train_adapters."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _stub_train_modules(monkeypatch):
    """Stub out all heavy deps so train_adapters and segment_gst load cleanly."""
    from dataclasses import dataclass, field

    voice_clone_pkg = types.ModuleType("voice_clone")
    voice_clone_pkg.__path__ = [str(_ROOT / "voice_clone")]
    monkeypatch.setitem(sys.modules, "voice_clone", voice_clone_pkg)

    kokoro_pkg = types.ModuleType("kokoro")
    kokoro_pkg.__path__ = [str(_ROOT / "kokoro")]
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_pkg)

    disc_pkg = types.ModuleType("voice_clone.discriminators")
    disc_pkg.__path__ = [str(_ROOT / "voice_clone" / "discriminators")]
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators", disc_pkg)

    @dataclass
    class LossWeights:
        lambda_mel: float = 20.0
        lambda_spk_contrastive: float = 5.0
        lambda_adv: float = 1.0
        lambda_fm: float = 2.0
        lambda_dur: float = 0.0
        lambda_f0: float = 0.0

    @dataclass
    class MelLossConfig:
        sample_rate: int = 24_000
        n_fft: int = 1024
        hop_length: int = 256
        win_length: int = 1024
        f_min: float = 0.0
        f_max: float | None = None

    @dataclass
    class TrainConfig:
        kokoro_repo_id: str = "dummy/kokoro"
        mhubert_repo_id: str = "dummy/mhubert"
        mhubert_extract_layer: int = 6
        wespeaker_checkpoint_path: str = "dummy/wespeaker/avg_model.pt"
        wespeaker_embedding_dim: int = 256
        wespeaker_sample_rate: int = 16_000
        universal_style_vector_path: str = "voice_clone/universal_style_vector.pt"
        feature_cache_root: str = "cache"
        disable_amp_for_stft: bool = True
        gst_embed_dim: int = 1024
        loss_weights: LossWeights = field(default_factory=LossWeights)
        mel: MelLossConfig = field(default_factory=MelLossConfig)
        contrastive_temperature: float = 0.1
        validate_cache_freshness: bool = True
        min_language_speakers: int = 2
        lr_g: float = 1e-4
        lr_d: float = 5e-5
        adam_b1: float = 0.8
        adam_b2: float = 0.99
        weight_decay_g: float = 0.0
        weight_decay_d: float = 0.0
        use_amp: bool = False
        log_interval: int = 1
        checkpoint_interval: int = 100
        save_final_checkpoint: bool = False
        warmup_steps: int = 200
        batch_size: int = 6
        grad_accum_steps: int = 1
        disc_start_step: int = 99_999_999
        speed: float = 1.0
        style_decoder_only_steps: int = 0
        gst_dropout: float = 0.0
        gst_conv_kernel_size: int = 5
        gst_conv_stride: int = 2
        gst_conv_padding: int = 2
        grad_clip_norm_g: float = 1.0
        grad_clip_norm_d: float = 1.0
        lr_min_g: float = 1e-4
        lr_min_d: float = 5e-5

    config_mod = types.ModuleType("voice_clone.config")
    config_mod.LossWeights = LossWeights
    config_mod.MelLossConfig = MelLossConfig
    config_mod.TrainConfig = TrainConfig
    config_mod.kokoro_vocab_and_context_length = lambda repo_id: ({}, 0)
    config_mod.load_kokoro_config = lambda repo_id: {"hidden_dim": 4, "n_layer": 2, "n_mels": 80, "vocab": {}}
    monkeypatch.setitem(sys.modules, "voice_clone.config", config_mod)

    dataset_mod = types.ModuleType("voice_clone.dataset")
    dataset_mod.VoiceCloneManifestDataset = object
    dataset_mod.collate_voice_clone_batch = lambda samples: samples
    monkeypatch.setitem(sys.modules, "voice_clone.dataset", dataset_mod)

    losses_mod = types.ModuleType("voice_clone.losses")
    losses_mod.MelReconstructionLoss = nn.Identity
    losses_mod.discriminator_loss_lsgan = lambda *a, **kw: torch.tensor(0.0)
    losses_mod.duration_loss_log_space = lambda *a, **kw: torch.tensor(0.0)
    losses_mod.feature_matching_loss = lambda *a, **kw: torch.tensor(0.0)
    losses_mod.generator_loss_lsgan = lambda *a, **kw: torch.tensor(0.0)
    losses_mod.masked_l1_loss = lambda *a, **kw: torch.tensor(0.0)
    losses_mod.speaker_contrastive_loss = lambda *a, **kw: torch.tensor(0.0)
    monkeypatch.setitem(sys.modules, "voice_clone.losses", losses_mod)

    mhubert_mod = types.ModuleType("voice_clone.mhubert_encoder")
    mhubert_mod.MHuBERTEncoder = lambda *a, **kw: types.SimpleNamespace(hidden_size=768, to=lambda self, d: self)
    monkeypatch.setitem(sys.modules, "voice_clone.mhubert_encoder", mhubert_mod)

    wespeaker_mod = types.ModuleType("voice_clone.wespeaker_sv")
    wespeaker_mod.WeSpeakerSV = types.SimpleNamespace(from_checkpoint=lambda *a, **kw: nn.Identity())
    monkeypatch.setitem(sys.modules, "voice_clone.wespeaker_sv", wespeaker_mod)

    disc_mod = types.ModuleType("voice_clone.discriminators.hifigan")
    disc_mod.HiFiGANMPDMSDDiscriminator = lambda *a, **kw: nn.Linear(1, 1)
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators.hifigan", disc_mod)

    kokoro_model_mod = types.ModuleType("kokoro.model")
    kokoro_model_mod.KModel = lambda **kw: nn.Linear(1, 1)
    monkeypatch.setitem(sys.modules, "kokoro.model", kokoro_model_mod)

    _load_module(monkeypatch, "voice_clone.segment_gst", _ROOT / "voice_clone" / "segment_gst.py")
    train_mod = _load_module(monkeypatch, "voice_clone.train_adapters", _ROOT / "voice_clone" / "train_adapters.py")
    return train_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T_Q, NUM_BASES = 4, 6, 32
EMBED_DIM, STYLE_DEC_DIM = 64, 32


@pytest.fixture
def train_mod(monkeypatch):
    return _stub_train_modules(monkeypatch)


@pytest.fixture
def synthetic_pooled():
    torch.manual_seed(0)
    return torch.randn(B, EMBED_DIM)


@pytest.fixture
def synthetic_style_dec():
    torch.manual_seed(1)
    return torch.randn(B, STYLE_DEC_DIM)


@pytest.fixture
def synthetic_style_pred():
    torch.manual_seed(2)
    return torch.randn(B, STYLE_DEC_DIM)


@pytest.fixture
def synthetic_attn_weights():
    # Produce a valid softmax distribution over NUM_BASES
    torch.manual_seed(3)
    raw = torch.randn(B, T_Q, NUM_BASES)
    return raw.softmax(dim=-1)


# ---------------------------------------------------------------------------
# _gst_internals_diagnostics
# ---------------------------------------------------------------------------

def test_internals_diagnostics_returns_finite_floats(train_mod, synthetic_pooled, synthetic_style_dec, synthetic_style_pred):
    result = train_mod._gst_internals_diagnostics(synthetic_pooled, synthetic_style_dec, synthetic_style_pred)
    assert isinstance(result, dict)
    assert len(result) > 0
    for key, val in result.items():
        assert isinstance(val, float), f"{key} is not a float"
        assert math.isfinite(val), f"{key} = {val} is not finite"


def test_internals_diagnostics_expected_keys(train_mod, synthetic_pooled, synthetic_style_dec, synthetic_style_pred):
    result = train_mod._gst_internals_diagnostics(synthetic_pooled, synthetic_style_dec, synthetic_style_pred)
    assert "gst/pooled_style_norm_mean" in result
    assert "gst/pooled_style_pairwise_cos_mean" in result
    assert "gst/style_dec_pairwise_cos_mean" in result
    assert "gst/style_pred_pairwise_cos_mean" in result


def test_internals_diagnostics_batch1_skips_pairwise(train_mod):
    pooled = torch.randn(1, EMBED_DIM)
    style_dec = torch.randn(1, STYLE_DEC_DIM)
    style_pred = torch.randn(1, STYLE_DEC_DIM)
    result = train_mod._gst_internals_diagnostics(pooled, style_dec, style_pred)
    # norm is always logged; pairwise metrics are skipped for batch=1
    assert "gst/pooled_style_norm_mean" in result
    assert "gst/pooled_style_pairwise_cos_mean" not in result
    assert "gst/style_dec_pairwise_cos_mean" not in result
    assert "gst/style_pred_pairwise_cos_mean" not in result


def test_internals_diagnostics_collapsed_batch_has_high_pairwise_cos(train_mod):
    # All items identical → pairwise cosine similarity = 1.0
    vec = torch.randn(1, EMBED_DIM).expand(B, -1).contiguous()
    dec = torch.randn(1, STYLE_DEC_DIM).expand(B, -1).contiguous()
    pred = torch.randn(1, STYLE_DEC_DIM).expand(B, -1).contiguous()
    result = train_mod._gst_internals_diagnostics(vec, dec, pred)
    assert result["gst/pooled_style_pairwise_cos_mean"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# _gst_attn_diagnostics
# ---------------------------------------------------------------------------

def test_attn_diagnostics_returns_finite_floats(train_mod, synthetic_attn_weights):
    result = train_mod._gst_attn_diagnostics(synthetic_attn_weights)
    assert isinstance(result, dict)
    for key, val in result.items():
        assert isinstance(val, float), f"{key} is not a float"
        assert math.isfinite(val), f"{key} = {val} is not finite"


def test_attn_diagnostics_expected_keys(train_mod, synthetic_attn_weights):
    result = train_mod._gst_attn_diagnostics(synthetic_attn_weights)
    assert "gst/attn_entropy_mean" in result
    assert "gst/attn_top1_index_mode_count" in result


def test_attn_diagnostics_entropy_positive(train_mod, synthetic_attn_weights):
    result = train_mod._gst_attn_diagnostics(synthetic_attn_weights)
    assert result["gst/attn_entropy_mean"] > 0.0


def test_attn_diagnostics_mode_count_in_valid_range(train_mod, synthetic_attn_weights):
    result = train_mod._gst_attn_diagnostics(synthetic_attn_weights)
    count = result["gst/attn_top1_index_mode_count"]
    assert 1.0 <= count <= float(B)


def test_attn_diagnostics_uniform_attention_high_entropy(train_mod):
    # Uniform distribution → maximum entropy = log(NUM_BASES)
    uniform = torch.full((B, T_Q, NUM_BASES), 1.0 / NUM_BASES)
    result = train_mod._gst_attn_diagnostics(uniform)
    expected_entropy = math.log(NUM_BASES)
    assert result["gst/attn_entropy_mean"] == pytest.approx(expected_entropy, rel=1e-4)


def test_attn_diagnostics_collapsed_attention_max_mode_count(train_mod):
    # All items attend exclusively to bank 0 → mode_count == B
    collapsed = torch.zeros(B, T_Q, NUM_BASES)
    collapsed[:, :, 0] = 1.0
    result = train_mod._gst_attn_diagnostics(collapsed)
    assert result["gst/attn_top1_index_mode_count"] == float(B)


# ---------------------------------------------------------------------------
# Integration: SegmentGST forward with need_weights=True feeds both helpers
# ---------------------------------------------------------------------------

def test_full_pipeline_diagnostics_finite(train_mod):
    gst_mod = sys.modules["voice_clone.segment_gst"]
    torch.manual_seed(42)
    gst = gst_mod.SegmentGST(
        num_bases=NUM_BASES,
        embed_dim=EMBED_DIM,
        frame_dim=16,
        num_heads=4,
        ref_dim=64,
        style_dec_dim=32,
        dropout=0.0,
        conv_kernel_size=3,
        conv_stride=1,
        conv_padding=1,
    )
    frames = torch.randn(B, 10, 16)
    mask = torch.ones(B, 10)
    out = gst(frames, mask, need_weights=True)

    assert out.pooled_style.shape == (B, EMBED_DIM)
    assert out.attn_weights is not None
    assert out.attn_weights.shape[0] == B
    assert out.attn_weights.shape[2] == NUM_BASES

    internals = train_mod._gst_internals_diagnostics(
        out.pooled_style.detach(), out.style_dec.detach(), out.style_pred.detach()
    )
    for key, val in internals.items():
        assert math.isfinite(val), f"{key} not finite"

    attn_diag = train_mod._gst_attn_diagnostics(out.attn_weights.detach())
    for key, val in attn_diag.items():
        assert math.isfinite(val), f"{key} not finite"
