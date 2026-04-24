"""Checkpoint roundtrip tests for the new SegmentGST-only schema."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch: pytest.MonkeyPatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def harness(monkeypatch, tmp_path):
    voice_clone_pkg = types.ModuleType("voice_clone")
    voice_clone_pkg.__path__ = [str(_ROOT / "voice_clone")]
    monkeypatch.setitem(sys.modules, "voice_clone", voice_clone_pkg)
    kokoro_pkg = types.ModuleType("kokoro")
    kokoro_pkg.__path__ = [str(_ROOT / "kokoro")]
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_pkg)
    kokoro_pipeline_mod = types.ModuleType("kokoro.pipeline")
    kokoro_pipeline_mod.KPipeline = object
    monkeypatch.setitem(sys.modules, "kokoro.pipeline", kokoro_pipeline_mod)
    disc_pkg = types.ModuleType("voice_clone.discriminators")
    disc_pkg.__path__ = [str(_ROOT / "voice_clone" / "discriminators")]
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators", disc_pkg)
    torchaudio_mod = types.ModuleType("torchaudio")
    torchaudio_mod.save = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_mod)

    config_mod = types.ModuleType("voice_clone.config")

    @dataclass
    class LossWeights:
        lambda_mel: float = 20.0
        lambda_spk_contrastive: float = 1.0
        lambda_adv: float = 1.0
        lambda_fm: float = 2.0
        lambda_dur: float = 1.0
        lambda_f0: float = 1.0

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
        universal_style_vector_path: str = str(tmp_path / "u.pt")
        feature_cache_root: str = "cache"
        disable_amp_for_stft: bool = True
        gst_embed_dim: int = 1024
        loss_weights: LossWeights = field(default_factory=LossWeights)
        mel: MelLossConfig = field(default_factory=MelLossConfig)
        contrastive_temperature: float = 0.07
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
        warmup_steps: int = 0
        batch_size: int = 2
        grad_accum_steps: int = 1
        disc_start_step: int = 99999999
        speed: float = 1.0
        gst_dropout: float = 0.0
        grad_clip_norm_g: float = 5.0
        grad_clip_norm_d: float = 1.0
        lr_min_g: float = 1e-4
        lr_min_d: float = 5e-5

    config_mod.LossWeights = LossWeights
    config_mod.MelLossConfig = MelLossConfig
    config_mod.TrainConfig = TrainConfig
    config_mod.kokoro_vocab_and_context_length = lambda repo_id: ({}, 0)
    config_mod.load_kokoro_config = lambda repo_id: {"hidden_dim": 4, "n_layer": 2, "n_mels": 80, "vocab": {}}
    monkeypatch.setitem(sys.modules, "voice_clone.config", config_mod)

    dataset_mod = types.ModuleType("voice_clone.dataset")
    dataset_mod.VoiceCloneManifestDataset = object
    dataset_mod.collate_voice_clone_batch = lambda samples: samples
    dataset_mod.load_audio_mono = lambda *args, **kwargs: torch.zeros(16_000)
    dataset_mod.normalize_lang_code = lambda lang: lang
    dataset_mod.phonemes_to_input_ids = lambda *args, **kwargs: torch.tensor([1, 2])
    dataset_mod.text_to_phonemes = lambda *args, **kwargs: "ab"
    monkeypatch.setitem(sys.modules, "voice_clone.dataset", dataset_mod)

    losses_mod = types.ModuleType("voice_clone.losses")
    losses_mod.MelReconstructionLoss = nn.Identity
    losses_mod.discriminator_loss_lsgan = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.duration_loss_log_space = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.feature_matching_loss = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.generator_loss_lsgan = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.masked_l1_loss = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.speaker_contrastive_loss = lambda *args, **kwargs: torch.tensor(0.0)
    monkeypatch.setitem(sys.modules, "voice_clone.losses", losses_mod)

    mhubert_mod = types.ModuleType("voice_clone.mhubert_encoder")
    mhubert_mod.MHuBERTEncoder = lambda *args, **kwargs: types.SimpleNamespace(hidden_size=768, to=lambda self, device: self)
    monkeypatch.setitem(sys.modules, "voice_clone.mhubert_encoder", mhubert_mod)

    wespeaker_mod = types.ModuleType("voice_clone.wespeaker_sv")
    wespeaker_mod.WeSpeakerSV = types.SimpleNamespace(from_checkpoint=lambda *args, **kwargs: nn.Identity())
    monkeypatch.setitem(sys.modules, "voice_clone.wespeaker_sv", wespeaker_mod)

    disc_mod = types.ModuleType("voice_clone.discriminators.hifigan")

    class DummyDisc(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)

    disc_mod.HiFiGANMPDMSDDiscriminator = DummyDisc
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators.hifigan", disc_mod)

    kokoro_model_mod = types.ModuleType("kokoro.model")

    class DummyKModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.predictor = types.SimpleNamespace(text_encoder=types.SimpleNamespace(adapters=None))
            self.decoder = types.SimpleNamespace(decoder_adapters=None, generator=types.SimpleNamespace(adapters=None))

    kokoro_model_mod.KModel = DummyKModel
    monkeypatch.setitem(sys.modules, "kokoro.model", kokoro_model_mod)

    torch.save(torch.zeros(256), tmp_path / "u.pt")
    segment_mod = _load_module(monkeypatch, "voice_clone.segment_gst", _ROOT / "voice_clone" / "segment_gst.py")
    train_mod = _load_module(monkeypatch, "voice_clone.train_adapters", _ROOT / "voice_clone" / "train_adapters.py")
    infer_mod = _load_module(monkeypatch, "voice_clone.infer", _ROOT / "voice_clone" / "infer.py")
    return config_mod, train_mod, infer_mod, segment_mod


def test_checkpoint_roundtrip_preserves_step_and_schema(harness, tmp_path):
    config_mod, train_mod, _infer_mod, segment_mod = harness
    cfg = config_mod.TrainConfig()
    gst = segment_mod.SegmentGST()
    disc = nn.Linear(1, 1)
    kmodel = nn.Linear(1, 1)
    opt_g = torch.optim.AdamW(gst.parameters(), lr=cfg.lr_g, betas=(cfg.adam_b1, cfg.adam_b2))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg.lr_d, betas=(cfg.adam_b1, cfg.adam_b2))
    path = tmp_path / "ckpt.pt"
    train_mod.save_checkpoint(path, gst=gst, disc=disc, kmodel=kmodel, opt_g=opt_g, opt_d=opt_d, step=42, cfg=cfg)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "segment_gst" in ckpt
    assert "kokoro_lora" not in ckpt
    step = train_mod.load_checkpoint(
        path,
        gst=segment_mod.SegmentGST(),
        disc=nn.Linear(1, 1),
        kmodel=nn.Linear(1, 1),
        opt_g=torch.optim.AdamW(segment_mod.SegmentGST().parameters(), lr=cfg.lr_g, betas=(cfg.adam_b1, cfg.adam_b2)),
        opt_d=torch.optim.AdamW(nn.Linear(1, 1).parameters(), lr=cfg.lr_d, betas=(cfg.adam_b1, cfg.adam_b2)),
        device=torch.device("cpu"),
    )
    assert step == 42


def test_infer_rejects_legacy_adapter_checkpoint(harness):
    _config_mod, _train_mod, infer_mod, segment_mod = harness
    with pytest.raises(ValueError, match="Legacy adapter/LoRA checkpoint"):
        infer_mod.apply_voice_clone_checkpoint(
            {"segment_gst": segment_mod.SegmentGST().state_dict(), "kokoro_lora": {}},
            gst=segment_mod.SegmentGST(),
            kmodel=nn.Linear(1, 1),
        )


def test_train_config_from_checkpoint_dict_restores_nested_dataclasses(harness):
    config_mod, _train_mod, infer_mod, _segment_mod = harness
    cfg = config_mod.TrainConfig()
    restored = infer_mod.train_config_from_checkpoint_dict(asdict(cfg))
    assert restored.contrastive_temperature == cfg.contrastive_temperature
    assert restored.loss_weights.lambda_dur == cfg.loss_weights.lambda_dur
