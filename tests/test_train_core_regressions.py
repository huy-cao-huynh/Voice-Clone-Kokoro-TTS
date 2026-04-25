"""Core training regressions for scheduler resume and validation guards."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

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


def _load_train_modules(monkeypatch):
    voice_clone_pkg = types.ModuleType("voice_clone")
    voice_clone_pkg.__path__ = [str(_ROOT / "voice_clone")]
    monkeypatch.setitem(sys.modules, "voice_clone", voice_clone_pkg)
    kokoro_pkg = types.ModuleType("kokoro")
    kokoro_pkg.__path__ = [str(_ROOT / "kokoro")]
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_pkg)
    disc_pkg = types.ModuleType("voice_clone.discriminators")
    disc_pkg.__path__ = [str(_ROOT / "voice_clone" / "discriminators")]
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators", disc_pkg)

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
        universal_style_vector_path: str = "voice_clone/universal_style_vector.pt"
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
    disc_mod.HiFiGANMPDMSDDiscriminator = lambda *args, **kwargs: nn.Linear(1, 1)
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators.hifigan", disc_mod)

    kokoro_model_mod = types.ModuleType("kokoro.model")
    kokoro_model_mod.KModel = lambda **kwargs: nn.Linear(1, 1)
    monkeypatch.setitem(sys.modules, "kokoro.model", kokoro_model_mod)

    _load_module(monkeypatch, "voice_clone.segment_gst", _ROOT / "voice_clone" / "segment_gst.py")
    return config_mod, _load_module(monkeypatch, "voice_clone.train_adapters", _ROOT / "voice_clone" / "train_adapters.py")


def test_checkpoint_resume_restores_scheduler_state(monkeypatch, tmp_path):
    config_mod, train_mod = _load_train_modules(monkeypatch)
    cfg = config_mod.TrainConfig()
    gst = nn.Linear(2, 2)
    disc = nn.Linear(2, 2)
    kmodel = nn.Linear(2, 2)
    adam_betas = (cfg.adam_b1, cfg.adam_b2)
    opt_g = torch.optim.AdamW(gst.parameters(), lr=cfg.lr_g, betas=adam_betas)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg.lr_d, betas=adam_betas)
    sched_g = train_mod._build_scheduler(opt_g, warmup_steps=2, total_steps=6, lr_min=cfg.lr_min_g)
    sched_d = train_mod._build_scheduler(opt_d, warmup_steps=1, total_steps=4, lr_min=cfg.lr_min_d)
    opt_g.step(); sched_g.step()
    opt_d.step(); sched_d.step()
    ckpt_path = tmp_path / "resume.pt"
    train_mod.save_checkpoint(
        ckpt_path,
        gst=gst,
        disc=disc,
        kmodel=kmodel,
        opt_g=opt_g,
        opt_d=opt_d,
        step=7,
        cfg=cfg,
        sched_g=sched_g,
        sched_d=sched_d,
        generator_updates=1,
        discriminator_updates=1,
    )
    resume_state: dict[str, object] = {}
    step = train_mod.load_checkpoint(
        ckpt_path,
        gst=nn.Linear(2, 2),
        disc=nn.Linear(2, 2),
        kmodel=nn.Linear(2, 2),
        opt_g=torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=cfg.lr_g, betas=adam_betas),
        opt_d=torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=cfg.lr_d, betas=adam_betas),
        device=torch.device("cpu"),
        sched_g=train_mod._build_scheduler(torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=cfg.lr_g, betas=adam_betas), 2, 6, cfg.lr_min_g),
        sched_d=train_mod._build_scheduler(torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=cfg.lr_d, betas=adam_betas), 1, 4, cfg.lr_min_d),
        resume_state=resume_state,
    )
    assert step == 7
    assert resume_state["scheduler_g_loaded"] is True
    assert resume_state["scheduler_d_loaded"] is True


def test_train_loop_validates_intervals(monkeypatch):
    config_mod, train_mod = _load_train_modules(monkeypatch)
    with pytest.raises(ValueError, match="grad_accum_steps"):
        train_mod.train_loop(object(), config_mod.TrainConfig(grad_accum_steps=0), torch.device("cpu"))
    with pytest.raises(ValueError, match="log_interval"):
        train_mod.train_loop(object(), config_mod.TrainConfig(log_interval=0), torch.device("cpu"))
    with pytest.raises(ValueError, match="checkpoint_interval"):
        train_mod.train_loop(object(), config_mod.TrainConfig(checkpoint_interval=0), torch.device("cpu"))


def test_train_loop_logs_fixed_validation_batch_to_wandb(monkeypatch):
    config_mod, train_mod = _load_train_modules(monkeypatch)

    class TinyDataset:
        def __len__(self):
            return 3

    class DummyRun:
        def __init__(self):
            self.calls = []

        def log(self, data, step=None):
            self.calls.append((data, step))

    class DummyAudio:
        def __init__(self, data, sample_rate, caption):
            self.data = data
            self.sample_rate = sample_rate
            self.caption = caption

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Audio=DummyAudio))

    cfg = config_mod.TrainConfig(batch_size=2, checkpoint_interval=1)
    param = nn.Parameter(torch.tensor(0.0))
    gst = nn.Linear(1, 1)
    sv_model = nn.Linear(1, 1)
    kmodel = nn.Linear(1, 1)
    disc = nn.Linear(1, 1)
    mel_loss_mod = nn.Identity()
    train_batch = {
        "target_wav_24k": torch.zeros(2, 8),
        "texts": ["train_a", "train_b"],
    }
    val_batch = {
        "target_wav_24k": torch.zeros(3, 8),
        "texts": ["val_a", "val_b", "val_c"],
    }

    monkeypatch.setattr(train_mod, "build_training_models", lambda cfg, device: (kmodel, gst, sv_model, disc, mel_loss_mod, {}))
    monkeypatch.setattr(train_mod, "generator_trainable_parameters", lambda kmodel, gst: [param])
    monkeypatch.setattr(train_mod, "_forward_batch_outputs", lambda *args, **kwargs: ([object()], torch.zeros(1, 256)))

    def fake_losses(*, batch, **kwargs):
        total = (param * 0.0) + 1.0
        prefix = "val" if len(batch["texts"]) == 3 else "train"
        metrics = {"loss_g": 1.0 if prefix == "train" else 2.0, "loss_mel": 0.5}
        pred = torch.zeros(len(batch["texts"]), 16)
        return total, metrics, pred

    monkeypatch.setattr(train_mod, "_compute_generator_losses", fake_losses)
    monkeypatch.setattr(train_mod, "create_val_dataloader", lambda dataset, *, batch_size, num_workers=0: [val_batch])
    monkeypatch.setattr(train_mod, "save_checkpoint", lambda *args, **kwargs: None)

    run = DummyRun()
    train_mod.train_loop(
        [train_batch],
        cfg,
        torch.device("cpu"),
        max_steps=1,
        ckpt_dir=Path("ckpt"),
        wandb_run=run,
        wandb_num_samples=3,
        val_dataset=TinyDataset(),
    )

    logged_keys = set()
    for data, _step in run.calls:
        logged_keys.update(data.keys())
    assert "train/loss_g" in logged_keys
    assert "val/loss_g" in logged_keys
    assert "val/audio_0" in logged_keys
