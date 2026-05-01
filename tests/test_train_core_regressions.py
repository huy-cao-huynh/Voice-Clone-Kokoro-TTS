"""Core training regressions for scheduler resume and validation guards."""

from __future__ import annotations

import csv
import importlib.util
import os
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
        style_decoder_only_steps: int = 0
        gst_dropout: float = 0.0
        gst_conv_kernel_size: int = 5
        gst_conv_stride: int = 2
        gst_conv_padding: int = 2
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


def _load_script_module(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


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


def test_train_launcher_only_forwards_max_steps_when_explicitly_set(monkeypatch):
    train_script = _load_script_module(monkeypatch, "scripts.train", _ROOT / "scripts" / "train.py")
    calls: list[dict[str, object]] = []

    def fake_run(cmd, env, cwd):
        calls.append({"cmd": cmd, "env": env, "cwd": cwd})
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_script.subprocess, "run", fake_run)
    monkeypatch.setattr(train_script.sys, "argv", ["train.py"])

    base_env = {
        "MANIFEST": "manifests/custom_train.jsonl",
        "VAL_MANIFEST": "",
        "PYTHONPATH": "",
    }

    with mock.patch.dict(os.environ, base_env, clear=True):
        with pytest.raises(SystemExit) as excinfo:
            train_script.main()
    assert excinfo.value.code == 0
    assert "--max-steps" not in calls[-1]["cmd"]

    with mock.patch.dict(os.environ, {**base_env, "MAX_STEPS": "75"}, clear=True):
        with pytest.raises(SystemExit) as excinfo:
            train_script.main()
    assert excinfo.value.code == 0
    assert "--max-steps" in calls[-1]["cmd"]
    max_steps_index = calls[-1]["cmd"].index("--max-steps")
    assert calls[-1]["cmd"][max_steps_index + 1] == "75"


def test_train_loop_runs_multiple_epochs_without_explicit_max_steps(monkeypatch):
    config_mod, train_mod = _load_train_modules(monkeypatch)
    cfg = config_mod.TrainConfig(batch_size=2)
    param = nn.Parameter(torch.tensor(0.0))

    class DummyGST(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)
            self.register_buffer("universal_style_vector", torch.zeros(256), persistent=True)

    gst = DummyGST()
    sv_model = nn.Linear(1, 1)
    kmodel = nn.Linear(1, 1)
    disc = nn.Linear(1, 1)
    mel_loss_mod = nn.Identity()
    train_batch = {
        "target_wav_24k": torch.zeros(2, 8),
        "texts": ["train_a", "train_b"],
    }

    monkeypatch.setattr(train_mod, "build_training_models", lambda cfg, device: (kmodel, gst, sv_model, disc, mel_loss_mod, {}))
    monkeypatch.setattr(train_mod, "generator_trainable_parameters", lambda kmodel, gst: [param])

    class DummyOutput:
        def __init__(self):
            self.audio = torch.zeros(1, 8)
            self.duration_logits = torch.ones(1, 4)
            self.rounded_durations = torch.ones(1, 4, dtype=torch.long)
            self.f0_pred = torch.zeros(1, 2)
            self.n_pred = torch.zeros(1, 2)

    monkeypatch.setattr(
        train_mod,
        "_forward_batch_outputs",
        lambda *args, **kwargs: ([DummyOutput() for _ in range(2)], torch.zeros(2, 256)),
    )

    def fake_losses(**kwargs):
        total = (param * 0.0) + 1.0
        metrics = {"loss_g": 1.0, "loss_mel": 0.5, "loss_spk_contrastive": 0.25}
        pred = torch.zeros(2, 16)
        return total, metrics, pred

    monkeypatch.setattr(train_mod, "_compute_generator_losses", fake_losses)
    seen_steps: list[int] = []
    train_mod.train_loop(
        [train_batch],
        cfg,
        torch.device("cpu"),
        epochs=2,
        report_callback=lambda step, loss: seen_steps.append(step),
    )
    assert seen_steps == [1, 2]


def test_train_loop_logs_fixed_validation_batch_to_wandb(monkeypatch, tmp_path):
    config_mod, train_mod = _load_train_modules(monkeypatch)

    class TinyDataset:
        def __len__(self):
            return 3

    class DummyRun:
        def __init__(self, run_dir: Path):
            self.calls = []
            self.dir = str(run_dir)

        def log(self, data, step=None):
            self.calls.append((data, step))

    class DummyAudio:
        def __init__(self, data, sample_rate, caption=None):
            self.data = data
            self.sample_rate = sample_rate
            self.caption = caption

    class DummyTable:
        def __init__(self, columns):
            self.columns = columns
            self.rows = []

        def add_data(self, *row):
            self.rows.append(row)

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Audio=DummyAudio, Table=DummyTable))

    cfg = config_mod.TrainConfig(batch_size=2, checkpoint_interval=1)
    param = nn.Parameter(torch.tensor(0.0))
    class DummyGST(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)
            self.register_buffer("universal_style_vector", torch.zeros(256), persistent=True)

    gst = DummyGST()
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
        "target_lengths": torch.tensor([8, 8, 8]),
        "texts": ["val_a", "val_b", "val_c"],
        "speaker_ids": ["s1", "s2", "s3"],
        "row_indices": torch.tensor([11, 12, 13]),
    }

    monkeypatch.setattr(train_mod, "build_training_models", lambda cfg, device: (kmodel, gst, sv_model, disc, mel_loss_mod, {}))
    monkeypatch.setattr(train_mod, "generator_trainable_parameters", lambda kmodel, gst: [param])

    class DummyOutput:
        def __init__(self, audio_len, frame_len):
            self.audio = torch.zeros(1, audio_len)
            self.duration_logits = torch.ones(1, 4)
            self.rounded_durations = torch.full((1, 4), frame_len // 4, dtype=torch.long)
            self.f0_pred = torch.zeros(1, frame_len)
            self.n_pred = torch.zeros(1, frame_len)

    def fake_forward_batch_outputs(*args, force_target_total_frames=False, batch=None, **kwargs):
        del kwargs
        if batch is None:
            batch = args[2]
        count = len(batch["texts"])
        audio_len = 8 if force_target_total_frames else 12
        frame_len = 2 if force_target_total_frames else 3
        outputs = [DummyOutput(audio_len=audio_len, frame_len=frame_len) for _ in range(count)]
        return outputs, torch.zeros(count, 256)

    monkeypatch.setattr(train_mod, "_forward_batch_outputs", fake_forward_batch_outputs)

    def fake_losses(*, batch, **kwargs):
        total = (param * 0.0) + 1.0
        prefix = "val" if len(batch["texts"]) == 3 else "train"
        metrics = {
            "loss_g": 1.0 if prefix == "train" else 2.0,
            "loss_mel": 0.5,
            "loss_spk_contrastive": 0.25 if prefix == "train" else 0.75,
        }
        pred = torch.zeros(len(batch["texts"]), 16)
        return total, metrics, pred

    monkeypatch.setattr(train_mod, "_compute_generator_losses", fake_losses)
    monkeypatch.setattr(train_mod, "create_val_dataloader", lambda dataset, *, batch_size, num_workers=0: [val_batch])
    monkeypatch.setattr(train_mod, "save_checkpoint", lambda *args, **kwargs: None)

    run = DummyRun(tmp_path / "wandb" / "run-001")
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
    assert "train/loss_spk_contrastive" in logged_keys
    assert "train/grad_norm_g" in logged_keys
    assert "val_free/loss_g" in logged_keys
    assert "val_tf/loss_g" in logged_keys
    assert "val/audio_table" in logged_keys
    assert "val/text_0" not in logged_keys
    assert "val/audio_gt_0" not in logged_keys
    assert "val_free/audio_pred_0" not in logged_keys
    assert "val_tf/audio_pred_0" not in logged_keys
    assert "val_free/len_ratio" in logged_keys
    assert "val_tf/len_ratio" in logged_keys
    assert "val_gap/loss_mel" in logged_keys
    assert "val_gap/loss_total" in logged_keys
    assert "val_gap/len_ratio" in logged_keys
    assert "train/lr_d" not in logged_keys
    assert "train/style_decoder_only" not in logged_keys

    table = next(data["val/audio_table"] for data, _step in run.calls if "val/audio_table" in data)
    assert table.columns == [
        "step",
        "row_index",
        "speaker_id",
        "text",
        "coverage_ratio",
        "gt_audio",
        "pred_audio_free",
        "pred_audio_tf",
        "gt_samples_full",
        "gt_samples_tf",
        "pred_samples_free",
        "pred_samples_tf",
    ]
    assert len(table.rows) == 3

    expected_csvs = {
        "train_loss_mel.csv": [(1, 0.5)],
        "train_loss_spk_contrastive.csv": [(1, 0.25)],
        "val_free_loss_spk_contrastive.csv": [(1, 0.75)],
        "val_tf_loss_spk_contrastive.csv": [(1, 0.75)],
    }
    for filename, expected_rows in expected_csvs.items():
        csv_path = Path(run.dir) / filename
        assert csv_path.is_file()
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [(int(row["step"]), float(row["value"])) for row in rows] == expected_rows
