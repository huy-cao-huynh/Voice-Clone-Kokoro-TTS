from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _DummyTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.Linear(2, 2)


class _DummyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = _DummyTextEncoder()


class _DummyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.Linear(2, 2)


class _DummyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder_adapters = nn.Linear(2, 2)
        self.generator = _DummyGenerator()


class _DummyKModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = _DummyPredictor()
        self.decoder = _DummyDecoder()


class _DummyGST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


class _DummyDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


def _load_train_modules(monkeypatch):
    root = _ROOT

    def load_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod

    voice_clone_pkg = types.ModuleType("voice_clone")
    voice_clone_pkg.__path__ = [str(root / "voice_clone")]
    monkeypatch.setitem(sys.modules, "voice_clone", voice_clone_pkg)

    kokoro_pkg = types.ModuleType("kokoro")
    kokoro_pkg.__path__ = [str(root / "kokoro")]
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_pkg)

    disc_pkg = types.ModuleType("voice_clone.discriminators")
    disc_pkg.__path__ = [str(root / "voice_clone" / "discriminators")]
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators", disc_pkg)

    config_mod = load_module("voice_clone.config", root / "voice_clone" / "config.py")

    adapters_mod = types.ModuleType("voice_clone.adapters")

    class DummyRegistry:
        duration_encoder = nn.ModuleList([nn.Linear(1, 1)])
        decoder = nn.ModuleList([nn.Linear(1, 1)])
        generator = nn.ModuleList([nn.Linear(1, 1)])

        @staticmethod
        def from_dims(**kwargs):
            return DummyRegistry()

    adapters_mod.AdapterRegistry = DummyRegistry
    monkeypatch.setitem(sys.modules, "voice_clone.adapters", adapters_mod)

    dataset_mod = types.ModuleType("voice_clone.dataset")
    dataset_mod.VoiceCloneManifestDataset = object
    dataset_mod.collate_voice_clone_batch = lambda samples: samples
    monkeypatch.setitem(sys.modules, "voice_clone.dataset", dataset_mod)

    losses_mod = types.ModuleType("voice_clone.losses")

    class DummyMelLoss(nn.Module):
        def forward(self, pred_wav, target_wav):
            return None

    losses_mod.MelReconstructionLoss = DummyMelLoss
    losses_mod.discriminator_loss_lsgan = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.feature_matching_loss = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.generator_loss_lsgan = lambda *args, **kwargs: torch.tensor(0.0)
    losses_mod.speaker_input_mel_from_waveform = lambda *args, **kwargs: torch.zeros(1)
    losses_mod.speaker_cosine_loss = lambda *args, **kwargs: torch.tensor(0.0)
    monkeypatch.setitem(sys.modules, "voice_clone.losses", losses_mod)

    segment_mod = types.ModuleType("voice_clone.segment_gst")

    class DummySegmentGST(nn.Module):
        pass

    segment_mod.SegmentGST = DummySegmentGST
    monkeypatch.setitem(sys.modules, "voice_clone.segment_gst", segment_mod)

    profiling_mod = types.ModuleType("voice_clone.train_profiling")
    profiling_mod.BreakdownAggregator = object
    profiling_mod.StepStopwatch = object
    profiling_mod.build_torch_profiler = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "voice_clone.train_profiling", profiling_mod)

    wespeaker_mod = types.ModuleType("voice_clone.wespeaker_sv")

    class DummyWeSpeaker(nn.Module):
        pass

    class DummyWeSpeakerOutput:
        frame_features = None
        pooled_embedding = None

    wespeaker_mod.WeSpeakerSV = DummyWeSpeaker
    wespeaker_mod.WeSpeakerSVOutput = DummyWeSpeakerOutput
    monkeypatch.setitem(sys.modules, "voice_clone.wespeaker_sv", wespeaker_mod)

    disc_mod = types.ModuleType("voice_clone.discriminators.hifigan")

    class DummyDiscriminator(nn.Module):
        pass

    disc_mod.HiFiGANMPDMSDDiscriminator = DummyDiscriminator
    monkeypatch.setitem(sys.modules, "voice_clone.discriminators.hifigan", disc_mod)

    kokoro_model_mod = types.ModuleType("kokoro.model")

    class DummyKModel(nn.Module):
        pass

    kokoro_model_mod.KModel = DummyKModel
    monkeypatch.setitem(sys.modules, "kokoro.model", kokoro_model_mod)

    train_mod = load_module("voice_clone.train_adapters", root / "voice_clone" / "train_adapters.py")
    return config_mod, train_mod


def test_checkpoint_resume_restores_scheduler_state(monkeypatch, tmp_path):
    config_mod, train_mod = _load_train_modules(monkeypatch)

    torch.manual_seed(0)

    cfg = config_mod.TrainConfig()
    kmodel = _DummyKModel()
    gst = _DummyGST()
    disc = _DummyDisc()

    opt_g = torch.optim.AdamW(list(kmodel.parameters()) + list(gst.parameters()), lr=cfg.lr_g)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=cfg.lr_d)
    sched_g = train_mod._build_scheduler(opt_g, warmup_steps=2, total_steps=6, lr_min=cfg.lr_min_g)
    sched_d = train_mod._build_scheduler(opt_d, warmup_steps=1, total_steps=4, lr_min=cfg.lr_min_d)

    for _ in range(3):
        opt_g.step()
        sched_g.step()
    for _ in range(2):
        opt_d.step()
        sched_d.step()

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
        generator_updates=3,
        discriminator_updates=2,
    )

    kmodel2 = _DummyKModel()
    gst2 = _DummyGST()
    disc2 = _DummyDisc()
    opt_g2 = torch.optim.AdamW(list(kmodel2.parameters()) + list(gst2.parameters()), lr=cfg.lr_g)
    opt_d2 = torch.optim.AdamW(disc2.parameters(), lr=cfg.lr_d)
    sched_g2 = train_mod._build_scheduler(opt_g2, warmup_steps=2, total_steps=6, lr_min=cfg.lr_min_g)
    sched_d2 = train_mod._build_scheduler(opt_d2, warmup_steps=1, total_steps=4, lr_min=cfg.lr_min_d)

    resume_state: dict[str, object] = {}
    step = train_mod.load_checkpoint(
        ckpt_path,
        gst=gst2,
        disc=disc2,
        kmodel=kmodel2,
        opt_g=opt_g2,
        opt_d=opt_d2,
        device=torch.device("cpu"),
        sched_g=sched_g2,
        sched_d=sched_d2,
        resume_state=resume_state,
    )

    assert step == 7
    assert resume_state["generator_updates"] == 3
    assert resume_state["discriminator_updates"] == 2
    assert resume_state["scheduler_g_loaded"] is True
    assert resume_state["scheduler_d_loaded"] is True

    opt_g.step()
    sched_g.step()
    opt_d.step()
    sched_d.step()

    opt_g2.step()
    sched_g2.step()
    opt_d2.step()
    sched_d2.step()

    assert opt_g2.param_groups[0]["lr"] == pytest.approx(opt_g.param_groups[0]["lr"])
    assert opt_d2.param_groups[0]["lr"] == pytest.approx(opt_d.param_groups[0]["lr"])


def test_train_loop_validates_accum_and_logging_intervals(monkeypatch):
    config_mod, train_mod = _load_train_modules(monkeypatch)

    with pytest.raises(ValueError, match="grad_accum_steps"):
        train_mod.train_loop(object(), config_mod.TrainConfig(grad_accum_steps=0), torch.device("cpu"))

    with pytest.raises(ValueError, match="log_interval"):
        train_mod.train_loop(object(), config_mod.TrainConfig(log_interval=0), torch.device("cpu"))

    with pytest.raises(ValueError, match="checkpoint_interval"):
        train_mod.train_loop(object(), config_mod.TrainConfig(checkpoint_interval=0), torch.device("cpu"))


def test_launcher_defers_device_selection_and_preserves_pythonpath(monkeypatch):
    import scripts.train as train_script

    captured: dict[str, object] = {}

    def fake_run(cmd, env, cwd):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        captured["cwd"] = cwd

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.delenv("DEVICE", raising=False)
    monkeypatch.setenv("PYTHONPATH", "custom-path")
    monkeypatch.setattr(train_script.subprocess, "run", fake_run)
    monkeypatch.setattr(train_script.sys, "argv", ["scripts/train.py"])

    with pytest.raises(SystemExit) as excinfo:
        train_script.main()

    assert excinfo.value.code == 0
    assert "--device" not in captured["cmd"]
    assert "--manifest-root" not in captured["cmd"]
    assert "--val-manifest-root" not in captured["cmd"]

    pythonpath = captured["env"]["PYTHONPATH"]
    entries = pythonpath.split(os.pathsep)
    assert entries[0] == str(train_script.REPO_ROOT)
    assert "custom-path" in entries


def test_launcher_preserves_explicit_manifest_root_overrides(monkeypatch):
    import scripts.train as train_script

    captured: dict[str, object] = {}

    def fake_run(cmd, env, cwd):
        captured["cmd"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setenv("MANIFEST_ROOT", "data/hi")
    monkeypatch.setenv("VAL_MANIFEST_ROOT", "data/hi")
    monkeypatch.setattr(train_script.subprocess, "run", fake_run)
    monkeypatch.setattr(train_script.sys, "argv", ["scripts/train.py"])

    with pytest.raises(SystemExit) as excinfo:
        train_script.main()

    assert excinfo.value.code == 0
    assert "--manifest-root" in captured["cmd"]
    assert "--val-manifest-root" in captured["cmd"]
