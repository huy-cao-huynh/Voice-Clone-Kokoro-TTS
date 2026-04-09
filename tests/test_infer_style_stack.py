"""Focused inference regressions for the style-conditioning stack."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_infer_module(monkeypatch):
    def load_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod

    voice_clone_pkg = types.ModuleType("voice_clone")
    voice_clone_pkg.__path__ = [str(_ROOT / "voice_clone")]
    monkeypatch.setitem(sys.modules, "voice_clone", voice_clone_pkg)

    kokoro_pkg = types.ModuleType("kokoro")
    kokoro_pkg.__path__ = [str(_ROOT / "kokoro")]
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_pkg)

    torchaudio_mod = types.ModuleType("torchaudio")
    torchaudio_mod.save = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_mod)

    config_mod = types.ModuleType("voice_clone.config")

    @dataclass
    class LossWeights:
        lambda_mel: float = 1.0
        lambda_spk: float = 15.0
        lambda_adv: float = 2.0
        lambda_fm: float = 15.0

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
        wespeaker_checkpoint_path: str = "dummy/wespeaker/avg_model.pt"
        wespeaker_embedding_dim: int = 256
        wespeaker_sample_rate: int = 16_000
        universal_style_vector_path: str = "voice_clone/universal_style_vector.pt"
        disable_amp_for_stft: bool = True
        adapter_bottleneck: int = 64
        loss_weights: LossWeights = field(default_factory=LossWeights)
        mel: MelLossConfig = field(default_factory=MelLossConfig)
        speed: float = 1.0

    config_mod.LossWeights = LossWeights
    config_mod.MelLossConfig = MelLossConfig
    config_mod.TrainConfig = TrainConfig
    config_mod.kokoro_vocab_and_context_length = lambda repo_id: ({}, 0)
    monkeypatch.setitem(sys.modules, "voice_clone.config", config_mod)

    dataset_mod = types.ModuleType("voice_clone.dataset")
    dataset_mod.load_audio_mono = lambda *args, **kwargs: torch.zeros(16_000)
    dataset_mod.normalize_lang_code = lambda lang: lang
    dataset_mod.phonemes_to_input_ids = lambda *args, **kwargs: torch.tensor([1, 2])
    dataset_mod.text_to_phonemes = lambda *args, **kwargs: "ab"
    monkeypatch.setitem(sys.modules, "voice_clone.dataset", dataset_mod)

    segment_mod = types.ModuleType("voice_clone.segment_gst")
    segment_mod.SegmentGST = object
    monkeypatch.setitem(sys.modules, "voice_clone.segment_gst", segment_mod)

    train_mod = types.ModuleType("voice_clone.train_adapters")
    train_mod.build_models = lambda *args, **kwargs: None
    train_mod._speaker_frame_mask = lambda ref_out: ref_out.frame_mask
    monkeypatch.setitem(sys.modules, "voice_clone.train_adapters", train_mod)

    wespeaker_mod = types.ModuleType("voice_clone.wespeaker_sv")
    wespeaker_mod.WeSpeakerSV = object
    monkeypatch.setitem(sys.modules, "voice_clone.wespeaker_sv", wespeaker_mod)

    kokoro_model_mod = types.ModuleType("kokoro.model")
    kokoro_model_mod.KModel = object
    monkeypatch.setitem(sys.modules, "kokoro.model", kokoro_model_mod)

    kokoro_pipeline_mod = types.ModuleType("kokoro.pipeline")
    kokoro_pipeline_mod.KPipeline = object
    monkeypatch.setitem(sys.modules, "kokoro.pipeline", kokoro_pipeline_mod)

    infer_mod = load_module("voice_clone.infer", _ROOT / "voice_clone" / "infer.py")
    return config_mod, infer_mod


def test_infer_waveform_uses_actual_reference_sample_rate(monkeypatch):
    config_mod, infer_mod = _load_infer_module(monkeypatch)

    cfg = config_mod.TrainConfig(wespeaker_sample_rate=8_000)
    ref_audio = torch.randn(16_000)
    calls: dict[str, object] = {}

    class DummySV:
        def eval(self):
            return self

        def __call__(self, waveforms, *, sampling_rate, grad_through_input):
            calls["sampling_rate"] = sampling_rate
            calls["grad_through_input"] = grad_through_input
            calls["waveform_shape"] = tuple(waveforms.shape)
            return type(
                "DummySVOutput",
                (),
                {
                    "frame_features": torch.randn(1, 5, 4),
                    "frame_mask": torch.tensor([[True, True, True, False, False]]),
                    "pooled_embedding": torch.randn(1, 256),
                },
            )()

    class DummyGST:
        def eval(self):
            return self

        def __call__(self, frame_features, frame_mask):
            calls["frame_mask"] = frame_mask.clone()
            return type("DummyGSTOutput", (), {"ref_s": torch.randn(1, 256)})(), None

    class DummyKModel:
        def eval(self):
            return self

        def forward_with_tokens(self, input_ids, ref_s, speed):
            calls["speed"] = speed
            return torch.zeros(1, 32), torch.ones(1, dtype=torch.long)

    def fake_torch_load(*args, **kwargs):
        calls["weights_only"] = kwargs.get("weights_only")
        return {"train_config": asdict(cfg)}

    monkeypatch.setattr(infer_mod.torch, "load", fake_torch_load)
    monkeypatch.setattr(infer_mod, "build_stack_for_inference", lambda cfg, device: (DummyKModel(), DummyGST(), DummySV()))
    monkeypatch.setattr(infer_mod, "apply_voice_clone_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer_mod, "normalize_lang_code", lambda lang: lang)
    monkeypatch.setattr(infer_mod, "KPipeline", lambda *args, **kwargs: object())
    monkeypatch.setattr(infer_mod, "text_to_phonemes", lambda *args, **kwargs: "ab")
    monkeypatch.setattr(infer_mod, "kokoro_vocab_and_context_length", lambda *args, **kwargs: ({"a": 1, "b": 2}, 8))
    monkeypatch.setattr(infer_mod, "phonemes_to_input_ids", lambda *args, **kwargs: torch.tensor([1, 2]))
    monkeypatch.setattr(infer_mod, "load_audio_mono", lambda *args, **kwargs: ref_audio.clone())

    wav = infer_mod.infer_waveform(
        ckpt_path="dummy.pt",
        ref_wav_path="ref.wav",
        text="hello",
        lang_code="a",
        device=torch.device("cpu"),
    )

    assert tuple(wav.shape) == (32,)
    assert calls["weights_only"] is True
    assert calls["sampling_rate"] == cfg.wespeaker_sample_rate
    assert calls["grad_through_input"] is False
    assert calls["waveform_shape"] == (1, ref_audio.numel())
    assert calls["speed"] == cfg.speed
    assert torch.equal(calls["frame_mask"], torch.tensor([[True, True, True, False, False]]))


def test_apply_voice_clone_checkpoint_rejects_missing_adapter_modules(monkeypatch):
    _config_mod, infer_mod = _load_infer_module(monkeypatch)

    class DummyModule:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    gst = DummyModule()
    kmodel = types.SimpleNamespace(
        predictor=types.SimpleNamespace(text_encoder=types.SimpleNamespace(adapters=None)),
        decoder=types.SimpleNamespace(
            decoder_adapters=DummyModule(),
            generator=types.SimpleNamespace(adapters=DummyModule()),
        ),
    )
    ckpt = {
        "segment_gst": {"bank": torch.tensor([1.0])},
        "duration_adapters": {"weight": torch.tensor([2.0])},
    }

    with pytest.raises(ValueError, match="duration adapters"):
        infer_mod.apply_voice_clone_checkpoint(ckpt, gst=gst, kmodel=kmodel)
