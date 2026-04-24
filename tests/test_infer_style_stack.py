"""Focused inference regressions for the SegmentGST-only style stack."""

from __future__ import annotations

import importlib.machinery
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
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    voice_clone_pkg = types.ModuleType("voice_clone")
    voice_clone_pkg.__path__ = [str(_ROOT / "voice_clone")]
    monkeypatch.setitem(sys.modules, "voice_clone", voice_clone_pkg)

    kokoro_pkg = types.ModuleType("kokoro")
    kokoro_pkg.__path__ = [str(_ROOT / "kokoro")]
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_pkg)

    torchaudio_mod = types.ModuleType("torchaudio")
    torchaudio_mod.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
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
        universal_style_vector_path: str = "voice_clone/universal_style_vector.pt"
        feature_cache_root: str = "cache"
        disable_amp_for_stft: bool = True
        gst_embed_dim: int = 1024
        loss_weights: LossWeights = field(default_factory=LossWeights)
        mel: MelLossConfig = field(default_factory=MelLossConfig)
        contrastive_temperature: float = 0.07
        validate_cache_freshness: bool = True
        min_language_speakers: int = 2
        speed: float = 1.0

    config_mod.LossWeights = LossWeights
    config_mod.MelLossConfig = MelLossConfig
    config_mod.TrainConfig = TrainConfig
    config_mod.kokoro_vocab_and_context_length = lambda repo_id: ({"a": 1, "b": 2}, 8)
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
    monkeypatch.setitem(sys.modules, "voice_clone.train_adapters", train_mod)

    mhubert_mod = types.ModuleType("voice_clone.mhubert_encoder")
    mhubert_mod.MHuBERTEncoder = object
    monkeypatch.setitem(sys.modules, "voice_clone.mhubert_encoder", mhubert_mod)

    kokoro_model_mod = types.ModuleType("kokoro.model")
    kokoro_model_mod.KModel = object
    monkeypatch.setitem(sys.modules, "kokoro.model", kokoro_model_mod)

    kokoro_pipeline_mod = types.ModuleType("kokoro.pipeline")
    kokoro_pipeline_mod.KPipeline = object
    monkeypatch.setitem(sys.modules, "kokoro.pipeline", kokoro_pipeline_mod)

    infer_mod = load_module("voice_clone.infer", _ROOT / "voice_clone" / "infer.py")
    return config_mod, infer_mod


def test_infer_waveform_has_no_adapter_style_path(monkeypatch):
    config_mod, infer_mod = _load_infer_module(monkeypatch)
    cfg = config_mod.TrainConfig()
    calls: dict[str, object] = {}

    class DummyMHuBERT:
        def eval(self):
            return self

        def __call__(self, waveforms_16k, attention_mask=None):
            calls["attention_mask"] = attention_mask
            return type("Out", (), {"hidden_states": torch.randn(1, 5, 768), "frame_mask": torch.ones(1, 5, dtype=torch.bool)})()

    class DummyGST:
        def eval(self):
            return self

        def __call__(self, frame_hidden_states, frame_mask):
            return type("Out", (), {"ref_s": torch.randn(1, 256)})(), None

    class DummyKModel:
        def eval(self):
            return self

        def forward_with_tokens(self, input_ids, ref_s, speed, return_training_outputs=False):
            calls["return_training_outputs"] = return_training_outputs
            return torch.zeros(1, 32), torch.ones(2, dtype=torch.long)

    monkeypatch.setattr(infer_mod.torch, "load", lambda *args, **kwargs: {"train_config": asdict(cfg), "segment_gst": {}})
    monkeypatch.setattr(infer_mod, "build_stack_for_inference", lambda cfg, device: (DummyKModel(), DummyGST(), DummyMHuBERT()))
    monkeypatch.setattr(infer_mod, "apply_voice_clone_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer_mod, "KPipeline", lambda *args, **kwargs: object())

    wav = infer_mod.infer_waveform(
        ckpt_path="dummy.pt",
        ref_wav_path="ref.wav",
        text="hello",
        lang_code="a",
        device=torch.device("cpu"),
    )
    assert tuple(wav.shape) == (32,)
    assert calls["attention_mask"] is None
    assert calls["return_training_outputs"] is False


def test_train_config_from_checkpoint_dict_ignores_unknown_keys(monkeypatch):
    config_mod, infer_mod = _load_infer_module(monkeypatch)
    restored = infer_mod.train_config_from_checkpoint_dict({"kokoro_repo_id": "dummy/kokoro", "unknown": 1})
    assert isinstance(restored, config_mod.TrainConfig)
