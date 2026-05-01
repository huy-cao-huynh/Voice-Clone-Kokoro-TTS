"""Tests for cache-builder prosody-cache integration."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = _ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cache_builder_loads_external_prosody_cache(monkeypatch, tmp_path):
    voice_clone_pkg = types.ModuleType("voice_clone")
    voice_clone_pkg.__path__ = [str(_ROOT / "voice_clone")]
    monkeypatch.setitem(sys.modules, "voice_clone", voice_clone_pkg)

    pipeline_mod = types.ModuleType("kokoro.pipeline")
    pipeline_mod.ALIASES = {"en-us": "a"}
    pipeline_mod.LANG_CODES = {"a": "English"}
    pipeline_mod.KPipeline = object
    monkeypatch.setitem(sys.modules, "kokoro.pipeline", pipeline_mod)

    mhubert_mod = types.ModuleType("voice_clone.mhubert_encoder")
    mhubert_mod.MHuBERTEncoder = object
    monkeypatch.setitem(sys.modules, "voice_clone.mhubert_encoder", mhubert_mod)

    wespeaker_mod = types.ModuleType("voice_clone.wespeaker_sv")
    wespeaker_mod.WeSpeakerSV = object
    monkeypatch.setitem(sys.modules, "voice_clone.wespeaker_sv", wespeaker_mod)

    cache_builder = _load_module("voice_clone.cache_builder", "voice_clone/cache_builder.py")

    prosody_path = tmp_path / "prosody.pt"
    torch.save(
        {
            "prosody_cache_schema_version": "phase1_mfa_v2",
            "duration_targets": torch.tensor([1.0, 2.0, 3.0]),
            "duration_mask": torch.tensor([1, 1, 0], dtype=torch.bool),
            "gt_dur_frames": torch.tensor([0, 4, 5], dtype=torch.long),
            "gt_dur_mask": torch.tensor([0, 1, 1], dtype=torch.bool),
            "prosody_enabled": True,
            "mfa_acoustic_model_id": "english_mfa",
            "mfa_lexicon_id": "english_mfa",
            "phoneme_set_version": "test-v1",
            "target_num_samples_24k": 5400,
            "gt_total_duration_frames": 9,
            "gt_total_duration_samples": 5400,
            "duration_coverage_ratio": 1.0,
            "coverage_min_ratio": 0.9,
            "coverage_max_ratio": 1.1,
            "coverage_accepted": True,
            "coverage_rejection_reason": None,
            "f0_targets": torch.tensor([100.0, 110.0]),
            "f0_mask": torch.tensor([1, 0], dtype=torch.bool),
        },
        prosody_path,
    )

    (
        dur,
        dur_mask,
        f0,
        f0_mask,
        gt_dur,
        prosody_enabled,
        acoustic_model,
        lexicon_id,
        phoneme_set_version,
        target_num_samples_24k,
        gt_total_duration_frames,
        gt_total_duration_samples,
        duration_coverage_ratio,
        coverage_min_ratio,
        coverage_max_ratio,
        coverage_accepted,
        coverage_rejection_reason,
    ) = cache_builder._load_row_prosody(
        {"text": "hello"},
        token_count=3,
        prosody_cache_path=prosody_path,
    )
    assert dur.tolist() == [1.0, 2.0, 3.0]
    assert dur_mask.tolist() == [True, True, False]
    assert gt_dur.tolist() == [0, 4, 5]
    assert prosody_enabled is True
    assert acoustic_model == "english_mfa"
    assert lexicon_id == "english_mfa"
    assert phoneme_set_version == "test-v1"
    assert target_num_samples_24k == 5400
    assert gt_total_duration_frames == 9
    assert gt_total_duration_samples == 5400
    assert duration_coverage_ratio == 1.0
    assert coverage_min_ratio == 0.9
    assert coverage_max_ratio == 1.1
    assert coverage_accepted is True
    assert coverage_rejection_reason is None
    assert f0.tolist() == [100.0, 110.0]
    assert f0_mask.tolist() == [True, False]
