"""Tests for MFA alignment helper logic."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_alignment_module(monkeypatch):
    voice_clone_pkg = types.ModuleType("voice_clone")
    voice_clone_pkg.__path__ = [str(_ROOT / "voice_clone")]
    monkeypatch.setitem(sys.modules, "voice_clone", voice_clone_pkg)

    config_mod = types.ModuleType("voice_clone.config")
    config_mod.kokoro_vocab_and_context_length = lambda repo_id: ({"a": 1, "b": 2, " ": 3, ".": 4}, 8)
    monkeypatch.setitem(sys.modules, "voice_clone.config", config_mod)

    dataset_mod = types.ModuleType("voice_clone.dataset")
    dataset_mod.build_manifest_row_fingerprint = lambda row, *, index: f"fp-{index}"
    dataset_mod.load_audio_mono = lambda path, *, target_sr: torch.zeros(target_sr)
    dataset_mod.normalize_lang_code = lambda lang_code: lang_code
    monkeypatch.setitem(sys.modules, "voice_clone.dataset", dataset_mod)

    torchaudio_mod = types.ModuleType("torchaudio")
    torchaudio_mod.save = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_mod)

    spec = importlib.util.spec_from_file_location(
        "voice_clone.alignment.mfa_pipeline",
        _ROOT / "voice_clone" / "alignment" / "mfa_pipeline.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "voice_clone.alignment.mfa_pipeline", module)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_textgrid_intervals_reads_phone_tier(monkeypatch, tmp_path):
    mod = _load_alignment_module(monkeypatch)
    path = tmp_path / "example.TextGrid"
    path.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    class = "IntervalTier"
    name = "phones"
    intervals: size = 2
    intervals [1]:
        xmin = 0
        xmax = 0.5
        text = "a"
    intervals [2]:
        xmin = 0.5
        xmax = 1.0
        text = "sil"
""",
        encoding="utf-8",
    )
    assert mod._parse_textgrid_intervals(path) == [(0.0, 0.5, "a"), (0.5, 1.0, "sil")]


def test_duration_projection_marks_non_acoustic_tokens(monkeypatch):
    mod = _load_alignment_module(monkeypatch)
    durations, mask, status = mod._durations_for_supported_row(
        phonemes="ab .",
        vocab={"a": 1, "b": 2, " ": 3, ".": 4},
        context_length=8,
        intervals=[(0.0, 0.5, "aa"), (0.5, 1.0, "bb")],
    )
    assert status == "aligned"
    assert durations.shape == (6,)
    assert mask.tolist() == [False, True, True, False, False, False]
    assert int(durations[1]) > 0
    assert int(durations[2]) > 0


def test_alignment_payload_rejects_short_coverage(monkeypatch, tmp_path):
    mod = _load_alignment_module(monkeypatch)
    wav_path = tmp_path / "x.wav"
    wav_path.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(mod, "load_audio_mono", lambda path, *, target_sr: torch.zeros(24_000))
    item = mod._PreparedRow(
        index=0,
        row={"text": "hello", "target_wav": str(wav_path)},
        normalized_lang="a",
        utterance_id="row_00000000",
        phonemes="ab",
        target_wav_resolved=wav_path,
        corpus_dir=tmp_path,
        aligned_dir=tmp_path,
        textgrid_path=tmp_path / "x.TextGrid",
        supported=True,
        model_info={"acoustic": "english_mfa", "dictionary": "english_mfa"},
    )
    item.textgrid_path.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    class = "IntervalTier"
    name = "phones"
    intervals: size = 1
    intervals [1]:
        xmin = 0
        xmax = 0.2
        text = "a"
""",
        encoding="utf-8",
    )
    payload = mod._build_alignment_payload(
        item,
        vocab={"a": 1, "b": 2},
        context_length=8,
        min_coverage_ratio=0.90,
        max_coverage_ratio=1.10,
    )
    assert payload["coverage_accepted"] is False
    assert payload["prosody_enabled"] is False
    assert payload["coverage_rejection_reason"] == "coverage_below_min"
