"""Tests for the cache-backed dataset path."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _ROOT / "voice_clone" / "dataset.py"


class _Harness:
    def __init__(self) -> None:
        self.audio_by_path: dict[str, tuple[np.ndarray, int]] = {}


@pytest.fixture
def dataset_mod(monkeypatch):
    harness = _Harness()

    class FakePipelineResult:
        def __init__(self, phonemes: str) -> None:
            self.phonemes = phonemes

    class FakeKPipeline:
        def __init__(self, *, lang_code: str, repo_id: str, model: bool) -> None:
            self.lang_code = lang_code

        def __call__(self, text: str, voice=None):
            return [FakePipelineResult("ab")]

    pipeline_mod = types.ModuleType("kokoro.pipeline")
    pipeline_mod.ALIASES = {"en-us": "a"}
    pipeline_mod.LANG_CODES = {"a": "English"}
    pipeline_mod.KPipeline = FakeKPipeline
    monkeypatch.setitem(sys.modules, "kokoro.pipeline", pipeline_mod)

    class FakeResample:
        def __init__(self, *, orig_freq: int, new_freq: int) -> None:
            self.orig_freq = orig_freq
            self.new_freq = new_freq

        def __call__(self, wav: torch.Tensor) -> torch.Tensor:
            new_len = int(round(wav.shape[-1] * self.new_freq / self.orig_freq))
            return torch.zeros(wav.shape[:-1] + (new_len,), dtype=wav.dtype)

    def fake_sf_read(path: str, *, always_2d: bool = False, dtype: str = "float32"):
        del always_2d, dtype
        data, sr = harness.audio_by_path[str(Path(path))]
        return data.copy(), sr

    torchaudio_mod = types.ModuleType("torchaudio")
    torchaudio_mod.transforms = types.SimpleNamespace(Resample=FakeResample)
    soundfile_mod = types.ModuleType("soundfile")
    soundfile_mod.read = fake_sf_read
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_mod)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile_mod)

    spec = importlib.util.spec_from_file_location("voice_clone.dataset", _DATASET_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "voice_clone.dataset", module)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module, harness


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_manifest(path: Path, row: dict[str, object]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return path


def _write_cache(ds_mod, manifest: Path, idx: int, payload: dict[str, object], *, cache_root: Path) -> None:
    cache_path = ds_mod.default_cache_row_path(manifest, idx, cache_root=cache_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def _row() -> dict[str, object]:
    return {
        "ref_wav": "clips/ref.wav",
        "target_wav": "clips/target.wav",
        "text": "hello",
        "lang_code": "en-us",
        "phonemes": "ab",
        "speaker_id": "spk-1",
        "duration_targets": [1.0, 2.0, 3.0, 4.0],
        "f0_targets": [100.0, 101.0, 102.0],
    }


def test_dataset_loads_cache_and_collate_pads(dataset_mod, tmp_path):
    ds, harness = dataset_mod
    row = _row()
    ref_path = tmp_path / "clips" / "ref.wav"
    tgt_path = tmp_path / "clips" / "target.wav"
    _touch(ref_path)
    _touch(tgt_path)
    harness.audio_by_path[str(ref_path)] = (np.ones(6_000, dtype=np.float32), 16_000)
    harness.audio_by_path[str(tgt_path)] = (np.ones(7_000, dtype=np.float32), 24_000)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", row)
    fingerprint = ds.build_manifest_row_fingerprint(row, index=0)
    _write_cache(
        ds,
        manifest,
        0,
        {
            "ref_hidden_states": torch.randn(5, 768),
            "ref_frame_mask": torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
            "target_wespeaker_embedding": torch.randn(256),
            "duration_targets": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "duration_mask": torch.tensor([1, 1, 1, 1], dtype=torch.bool),
            "f0_targets": torch.tensor([100.0, 101.0, 102.0]),
            "f0_mask": torch.tensor([1, 1, 0], dtype=torch.bool),
            "manifest_fingerprint": fingerprint,
        },
        cache_root=tmp_path / "cache",
    )

    dataset = ds.VoiceCloneManifestDataset(
        manifest,
        kokoro_repo_id="repo",
        vocab={"a": 1, "b": 2},
        context_length=8,
        manifest_root=tmp_path,
        feature_cache_root=tmp_path / "cache",
    )
    sample = dataset[0]
    batch = ds.collate_voice_clone_batch([sample, sample])
    assert sample["target_wespeaker_embedding"].shape == (256,)
    assert batch["ref_hidden_states"].shape == (2, 5, 768)
    assert batch["target_wespeaker_embedding"].shape == (2, 256)
    assert batch["duration_mask"].dtype is torch.bool


def test_dataset_rejects_stale_cache(dataset_mod, tmp_path):
    ds, harness = dataset_mod
    row = _row()
    ref_path = tmp_path / "clips" / "ref.wav"
    tgt_path = tmp_path / "clips" / "target.wav"
    _touch(ref_path)
    _touch(tgt_path)
    harness.audio_by_path[str(ref_path)] = (np.ones(6_000, dtype=np.float32), 16_000)
    harness.audio_by_path[str(tgt_path)] = (np.ones(7_000, dtype=np.float32), 24_000)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", row)
    _write_cache(
        ds,
        manifest,
        0,
        {
            "ref_hidden_states": torch.randn(5, 768),
            "ref_frame_mask": torch.ones(5, dtype=torch.bool),
            "target_wespeaker_embedding": torch.randn(256),
            "duration_targets": torch.ones(4),
            "duration_mask": torch.ones(4, dtype=torch.bool),
            "f0_targets": torch.ones(3),
            "f0_mask": torch.ones(3, dtype=torch.bool),
            "manifest_fingerprint": "stale",
        },
        cache_root=tmp_path / "cache",
    )
    with pytest.raises(ValueError, match="Stale feature cache"):
        ds.VoiceCloneManifestDataset(
            manifest,
            kokoro_repo_id="repo",
            vocab={"a": 1, "b": 2},
            context_length=8,
            manifest_root=tmp_path,
            feature_cache_root=tmp_path / "cache",
        )


def test_dataset_rejects_missing_cache(dataset_mod, tmp_path):
    ds, harness = dataset_mod
    row = _row()
    ref_path = tmp_path / "clips" / "ref.wav"
    tgt_path = tmp_path / "clips" / "target.wav"
    _touch(ref_path)
    _touch(tgt_path)
    harness.audio_by_path[str(ref_path)] = (np.ones(6_000, dtype=np.float32), 16_000)
    harness.audio_by_path[str(tgt_path)] = (np.ones(7_000, dtype=np.float32), 24_000)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", row)
    with pytest.raises(FileNotFoundError, match="Feature cache row not found"):
        ds.VoiceCloneManifestDataset(
            manifest,
            kokoro_repo_id="repo",
            vocab={"a": 1, "b": 2},
            context_length=8,
            manifest_root=tmp_path,
            feature_cache_root=tmp_path / "cache",
        )
