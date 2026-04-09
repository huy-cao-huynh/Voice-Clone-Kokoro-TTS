"""Focused tests for the production data path in ``voice_clone.dataset``."""

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


class _DatasetHarness:
    def __init__(self) -> None:
        self.audio_by_path: dict[str, tuple[np.ndarray, int]] = {}
        self.pipeline_outputs: dict[str, list[str]] = {}
        self.pipeline_inits: list[tuple[str, str, bool]] = []
        self.pipeline_calls: list[tuple[str, None]] = []
        self.resampler_inits: list[tuple[int, int]] = []
        self.resampler_calls: list[tuple[int, int, tuple[int, ...]]] = []


@pytest.fixture
def dataset_mod(monkeypatch):
    harness = _DatasetHarness()

    class FakePipelineResult:
        def __init__(self, phonemes: str) -> None:
            self.phonemes = phonemes

    class FakeKPipeline:
        def __init__(self, *, lang_code: str, repo_id: str, model: bool) -> None:
            harness.pipeline_inits.append((lang_code, repo_id, model))

        def __call__(self, text: str, voice=None):
            harness.pipeline_calls.append((text, voice))
            return [FakePipelineResult(p) for p in harness.pipeline_outputs.get(text, ["ab"])]

    pipeline_mod = types.ModuleType("kokoro.pipeline")
    pipeline_mod.ALIASES = {"en-us": "a", "en-gb": "b"}
    pipeline_mod.LANG_CODES = {"a": "American English", "b": "British English", "h": "Hindi"}
    pipeline_mod.KPipeline = FakeKPipeline

    kokoro_mod = types.ModuleType("kokoro")
    kokoro_mod.pipeline = pipeline_mod

    class FakeResample:
        def __init__(self, *, orig_freq: int, new_freq: int) -> None:
            self.orig_freq = orig_freq
            self.new_freq = new_freq
            harness.resampler_inits.append((orig_freq, new_freq))

        def __call__(self, wav: torch.Tensor) -> torch.Tensor:
            harness.resampler_calls.append((self.orig_freq, self.new_freq, tuple(wav.shape)))
            new_len = int(round(wav.shape[-1] * self.new_freq / self.orig_freq))
            return torch.zeros(wav.shape[:-1] + (new_len,), dtype=wav.dtype)

    def fake_sf_read(path: str, *, always_2d: bool = False, dtype: str = "float32"):
        assert dtype == "float32"
        try:
            data, sr = harness.audio_by_path[str(Path(path))]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        return data.copy(), sr

    torchaudio_mod = types.ModuleType("torchaudio")
    torchaudio_mod.transforms = types.SimpleNamespace(Resample=FakeResample)
    torchaudio_mod.load = lambda path: (_ for _ in ()).throw(AssertionError("soundfile path expected in tests"))

    soundfile_mod = types.ModuleType("soundfile")
    soundfile_mod.read = fake_sf_read

    monkeypatch.setitem(sys.modules, "kokoro", kokoro_mod)
    monkeypatch.setitem(sys.modules, "kokoro.pipeline", pipeline_mod)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_mod)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile_mod)

    spec = importlib.util.spec_from_file_location("voice_clone.dataset", _DATASET_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "voice_clone.dataset", module)
    spec.loader.exec_module(module)

    return module, harness


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_manifest(path: Path, *rows: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _dataset_row(**overrides):
    row = {
        "ref_wav": "clips/ref.wav",
        "target_wav": "clips/target.wav",
        "text": "hello",
        "lang_code": "en-us",
        "phonemes": "ab",
        "speaker_id": "spk-1",
    }
    row.update(overrides)
    return row


def test_normalize_lang_code_uses_aliases_and_rejects_unknown(dataset_mod):
    ds, _ = dataset_mod
    assert ds.normalize_lang_code("EN-US") == "a"
    assert ds.normalize_lang_code("b") == "b"
    with pytest.raises(ValueError, match="Unknown lang_code"):
        ds.normalize_lang_code("xx-fake")


def test_phonemes_to_input_ids_filters_unknowns_but_rejects_all_oov(dataset_mod):
    ds, _ = dataset_mod
    vocab = {"a": 1, "b": 2}
    assert ds.phonemes_to_input_ids(vocab, "axb", context_length=10).tolist() == [0, 1, 2, 0]
    with pytest.raises(ValueError, match="no in-vocabulary tokens"):
        ds.phonemes_to_input_ids(vocab, "xxx", context_length=10)


def test_text_to_phonemes_rejects_empty_g2p_output(dataset_mod):
    ds, harness = dataset_mod
    harness.pipeline_outputs["empty"] = [""]
    pipe = ds.KPipeline(lang_code="a", repo_id="repo", model=False)
    with pytest.raises(ValueError, match="empty phoneme string"):
        ds.text_to_phonemes(pipe, "empty")


def test_load_audio_mono_downmixes_resamples_and_reuses_cached_resampler(dataset_mod, tmp_path):
    ds, harness = dataset_mod
    wav_path = tmp_path / "stereo.wav"
    _touch(wav_path)
    harness.audio_by_path[str(wav_path)] = (np.ones((6, 2), dtype=np.float32), 48_000)

    wav1 = ds.load_audio_mono(wav_path, target_sr=16_000)
    wav2 = ds.load_audio_mono(wav_path, target_sr=16_000)

    assert wav1.dim() == 1
    assert wav1.shape[0] == 2
    assert wav2.shape[0] == 2
    assert harness.resampler_inits == [(48_000, 16_000)]
    assert len(harness.resampler_calls) == 2


def test_dataset_init_fails_fast_for_bad_rows(dataset_mod, tmp_path):
    ds, _ = dataset_mod
    manifest = tmp_path / "bad.jsonl"

    _write_manifest(manifest, _dataset_row(text="   "))
    with pytest.raises(ValueError, match="text must be non-empty"):
        ds.VoiceCloneManifestDataset(manifest, kokoro_repo_id="repo", vocab={"a": 1}, context_length=8)

    _write_manifest(manifest, _dataset_row(lang_code="xx-fake"))
    with pytest.raises(ValueError, match="Unknown lang_code"):
        ds.VoiceCloneManifestDataset(manifest, kokoro_repo_id="repo", vocab={"a": 1}, context_length=8)

    _write_manifest(manifest, _dataset_row(ref_wav="clips/missing.wav"))
    with pytest.raises(FileNotFoundError, match="audio file not found"):
        ds.VoiceCloneManifestDataset(manifest, kokoro_repo_id="repo", vocab={"a": 1}, context_length=8)


def test_dataset_uses_provided_phonemes_and_returns_expected_fields(dataset_mod, tmp_path):
    ds, harness = dataset_mod
    ref_path = tmp_path / "clips" / "ref.wav"
    tgt_path = tmp_path / "clips" / "target.wav"
    _touch(ref_path)
    _touch(tgt_path)
    harness.audio_by_path[str(ref_path)] = (np.ones(6_000, dtype=np.float32), 16_000)
    harness.audio_by_path[str(tgt_path)] = (np.ones(7_200, dtype=np.float32), 24_000)
    manifest = _write_manifest(tmp_path / "ok.jsonl", _dataset_row())

    dataset = ds.VoiceCloneManifestDataset(
        manifest,
        kokoro_repo_id="repo",
        vocab={"a": 1, "b": 2},
        context_length=8,
        manifest_root=tmp_path,
    )
    sample = dataset[0]

    assert sample["lang_code"] == "a"
    assert sample["speaker_id"] == "spk-1"
    assert sample["input_ids"].tolist() == [0, 1, 2, 0]
    assert harness.pipeline_inits == []
    assert harness.pipeline_calls == []


def test_dataset_rejects_multi_chunk_g2p_at_init(dataset_mod, tmp_path):
    ds, harness = dataset_mod
    ref_path = tmp_path / "clips" / "ref.wav"
    tgt_path = tmp_path / "clips" / "target.wav"
    _touch(ref_path)
    _touch(tgt_path)
    manifest = _write_manifest(
        tmp_path / "g2p.jsonl",
        _dataset_row(phonemes=None, text="split me"),
    )
    harness.pipeline_outputs["split me"] = ["ab", "cd"]

    with pytest.raises(ValueError, match="produced 2 chunks"):
        ds.VoiceCloneManifestDataset(
            manifest,
            kokoro_repo_id="repo",
            vocab={"a": 1, "b": 2, "c": 3, "d": 4},
            context_length=10,
            manifest_root=tmp_path,
            preload_phonemes=True,
        )


def test_dataset_rejects_too_short_audio_and_oov_phonemes(dataset_mod, tmp_path):
    ds, harness = dataset_mod
    ref_path = tmp_path / "clips" / "ref.wav"
    tgt_path = tmp_path / "clips" / "target.wav"
    _touch(ref_path)
    _touch(tgt_path)
    manifest = _write_manifest(tmp_path / "short.jsonl", _dataset_row(phonemes="xxx"))
    harness.audio_by_path[str(ref_path)] = (np.ones(4_799, dtype=np.float32), 16_000)
    harness.audio_by_path[str(tgt_path)] = (np.ones(7_200, dtype=np.float32), 24_000)

    dataset = ds.VoiceCloneManifestDataset(
        manifest,
        kokoro_repo_id="repo",
        vocab={"a": 1, "b": 2},
        context_length=10,
        manifest_root=tmp_path,
    )
    with pytest.raises(ValueError, match="Reference audio too short"):
        dataset[0]

    harness.audio_by_path[str(ref_path)] = (np.ones(6_000, dtype=np.float32), 16_000)
    with pytest.raises(ValueError, match="no in-vocabulary tokens"):
        dataset[0]


def test_collate_voice_clone_batch_uses_real_function(dataset_mod):
    ds, _ = dataset_mod
    sample = {
        "ref_wav_16k": torch.randn(16_000),
        "target_wav_24k": torch.randn(24_000),
        "input_ids": torch.tensor([0, 1, 2, 0]),
        "text": "hello",
        "speaker_id": "spk-1",
    }

    batch = ds.collate_voice_clone_batch([sample])
    assert batch["ref_wav_16k"].shape == (1, 16_000)
    assert batch["target_wav_24k"].shape == (1, 24_000)
    assert batch["input_ids"].shape == (1, 4)
    assert batch["text"] == "hello"
    assert batch["speaker_id"] == "spk-1"

    with pytest.raises(ValueError, match="batch size 1"):
        ds.collate_voice_clone_batch([sample, sample])
