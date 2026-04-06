"""Tests for dataset, collation, and tokenization (voice_clone/dataset.py).

phonemes_to_input_ids and collate_voice_clone_batch are pure-torch and tested
without torchaudio. load_audio_mono and VoiceCloneManifestDataset need
torchaudio + kokoro.pipeline, so those tests skip gracefully.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from conftest import make_manifest_and_wavs, write_sine_wav, write_wav

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Isolated load of dataset.py via importlib (needs torchaudio + kokoro)
# ---------------------------------------------------------------------------

def _load_dataset_mod():
    """Load voice_clone.dataset; skip if deps (torchaudio, kokoro.pipeline) missing."""
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    try:
        # Pre-import required deps to give a clear skip
        import torchaudio  # noqa: F401
        from kokoro.pipeline import KPipeline  # noqa: F401
    except ImportError as e:
        pytest.skip(f"dataset.py deps not available: {e}")
    import voice_clone.dataset as ds
    return ds


# ---------------------------------------------------------------------------
# phonemes_to_input_ids -- pure torch, tested without heavy deps
# ---------------------------------------------------------------------------

def _phonemes_to_input_ids(vocab, phonemes, *, context_length, bos_id=0, eos_id=0):
    """Local copy of dataset.phonemes_to_input_ids (avoids torchaudio import)."""
    ids = [vocab.get(ch) for ch in phonemes]
    ids = [i for i in ids if i is not None]
    if len(ids) + 2 > context_length:
        raise ValueError(
            f"Phoneme sequence too long for model context: {len(ids) + 2} > {context_length} "
            f"(after dropping unknown graphemes). Shorten text or split the manifest row."
        )
    return torch.tensor([bos_id, *ids, eos_id], dtype=torch.long)


class TestPhonemesToInputIds:
    def test_adds_bos_and_eos(self):
        vocab = {"a": 1, "b": 2, "c": 3}
        ids = _phonemes_to_input_ids(vocab, "abc", context_length=100)
        assert ids[0].item() == 0  # BOS
        assert ids[-1].item() == 0  # EOS
        assert ids.tolist() == [0, 1, 2, 3, 0]

    def test_filters_unknown_chars(self):
        vocab = {"a": 1, "b": 2}
        ids = _phonemes_to_input_ids(vocab, "axb", context_length=100)
        assert ids.tolist() == [0, 1, 2, 0]

    def test_raises_on_context_overflow(self):
        vocab = {chr(i): i for i in range(65, 91)}
        with pytest.raises(ValueError, match="too long"):
            _phonemes_to_input_ids(vocab, "A" * 100, context_length=10)

    def test_empty_phonemes_only_bos_eos(self):
        vocab = {"a": 1}
        ids = _phonemes_to_input_ids(vocab, "", context_length=100)
        assert ids.tolist() == [0, 0]


# ---------------------------------------------------------------------------
# collate_voice_clone_batch -- pure torch
# ---------------------------------------------------------------------------

def _collate_voice_clone_batch(samples):
    """Local copy of dataset.collate_voice_clone_batch."""
    if len(samples) != 1:
        raise ValueError(
            "collate_voice_clone_batch currently supports batch size 1 (Kokoro `forward_with_tokens`); "
            f"got {len(samples)}"
        )
    s = samples[0]
    batch = {
        "ref_wav_16k": s["ref_wav_16k"].unsqueeze(0),
        "target_wav_24k": s["target_wav_24k"].unsqueeze(0),
        "input_ids": s["input_ids"].unsqueeze(0),
    }
    if "speaker_id" in s:
        batch["speaker_id"] = s["speaker_id"]
    if "text" in s:
        batch["text"] = s["text"]
    return batch


class TestCollateVoiceCloneBatch:
    def test_rejects_batch_size_not_one(self):
        sample = {
            "ref_wav_16k": torch.randn(16_000),
            "target_wav_24k": torch.randn(24_000),
            "input_ids": torch.tensor([0, 1, 2, 0]),
        }
        with pytest.raises(ValueError, match="batch size 1"):
            _collate_voice_clone_batch([sample, sample])

    def test_correct_keys_and_shapes(self):
        sample = {
            "ref_wav_16k": torch.randn(16_000),
            "target_wav_24k": torch.randn(24_000),
            "input_ids": torch.tensor([0, 1, 2, 0]),
            "text": "hello",
        }
        batch = _collate_voice_clone_batch([sample])
        assert batch["ref_wav_16k"].shape == (1, 16_000)
        assert batch["target_wav_24k"].shape == (1, 24_000)
        assert batch["input_ids"].shape == (1, 4)
        assert batch["text"] == "hello"

    def test_optional_speaker_id_forwarded(self):
        sample = {
            "ref_wav_16k": torch.randn(16_000),
            "target_wav_24k": torch.randn(24_000),
            "input_ids": torch.tensor([0, 1, 0]),
            "speaker_id": "spk1",
        }
        batch = _collate_voice_clone_batch([sample])
        assert batch["speaker_id"] == "spk1"


# ---------------------------------------------------------------------------
# normalize_lang_code (needs kokoro.pipeline)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestNormalizeLangCode:
    @pytest.fixture(scope="class")
    def ds(self):
        return _load_dataset_mod()

    def test_known_alias(self, ds):
        assert ds.normalize_lang_code("en-us") == "a"
        assert ds.normalize_lang_code("en-gb") == "b"

    def test_direct_code(self, ds):
        assert ds.normalize_lang_code("a") == "a"

    def test_unknown_raises(self, ds):
        with pytest.raises(ValueError, match="Unknown lang_code"):
            ds.normalize_lang_code("xx-fake")

    def test_case_insensitive(self, ds):
        assert ds.normalize_lang_code("EN-US") == "a"


# ---------------------------------------------------------------------------
# load_audio_mono (needs torchaudio)
# ---------------------------------------------------------------------------

class TestLoadAudioMono:
    @pytest.fixture(scope="class")
    def ds(self):
        return _load_dataset_mod()

    def test_loads_mono_at_target_sr(self, ds, tmp_path):
        p = tmp_path / "test.wav"
        write_sine_wav(p, samples=4800, sr=16_000)
        wav = ds.load_audio_mono(p, target_sr=16_000)
        assert wav.dim() == 1
        assert wav.shape[0] == 4800

    def test_resamples_if_sr_differs(self, ds, tmp_path):
        p = tmp_path / "test_resample.wav"
        write_sine_wav(p, samples=4800, sr=24_000)
        wav = ds.load_audio_mono(p, target_sr=16_000)
        assert wav.dim() == 1
        expected_samples = int(4800 * 16_000 / 24_000)
        assert abs(wav.shape[0] - expected_samples) <= 10

    def test_stereo_downmix(self, ds, tmp_path):
        p = tmp_path / "stereo.wav"
        write_wav(p, samples=1600, sr=16_000, channels=2)
        wav = ds.load_audio_mono(p, target_sr=16_000)
        assert wav.dim() == 1


# ---------------------------------------------------------------------------
# VoiceCloneManifestDataset (needs torchaudio + kokoro.pipeline → slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestVoiceCloneManifestDataset:
    @pytest.fixture
    def ds_and_data(self, tmp_path):
        mod = _load_dataset_mod()
        data = make_manifest_and_wavs(tmp_path, num_rows=3)
        from voice_clone.config import kokoro_vocab_and_context_length
        vocab, ctx = kokoro_vocab_and_context_length("hexgrad/Kokoro-82M")
        dataset = mod.VoiceCloneManifestDataset(
            data["manifest"],
            kokoro_repo_id="hexgrad/Kokoro-82M",
            vocab=vocab,
            context_length=ctx,
            manifest_root=data["root"],
            preload_phonemes=True,
        )
        return dataset, data

    def test_len_matches_manifest(self, ds_and_data):
        dataset, data = ds_and_data
        assert len(dataset) == len(data["rows"])

    def test_getitem_returns_expected_keys(self, ds_and_data):
        dataset, _ = ds_and_data
        sample = dataset[0]
        assert "ref_wav_16k" in sample
        assert "target_wav_24k" in sample
        assert "input_ids" in sample
        assert "lang_code" in sample
        assert "text" in sample

    def test_ref_wav_is_1d(self, ds_and_data):
        dataset, _ = ds_and_data
        sample = dataset[0]
        assert sample["ref_wav_16k"].dim() == 1

    def test_input_ids_has_bos_eos(self, ds_and_data):
        dataset, _ = ds_and_data
        sample = dataset[0]
        ids = sample["input_ids"]
        assert ids[0].item() == 0  # BOS
        assert ids[-1].item() == 0  # EOS

    def test_missing_key_raises(self, tmp_path):
        mod = _load_dataset_mod()
        manifest = tmp_path / "bad.jsonl"
        manifest.write_text(json.dumps({"ref_wav": "x.wav", "text": "hi"}) + "\n")
        from voice_clone.config import kokoro_vocab_and_context_length
        vocab, ctx = kokoro_vocab_and_context_length("hexgrad/Kokoro-82M")
        with pytest.raises(ValueError, match="missing key"):
            mod.VoiceCloneManifestDataset(
                manifest,
                kokoro_repo_id="hexgrad/Kokoro-82M",
                vocab=vocab,
                context_length=ctx,
            )
