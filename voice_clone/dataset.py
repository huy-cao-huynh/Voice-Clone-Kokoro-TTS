"""Multilingual voice-clone dataset with offline feature-cache support."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from kokoro.pipeline import ALIASES, KPipeline, LANG_CODES


def normalize_lang_code(lang_code: str) -> str:
    c = lang_code.lower().strip()
    c = ALIASES.get(c, c)
    if c not in LANG_CODES:
        raise ValueError(f"Unknown lang_code {lang_code!r}; expected one of {sorted(LANG_CODES)}")
    return c


def phonemes_to_input_ids(
    vocab: Dict[str, int],
    phonemes: str,
    *,
    context_length: int,
    bos_id: int = 0,
    eos_id: int = 0,
) -> torch.LongTensor:
    ids = [vocab.get(ch) for ch in phonemes]
    ids = [i for i in ids if i is not None]
    if not ids:
        raise ValueError("Phoneme sequence produced no in-vocabulary tokens after filtering unknown graphemes.")
    if len(ids) + 2 > context_length:
        raise ValueError(f"Phoneme sequence too long for model context: {len(ids) + 2} > {context_length}")
    return torch.tensor([bos_id, *ids, eos_id], dtype=torch.long)


def text_to_phonemes(
    pipeline: KPipeline,
    text: str,
    *,
    strict_single_chunk: bool = True,
    max_phoneme_chars: int = 510,
) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Empty text for phonemization.")

    results = list(pipeline(text, voice=None))
    if not results:
        raise ValueError("G2P produced no segments (empty pipeline output).")
    if len(results) > 1:
        msg = f"G2P produced {len(results)} chunks for one row."
        if strict_single_chunk:
            raise ValueError(msg)
        warnings.warn(msg + " Concatenating phoneme chunks because strict_single_chunk=False.", UserWarning)

    phonemes = "".join(r.phonemes for r in results)
    if not phonemes.strip():
        raise ValueError("G2P produced an empty phoneme string.")
    if len(phonemes) > max_phoneme_chars:
        raise ValueError(f"Phoneme string length {len(phonemes)} exceeds max_phoneme_chars={max_phoneme_chars}.")
    return phonemes


@lru_cache(maxsize=16)
def _resampler(orig_sr: int, target_sr: int):
    return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)


def load_audio_mono(path: Union[str, Path], *, target_sr: int) -> torch.Tensor:
    path = Path(path)
    try:
        import soundfile as sf
    except ImportError:
        wav, sr = torchaudio.load(str(path))
    else:
        data, sr = sf.read(str(path), always_2d=False, dtype="float32")
        t = torch.from_numpy(np.ascontiguousarray(data))
        wav = t.unsqueeze(0) if t.ndim == 1 else t.T
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        wav = _resampler(sr, target_sr)(wav.unsqueeze(0)).squeeze(0)
    if not torch.isfinite(wav).all():
        raise ValueError(f"Non-finite audio samples in {path}")
    return wav


def build_manifest_row_fingerprint(row: Dict[str, Any], *, index: int) -> str:
    stable = {
        "index": int(index),
        "ref_wav": row["ref_wav"],
        "target_wav": row["target_wav"],
        "text": row["text"],
        "lang_code": normalize_lang_code(row["lang_code"]),
        "phonemes": row.get("phonemes"),
        "speaker_id": row.get("speaker_id"),
    }
    payload = json.dumps(stable, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def default_cache_row_path(
    manifest_path: Union[str, Path],
    row_index: int,
    *,
    cache_root: Union[str, Path] = "cache",
) -> Path:
    manifest = Path(manifest_path)
    return Path(cache_root) / manifest.stem / f"{int(row_index)}.pt"


def _to_bool_mask(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.bool) if x.dtype is not torch.bool else x


def load_cache_row(cache_path: Union[str, Path], *, expected_fingerprint: Optional[str] = None) -> Dict[str, Any]:
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache row not found: {path}")
    row = torch.load(path, map_location="cpu", weights_only=False)
    if expected_fingerprint is not None and row.get("manifest_fingerprint") != expected_fingerprint:
        raise ValueError(f"Stale feature cache at {path}")
    required = {
        "ref_hidden_states",
        "ref_frame_mask",
        "target_wespeaker_embedding",
        "duration_targets",
        "duration_mask",
        "f0_targets",
        "f0_mask",
        "manifest_fingerprint",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"Cache row {path} missing required keys: {missing}")
    return row


class VoiceCloneManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Union[str, Path],
        *,
        kokoro_repo_id: str,
        vocab: Dict[str, int],
        context_length: int,
        manifest_root: Optional[Union[str, Path]] = None,
        feature_cache_root: Union[str, Path] = "cache",
        strict_single_chunk: bool = True,
        preload_phonemes: bool = True,
        validate_cache_freshness: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.manifest_root = Path(manifest_root) if manifest_root else self.manifest_path.parent
        self.kokoro_repo_id = kokoro_repo_id
        self.vocab = vocab
        self.context_length = context_length
        self.feature_cache_root = Path(feature_cache_root)
        self.strict_single_chunk = strict_single_chunk
        self.validate_cache_freshness = validate_cache_freshness

        self.rows: List[Dict[str, Any]] = []
        self.cache_metadata: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._validate_row(row, line_no)
                idx = len(self.rows)
                fingerprint = build_manifest_row_fingerprint(row, index=idx)
                cache_path = default_cache_row_path(self.manifest_path, idx, cache_root=self.feature_cache_root)
                cached = load_cache_row(
                    cache_path,
                    expected_fingerprint=fingerprint if self.validate_cache_freshness else None,
                )
                self.rows.append(row)
                self.cache_metadata.append(
                    {
                        "path": cache_path,
                        "fingerprint": fingerprint,
                        "ref_frames": tuple(cached["ref_hidden_states"].shape),
                        "target_embedding_shape": tuple(cached["target_wespeaker_embedding"].shape),
                    }
                )

        self._pipelines: Dict[str, KPipeline] = {}
        self._phoneme_cache: Optional[List[str]] = None
        if preload_phonemes:
            self._phoneme_cache = [self._phonemes_for_row(i) for i in range(len(self.rows))]

    def _validate_row(self, row: Dict[str, Any], line_no: int) -> None:
        for key in ("ref_wav", "target_wav", "text", "lang_code"):
            if key not in row:
                raise ValueError(f"{self.manifest_path}:{line_no}: missing key {key!r}")
        normalize_lang_code(row["lang_code"])
        for key in ("ref_wav", "target_wav"):
            resolved = self._resolve_path(row[key])
            if not resolved.is_file():
                raise FileNotFoundError(f"{self.manifest_path}:{line_no}: audio file not found: {resolved}")

    def _resolve_path(self, p: Union[str, Path]) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = self.manifest_root / path
        return path

    def _pipeline_for_lang(self, lang_code: str) -> KPipeline:
        lang = normalize_lang_code(str(lang_code))
        if lang not in self._pipelines:
            self._pipelines[lang] = KPipeline(lang_code=lang, repo_id=self.kokoro_repo_id, model=False)
        return self._pipelines[lang]

    def _phonemes_for_row(self, index: int) -> str:
        row = self.rows[index]
        if row.get("phonemes") is not None:
            return str(row["phonemes"])
        return text_to_phonemes(self._pipeline_for_lang(row["lang_code"]), row["text"], strict_single_chunk=self.strict_single_chunk)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        ref_wav = load_audio_mono(self._resolve_path(row["ref_wav"]), target_sr=16_000)
        target_wav = load_audio_mono(self._resolve_path(row["target_wav"]), target_sr=24_000)
        if ref_wav.numel() < 4_800:
            raise ValueError("Reference audio too short")
        if target_wav.numel() < 4_800:
            raise ValueError("Target audio too short")

        phonemes = self._phoneme_cache[index] if self._phoneme_cache is not None else self._phonemes_for_row(index)
        cache = load_cache_row(
            self.cache_metadata[index]["path"],
            expected_fingerprint=self.cache_metadata[index]["fingerprint"] if self.validate_cache_freshness else None,
        )
        sample: Dict[str, Any] = {
            "ref_wav_16k": ref_wav,
            "target_wav_24k": target_wav,
            "input_ids": phonemes_to_input_ids(self.vocab, phonemes, context_length=self.context_length),
            "lang_code": normalize_lang_code(str(row["lang_code"])),
            "text": row["text"],
            "row_index": index,
            "ref_hidden_states": cache["ref_hidden_states"].detach().float(),
            "ref_frame_mask": _to_bool_mask(cache["ref_frame_mask"]),
            "target_wespeaker_embedding": cache["target_wespeaker_embedding"].detach().float(),
            "duration_targets": cache["duration_targets"].detach().float(),
            "duration_mask": _to_bool_mask(cache["duration_mask"]),
            "f0_targets": cache["f0_targets"].detach().float(),
            "f0_mask": _to_bool_mask(cache["f0_mask"]),
        }
        if row.get("speaker_id") is not None:
            sample["speaker_id"] = str(row["speaker_id"])
        return sample


def collate_voice_clone_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.nn.utils.rnn.pad_sequence([s["input_ids"] for s in samples], batch_first=True, padding_value=0)
    input_ids_lengths = torch.tensor([s["input_ids"].numel() for s in samples], dtype=torch.long)
    ref_wav_16k = torch.nn.utils.rnn.pad_sequence([s["ref_wav_16k"] for s in samples], batch_first=True, padding_value=0.0)
    ref_lengths = torch.tensor([s["ref_wav_16k"].numel() for s in samples], dtype=torch.long)
    target_wav_24k = torch.nn.utils.rnn.pad_sequence([s["target_wav_24k"] for s in samples], batch_first=True, padding_value=0.0)
    target_lengths = torch.tensor([s["target_wav_24k"].numel() for s in samples], dtype=torch.long)

    ref_hidden_states = torch.nn.utils.rnn.pad_sequence(
        [s["ref_hidden_states"] for s in samples], batch_first=True, padding_value=0.0
    )
    ref_frame_mask = torch.nn.utils.rnn.pad_sequence(
        [s["ref_frame_mask"].to(dtype=torch.bool) for s in samples], batch_first=True, padding_value=False
    )
    duration_targets = torch.nn.utils.rnn.pad_sequence(
        [s["duration_targets"] for s in samples], batch_first=True, padding_value=0.0
    )
    duration_mask = torch.nn.utils.rnn.pad_sequence(
        [s["duration_mask"].to(dtype=torch.bool) for s in samples], batch_first=True, padding_value=False
    )
    f0_targets = torch.nn.utils.rnn.pad_sequence([s["f0_targets"] for s in samples], batch_first=True, padding_value=0.0)
    f0_mask = torch.nn.utils.rnn.pad_sequence(
        [s["f0_mask"].to(dtype=torch.bool) for s in samples], batch_first=True, padding_value=False
    )
    target_wespeaker_embedding = torch.stack([s["target_wespeaker_embedding"] for s in samples], dim=0)

    batch: Dict[str, Any] = {
        "ref_wav_16k": ref_wav_16k,
        "target_wav_24k": target_wav_24k,
        "input_ids": input_ids,
        "input_ids_lengths": input_ids_lengths,
        "ref_lengths": ref_lengths,
        "target_lengths": target_lengths,
        "ref_hidden_states": ref_hidden_states,
        "ref_frame_mask": ref_frame_mask,
        "target_wespeaker_embedding": target_wespeaker_embedding,
        "duration_targets": duration_targets,
        "duration_mask": duration_mask,
        "f0_targets": f0_targets,
        "f0_mask": f0_mask,
        "texts": [s.get("text", "") for s in samples],
        "row_indices": torch.tensor([int(s["row_index"]) for s in samples], dtype=torch.long),
    }
    speaker_ids = [s.get("speaker_id") for s in samples]
    if any(s is not None for s in speaker_ids):
        batch["speaker_ids"] = speaker_ids
        if len(samples) == 1:
            batch["speaker_id"] = speaker_ids[0]
    if len(samples) == 1:
        batch["text"] = samples[0].get("text", "")
    else:
        batch["text"] = batch["texts"][0] if batch["texts"] else ""
    return batch
