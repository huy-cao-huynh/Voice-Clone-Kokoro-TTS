"""Multilingual voice-clone dataset: manifest rows with ``lang_code`` and Kokoro G2P via ``KPipeline(model=False)``."""

from __future__ import annotations

from functools import lru_cache
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
    """Match :class:`KPipeline` language normalization (aliases + lower case)."""
    c = lang_code.lower().strip()
    c = ALIASES.get(c, c)
    if c not in LANG_CODES:
        raise ValueError(f"Unknown lang_code {lang_code!r}; expected one of {sorted(LANG_CODES)} (or aliases like en-us).")
    return c


def phonemes_to_input_ids(
    vocab: Dict[str, int],
    phonemes: str,
    *,
    context_length: int,
    bos_id: int = 0,
    eos_id: int = 0,
) -> torch.LongTensor:
    """Map a phoneme string to Kokoro ``input_ids`` with BOS/EOS, same filtering as :meth:`KModel.forward`."""
    ids = [vocab.get(ch) for ch in phonemes]
    ids = [i for i in ids if i is not None]
    if not ids:
        raise ValueError(
            "Phoneme sequence produced no in-vocabulary tokens after filtering unknown graphemes. "
            "Check the manifest row language, phonemes, and Kokoro repo/vocab compatibility."
        )
    if len(ids) + 2 > context_length:
        raise ValueError(
            f"Phoneme sequence too long for model context: {len(ids) + 2} > {context_length} "
            f"(after dropping unknown graphemes). Shorten text or split the manifest row."
        )
    return torch.tensor([bos_id, *ids, eos_id], dtype=torch.long)


def text_to_phonemes(
    pipeline: KPipeline,
    text: str,
    *,
    strict_single_chunk: bool = True,
    max_phoneme_chars: int = 510,
) -> str:
    """
    Run a quiet :class:`KPipeline` (``model=False``) on one training segment.

    If the pipeline emits multiple chunks (long English or long non-English), by default we raise so
    manifest rows stay aligned with a single ``target_wav``. Set ``strict_single_chunk=False`` to
    concatenate chunk phoneme strings (use only when you know that matches your supervision).
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty text for phonemization.")

    results = list(pipeline(text, voice=None))
    if not results:
        raise ValueError("G2P produced no segments (empty pipeline output).")

    if len(results) > 1:
        msg = (
            f"G2P produced {len(results)} chunks for one row; use shorter lines or split ref/target "
            f"so one manifest row maps to one acoustic segment."
        )
        if strict_single_chunk:
            raise ValueError(msg)
        warnings.warn(msg + " Concatenating phoneme chunks because strict_single_chunk=False.", UserWarning)

    phonemes = "".join(r.phonemes for r in results)
    if not phonemes.strip():
        raise ValueError("G2P produced an empty phoneme string.")
    if len(phonemes) > max_phoneme_chars:
        raise ValueError(
            f"Phoneme string length {len(phonemes)} exceeds max_phoneme_chars={max_phoneme_chars}."
        )
    return phonemes


@lru_cache(maxsize=16)
def _resampler(orig_sr: int, target_sr: int):
    return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)


def load_audio_mono(path: Union[str, Path], *, target_sr: int) -> torch.Tensor:
    """Load a file as ``(time,)`` float32 mono at ``target_sr``."""
    path = Path(path)
    try:
        import soundfile as sf
    except ImportError:
        wav, sr = torchaudio.load(str(path))
    else:
        data, sr = sf.read(str(path), always_2d=False, dtype="float32")
        t = torch.from_numpy(np.ascontiguousarray(data))
        if t.ndim == 1:
            wav = t.unsqueeze(0)
        else:
            wav = t.T
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        wav = _resampler(sr, target_sr)(wav.unsqueeze(0)).squeeze(0)
    if not torch.isfinite(wav).all():
        raise ValueError(f"Non-finite audio samples (NaN/Inf) in {path}")
    return wav


class VoiceCloneManifestDataset(Dataset):
    """
    JSONL manifest: one JSON object per line with audio paths, text, and ``lang_code``.

    Required keys per row: ``ref_wav``, ``target_wav``, ``text``, ``lang_code``.
    Optional: ``phonemes`` (skip G2P), ``speaker_id`` (string, returned for logging).

    Paths are resolved relative to ``manifest_root`` when not absolute.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        *,
        kokoro_repo_id: str,
        vocab: Dict[str, int],
        context_length: int,
        manifest_root: Optional[Union[str, Path]] = None,
        strict_single_chunk: bool = True,
        preload_phonemes: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.manifest_root = Path(manifest_root) if manifest_root else self.manifest_path.parent
        self.kokoro_repo_id = kokoro_repo_id
        self.vocab = vocab
        self.context_length = context_length
        self.strict_single_chunk = strict_single_chunk

        self.rows: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{self.manifest_path}:{line_no}: invalid JSON") from e
                self._validate_row(row, line_no)
                self.rows.append(row)

        self._pipelines: Dict[str, KPipeline] = {}
        self._phoneme_cache: Optional[List[str]] = None

        if preload_phonemes:
            self._phoneme_cache = [self._phonemes_for_row(i) for i in range(len(self.rows))]

    def _validate_row(self, row: Dict[str, Any], line_no: int) -> None:
        for key in ("ref_wav", "target_wav", "text", "lang_code"):
            if key not in row:
                raise ValueError(f"{self.manifest_path}:{line_no}: missing key {key!r}")
        for key in ("ref_wav", "target_wav"):
            value = row[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{self.manifest_path}:{line_no}: {key} must be a non-empty string path")
        if not isinstance(row["text"], str):
            raise ValueError(f"{self.manifest_path}:{line_no}: text must be a string")
        if not row["text"].strip():
            raise ValueError(f"{self.manifest_path}:{line_no}: text must be non-empty after stripping whitespace")
        if not isinstance(row["lang_code"], str):
            raise ValueError(f"{self.manifest_path}:{line_no}: lang_code must be a string")
        normalize_lang_code(row["lang_code"])
        if "phonemes" in row and row["phonemes"] is not None:
            if not isinstance(row["phonemes"], str):
                raise ValueError(f"{self.manifest_path}:{line_no}: phonemes must be a string when provided")
            if not row["phonemes"].strip():
                raise ValueError(f"{self.manifest_path}:{line_no}: phonemes must be non-empty when provided")
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
            self._pipelines[lang] = KPipeline(
                lang_code=lang,
                repo_id=self.kokoro_repo_id,
                model=False,
            )
        return self._pipelines[lang]

    def _phonemes_for_row(self, index: int) -> str:
        row = self.rows[index]
        if "phonemes" in row and row["phonemes"] is not None:
            return str(row["phonemes"])
        pipe = self._pipeline_for_lang(row["lang_code"])
        return text_to_phonemes(pipe, row["text"], strict_single_chunk=self.strict_single_chunk)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        ref_path = self._resolve_path(row["ref_wav"])
        tgt_path = self._resolve_path(row["target_wav"])

        ref_wav = load_audio_mono(ref_path, target_sr=16_000)
        target_wav = load_audio_mono(tgt_path, target_sr=24_000)

        MIN_REF_SAMPLES_16K = 4800  # ~0.3s at 16 kHz
        MIN_TGT_SAMPLES_24K = 4800  # ~0.2s at 24 kHz
        if ref_wav.shape[0] < MIN_REF_SAMPLES_16K:
            raise ValueError(
                f"Reference audio too short ({ref_wav.shape[0]} samples at 16 kHz, "
                f"need >= {MIN_REF_SAMPLES_16K}): {ref_path}"
            )
        if target_wav.shape[0] < MIN_TGT_SAMPLES_24K:
            raise ValueError(
                f"Target audio too short ({target_wav.shape[0]} samples at 24 kHz, "
                f"need >= {MIN_TGT_SAMPLES_24K}): {tgt_path}"
            )

        if self._phoneme_cache is not None:
            phonemes = self._phoneme_cache[index]
        else:
            phonemes = self._phonemes_for_row(index)

        lang = normalize_lang_code(str(row["lang_code"]))
        input_ids = phonemes_to_input_ids(self.vocab, phonemes, context_length=self.context_length)

        out: Dict[str, Any] = {
            "ref_wav_16k": ref_wav,
            "target_wav_24k": target_wav,
            "input_ids": input_ids,
            "lang_code": lang,
            "text": row["text"],
        }
        if "speaker_id" in row and row["speaker_id"] is not None:
            out["speaker_id"] = str(row["speaker_id"])
        return out


def collate_voice_clone_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch size 1 collate for variable-length ``input_ids`` and waveforms (same as training today)."""
    if len(samples) != 1:
        raise ValueError(
            "collate_voice_clone_batch currently supports batch size 1 (Kokoro `forward_with_tokens`); "
            f"got {len(samples)}"
        )
    s = samples[0]
    batch: Dict[str, Any] = {
        "ref_wav_16k": s["ref_wav_16k"].unsqueeze(0),
        "target_wav_24k": s["target_wav_24k"].unsqueeze(0),
        "input_ids": s["input_ids"].unsqueeze(0),
    }
    if "speaker_id" in s:
        batch["speaker_id"] = s["speaker_id"]
    if "text" in s:
        batch["text"] = s["text"]
    return batch
