"""Prepare and finalize MFA-backed duration alignments for manifest rows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..config import kokoro_vocab_and_context_length
from ..dataset import build_manifest_row_fingerprint, load_audio_mono, normalize_lang_code

SUPPORTED_MFA_LANGUAGES: dict[str, dict[str, str]] = {
    "a": {"acoustic": "english_mfa", "dictionary": "english_mfa"},
    "b": {"acoustic": "english_mfa", "dictionary": "english_mfa"},
    "e": {"acoustic": "spanish_mfa", "dictionary": "spanish_mfa"},
    "f": {"acoustic": "french_mfa", "dictionary": "french_mfa"},
    "i": {"acoustic": "italian_cv", "dictionary": "italian_cv"},
    "j": {"acoustic": "japanese_mfa", "dictionary": "japanese_mfa"},
    "z": {"acoustic": "mandarin_mfa", "dictionary": "mandarin_china_mfa"},
}
DEFAULT_MIN_COVERAGE_RATIO = 0.90
DEFAULT_MAX_COVERAGE_RATIO = 1.10
_KOKORO_DURATION_FRAME_SAMPLES_24K = 600

_SILENCE_LABELS = {
    "",
    "sp",
    "spn",
    "sil",
    "silence",
    "pau",
    "br",
    "<eps>",
    "eps",
    "noise",
    "unk",
}
_NON_ACOUSTIC_CHARS = set(" \t\r\n.,;:!?\"'`“”‘’()[]{}<>|/\\-_=+~")


def default_alignment_workspace(manifest_path: Path, *, alignments_root: Path = Path("alignments")) -> Path:
    return Path(alignments_root) / manifest_path.stem


def default_alignment_row_path(
    manifest_path: Path,
    row_index: int,
    *,
    alignments_root: Path = Path("alignments"),
) -> Path:
    return default_alignment_workspace(manifest_path, alignments_root=alignments_root) / "rows" / f"{int(row_index)}.pt"


@dataclass
class _PreparedRow:
    index: int
    row: Dict[str, Any]
    normalized_lang: str
    utterance_id: str
    phonemes: str
    target_wav_resolved: Path
    corpus_dir: Optional[Path]
    aligned_dir: Optional[Path]
    textgrid_path: Optional[Path]
    supported: bool
    model_info: Optional[dict[str, str]]


def _normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = " ".join(text.split())
    return text


def _resolved_path(path_value: str, *, manifest_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = manifest_root / path
    return path


def _write_mfa_wav(src: Path, dst: Path) -> None:
    wav = load_audio_mono(src, target_sr=16_000)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
    except ImportError:
        import wave

        pcm = wav.clamp(-1.0, 1.0).mul(32767.0).round().to(dtype=torch.int16).cpu().numpy()
        with wave.open(str(dst), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(pcm.tobytes())
        return

    sf.write(str(dst), wav.cpu().numpy(), 16_000)


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _prepare_rows(
    manifest_path: Path,
    *,
    manifest_root: Path,
    alignments_root: Path,
) -> list[_PreparedRow]:
    workspace = default_alignment_workspace(manifest_path, alignments_root=alignments_root)
    rows = _load_manifest_rows(manifest_path)
    prepared: list[_PreparedRow] = []
    for idx, row in enumerate(rows):
        lang = normalize_lang_code(str(row["lang_code"]))
        utterance_id = f"row_{idx:08d}"
        supported = lang in SUPPORTED_MFA_LANGUAGES
        model_info = SUPPORTED_MFA_LANGUAGES.get(lang)
        corpus_dir = workspace / "corpus" / lang if supported else None
        aligned_dir = workspace / "aligned" / lang if supported else None
        textgrid_path = (aligned_dir / f"{utterance_id}.TextGrid") if aligned_dir is not None else None
        prepared.append(
            _PreparedRow(
                index=idx,
                row=row,
                normalized_lang=lang,
                utterance_id=utterance_id,
                phonemes=str(row.get("phonemes") or ""),
                target_wav_resolved=_resolved_path(str(row["target_wav"]), manifest_root=manifest_root),
                corpus_dir=corpus_dir,
                aligned_dir=aligned_dir,
                textgrid_path=textgrid_path,
                supported=supported,
                model_info=model_info,
            )
        )
    return prepared


def _write_workspace(
    manifest_path: Path,
    *,
    manifest_root: Path,
    alignments_root: Path,
    skip_existing: bool,
) -> list[_PreparedRow]:
    workspace = default_alignment_workspace(manifest_path, alignments_root=alignments_root)
    prepared = _prepare_rows(manifest_path, manifest_root=manifest_root, alignments_root=alignments_root)
    workspace.mkdir(parents=True, exist_ok=True)
    for item in prepared:
        if not item.supported or item.corpus_dir is None:
            continue
        wav_path = item.corpus_dir / f"{item.utterance_id}.wav"
        lab_path = item.corpus_dir / f"{item.utterance_id}.lab"
        if not (skip_existing and wav_path.is_file() and lab_path.is_file()):
            _write_mfa_wav(item.target_wav_resolved, wav_path)
            lab_path.parent.mkdir(parents=True, exist_ok=True)
            lab_path.write_text(_normalize_transcript(str(item.row["text"])), encoding="utf-8")
    metadata = {
        "manifest_path": str(manifest_path),
        "manifest_root": str(manifest_root),
        "rows": [
            {
                "index": item.index,
                "utterance_id": item.utterance_id,
                "lang_code": item.normalized_lang,
                "supported": item.supported,
                "textgrid_path": str(item.textgrid_path) if item.textgrid_path is not None else None,
                "acoustic_model": item.model_info["acoustic"] if item.model_info is not None else None,
                "dictionary": item.model_info["dictionary"] if item.model_info is not None else None,
            }
            for item in prepared
        ],
    }
    (workspace / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    return prepared


def _conda_run(conda_env: str, cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    full_cmd = ["conda", "run", "-n", conda_env, *cmd]
    subprocess.run(full_cmd, check=True, cwd=str(cwd) if cwd is not None else None)


def _conda_capture(conda_env: str, cmd: Sequence[str]) -> str:
    full_cmd = ["conda", "run", "-n", conda_env, *cmd]
    out = subprocess.run(full_cmd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def _model_status(conda_env: str, model_type: str, model_name: str) -> tuple[bool, bool]:
    expr = (
        "from montreal_forced_aligner.models import ModelManager\n"
        f"model_type = {model_type!r}\n"
        f"model_name = {model_name!r}\n"
        "manager = ModelManager()\n"
        "local_ok = model_name in manager.local_models.get(model_type, [])\n"
        "remote_ok = model_name in manager.remote_models.get(model_type, {})\n"
        "print(f'{int(local_ok)} {int(remote_ok)}')\n"
    )
    output = _conda_capture(conda_env, ["python", "-c", expr]).strip()
    parts = output.split()
    if len(parts) != 2:
        raise RuntimeError(f"Unexpected MFA model status output for {model_type}:{model_name!r}: {output!r}")
    return parts[0] == "1", parts[1] == "1"


def _ensure_model_available(conda_env: str, model_type: str, model_name: str) -> None:
    local_ok, remote_ok = _model_status(conda_env, model_type, model_name)
    if local_ok:
        return
    if not remote_ok:
        raise RuntimeError(
            f"MFA model `{model_name}` for `{model_type}` is neither installed locally nor visible in the remote index "
            f"from env `{conda_env}`. Check `conda run -n {conda_env} mfa model list {model_type}` and the MFA model docs."
        )
    _conda_run(conda_env, ["mfa", "model", "download", model_type, model_name])


def _required_tokenizer_bootstrap_commands(languages: Sequence[str]) -> list[str]:
    cmds = ["conda install -n aligner -c conda-forge spacy"]
    langs = set(languages)
    if {"a", "b", "e", "f", "i"} & langs:
        cmds.extend(
            [
                "conda run -n aligner python -m spacy download en_core_web_sm",
                "conda run -n aligner python -m spacy download es_core_news_sm",
                "conda run -n aligner python -m spacy download fr_core_news_sm",
                "conda run -n aligner python -m spacy download it_core_news_sm",
            ]
        )
    if "j" in langs:
        cmds.append("conda install -n aligner -c conda-forge sudachipy sudachidict-core")
    if "z" in langs:
        cmds.append("conda run -n aligner pip install spacy-pkuseg dragonmapper hanziconv")
    return cmds


def _preflight_tokenizers(prepared: Sequence[_PreparedRow], *, conda_env: str) -> None:
    langs = sorted({item.normalized_lang for item in prepared if item.supported})
    if not langs:
        return
    expr = (
        "from montreal_forced_aligner.tokenization.spacy import check_language_tokenizer_availability\n"
        "from montreal_forced_aligner.data import Language\n"
        f"langs = {langs!r}\n"
        "mapping = {\n"
        "    'a': Language.english,\n"
        "    'b': Language.english,\n"
        "    'e': Language.spanish,\n"
        "    'f': Language.french,\n"
        "    'i': Language.italian,\n"
        "    'j': Language.japanese,\n"
        "    'z': Language.chinese,\n"
        "}\n"
        "errors = []\n"
        "for code in langs:\n"
        "    label = mapping[code]\n"
        "    try:\n"
        "        check_language_tokenizer_availability(label)\n"
        "    except Exception as exc:\n"
        "        errors.append(f'{code}:{label}:{type(exc).__name__}:{exc}')\n"
        "print('\\n'.join(errors))\n"
    )
    output = _conda_capture(conda_env, ["python", "-c", expr])
    if not output:
        return
    hints = "\n".join(f"  {cmd}" for cmd in _required_tokenizer_bootstrap_commands(langs))
    raise RuntimeError(
        "The aligner environment is missing tokenizer dependencies required by MFA.\n"
        f"Tokenizer preflight errors:\n{output}\n\n"
        "Install the missing support in `aligner` and rerun:\n"
        f"{hints}"
    )


def _run_mfa_alignment(
    prepared: Sequence[_PreparedRow],
    *,
    conda_env: str,
    jobs: int,
    skip_existing: bool,
    clean: bool,
) -> None:
    _preflight_tokenizers(prepared, conda_env=conda_env)
    by_lang: dict[str, list[_PreparedRow]] = {}
    for item in prepared:
        if item.supported:
            by_lang.setdefault(item.normalized_lang, []).append(item)
    for lang, items in sorted(by_lang.items()):
        model_info = items[0].model_info
        assert model_info is not None
        corpus_dir = items[0].corpus_dir
        aligned_dir = items[0].aligned_dir
        assert corpus_dir is not None and aligned_dir is not None
        if skip_existing and aligned_dir.is_dir():
            expected = [aligned_dir / f"{item.utterance_id}.TextGrid" for item in items]
            if expected and all(path.is_file() for path in expected):
                continue
        _ensure_model_available(conda_env, "acoustic", model_info["acoustic"])
        _ensure_model_available(conda_env, "dictionary", model_info["dictionary"])
        aligned_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "mfa",
            "align",
            str(corpus_dir),
            model_info["dictionary"],
            model_info["acoustic"],
            str(aligned_dir),
            "--num_jobs",
            str(max(int(jobs), 1)),
            "--single_speaker",
        ]
        if clean:
            cmd.append("--clean")
        _conda_run(conda_env, cmd)


def _parse_quoted(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.replace('""', '"')


def _parse_textgrid_intervals(path: Path) -> list[tuple[float, float, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    intervals: list[tuple[float, float, str]] = []
    tier_name: Optional[str] = None
    in_interval = False
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    text: Optional[str] = None
    capture = False
    phone_tiers = {"phones", "phone", "segments"}
    for raw in lines:
        line = raw.strip()
        if line.startswith("name = "):
            tier_name = _parse_quoted(line.split("=", 1)[1])
            capture = tier_name.lower() in phone_tiers
            continue
        if not capture:
            continue
        if line.startswith("intervals ["):
            in_interval = True
            xmin = None
            xmax = None
            text = None
            continue
        if not in_interval:
            continue
        if line.startswith("xmin = "):
            xmin = float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("xmax = "):
            xmax = float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("text = "):
            text = _parse_quoted(line.split("=", 1)[1])
            if xmin is not None and xmax is not None:
                intervals.append((xmin, xmax, text or ""))
            in_interval = False
    return intervals


def _is_non_acoustic_token(ch: str) -> bool:
    if ch in _NON_ACOUSTIC_CHARS:
        return True
    category = unicodedata.category(ch)
    return category.startswith("P") or category in {"Zs", "Cc"}


def _partition_evenly(total_items: int, total_buckets: int) -> list[tuple[int, int]]:
    if total_buckets <= 0:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for bucket_idx in range(total_buckets):
        start = int(round(bucket_idx * total_items / total_buckets))
        end = int(round((bucket_idx + 1) * total_items / total_buckets))
        if end <= start:
            end = min(total_items, start + 1)
        spans.append((start, min(end, total_items)))
        cursor = end
    if spans and spans[-1][1] < total_items:
        spans[-1] = (spans[-1][0], total_items)
    return spans


def _frames_from_interval(start: float, end: float) -> int:
    return max(int(round((float(end) - float(start)) * 40.0)), 1)


def _durations_for_supported_row(
    *,
    phonemes: str,
    vocab: Dict[str, int],
    context_length: int,
    intervals: Sequence[tuple[float, float, str]],
) -> tuple[torch.LongTensor, torch.BoolTensor, str]:
    token_chars = [ch for ch in phonemes if vocab.get(ch) is not None]
    if len(token_chars) + 2 > int(context_length):
        raise ValueError(f"Phoneme sequence too long for model context: {len(token_chars) + 2} > {context_length}")
    if not token_chars:
        raise ValueError("No in-vocabulary phoneme characters available for alignment.")

    durations = torch.zeros(len(token_chars) + 2, dtype=torch.long)
    mask = torch.zeros(len(token_chars) + 2, dtype=torch.bool)
    acoustic_positions = [i + 1 for i, ch in enumerate(token_chars) if not _is_non_acoustic_token(ch)]
    non_acoustic_positions = [i + 1 for i, ch in enumerate(token_chars) if _is_non_acoustic_token(ch)]
    if not acoustic_positions:
        return durations, mask, "fallback:no_acoustic_intervals"
    if not intervals:
        return durations, mask, "fallback:no_intervals"

    speech_intervals = [(s, e, lab) for (s, e, lab) in intervals if str(lab).strip().lower() not in _SILENCE_LABELS]
    silence_intervals = [(s, e, lab) for (s, e, lab) in intervals if str(lab).strip().lower() in _SILENCE_LABELS]
    if not speech_intervals:
        return durations, mask, "fallback:no_speech_intervals"

    def _allocate_frames(positions: list[int], frame_total: int) -> None:
        if not positions or frame_total <= 0:
            return
        base = frame_total // len(positions)
        rem = frame_total % len(positions)
        for idx, pos in enumerate(positions):
            durations[pos] += base + (1 if idx < rem else 0)
            mask[pos] = True

    # Pass 1: preserve active speech timing on acoustic tokens.
    if len(speech_intervals) >= len(acoustic_positions):
        spans = _partition_evenly(len(speech_intervals), len(acoustic_positions))
        for pos, (start_idx, end_idx) in zip(acoustic_positions, spans):
            frame_total = sum(_frames_from_interval(*speech_intervals[j][:2]) for j in range(start_idx, end_idx))
            durations[pos] = max(frame_total, 1)
            mask[pos] = True
    else:
        spans = _partition_evenly(len(acoustic_positions), len(speech_intervals))
        for speech_idx, (start_pos, end_pos) in enumerate(spans):
            start, end, _label = speech_intervals[speech_idx]
            frame_total = _frames_from_interval(start, end)
            char_positions = acoustic_positions[start_pos:end_pos]
            if not char_positions:
                continue
            _allocate_frames(char_positions, max(frame_total, len(char_positions)))

    # Pass 2: assign silence time to explicit non-acoustic tokens first.
    fallback_silence_frames = 0
    if silence_intervals:
        if non_acoustic_positions:
            if len(silence_intervals) >= len(non_acoustic_positions):
                spans = _partition_evenly(len(silence_intervals), len(non_acoustic_positions))
                for pos, (start_idx, end_idx) in zip(non_acoustic_positions, spans):
                    frame_total = sum(_frames_from_interval(*silence_intervals[j][:2]) for j in range(start_idx, end_idx))
                    _allocate_frames([pos], frame_total)
            else:
                spans = _partition_evenly(len(non_acoustic_positions), len(silence_intervals))
                for silence_idx, (start_pos, end_pos) in enumerate(spans):
                    start, end, _label = silence_intervals[silence_idx]
                    frame_total = _frames_from_interval(start, end)
                    char_positions = non_acoustic_positions[start_pos:end_pos]
                    if not char_positions:
                        fallback_silence_frames += frame_total
                        continue
                    _allocate_frames(char_positions, frame_total)
        else:
            fallback_silence_frames = sum(_frames_from_interval(start, end) for start, end, _label in silence_intervals)

    # If silence cannot be represented explicitly, absorb it into neighboring acoustic tokens
    # so the total frame budget still matches the TextGrid time domain.
    if fallback_silence_frames > 0:
        _allocate_frames(acoustic_positions, fallback_silence_frames)

    total_interval_frames = sum(_frames_from_interval(start, end) for start, end, _label in intervals)
    projected_frames = int(durations.sum().item())
    if total_interval_frames > 0:
        projected_ratio = float(projected_frames) / float(total_interval_frames)
        if projected_ratio < 0.95 or projected_ratio > 1.05:
            speech_frames = sum(_frames_from_interval(start, end) for start, end, _label in speech_intervals)
            silence_frames = sum(_frames_from_interval(start, end) for start, end, _label in silence_intervals)
            non_acoustic_frames = int(durations[non_acoustic_positions].sum().item()) if non_acoustic_positions else 0
            print(
                "[mfa_pipeline] duration_projection_mismatch "
                f"tokens={len(token_chars)} acoustic_tokens={len(acoustic_positions)} "
                f"non_acoustic_tokens={len(non_acoustic_positions)} speech_frames={speech_frames} "
                f"silence_frames={silence_frames} projected_frames={projected_frames} "
                f"projected_ratio={projected_ratio:.3f} fallback_silence_frames={fallback_silence_frames} "
                f"non_acoustic_assigned_frames={non_acoustic_frames}"
            )

    return durations, mask, "aligned"


def _build_alignment_payload(
    item: _PreparedRow,
    *,
    vocab: Dict[str, int],
    context_length: int,
    min_coverage_ratio: float,
    max_coverage_ratio: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "row_index": item.index,
        "manifest_fingerprint": build_manifest_row_fingerprint(item.row, index=item.index),
        "phoneme_set_version": "mfa-char-projection-v1",
        "mfa_acoustic_model_id": item.model_info["acoustic"] if item.model_info is not None else None,
        "mfa_lexicon_id": item.model_info["dictionary"] if item.model_info is not None else None,
        "lang_code": item.normalized_lang,
    }
    if not item.supported or item.textgrid_path is None:
        token_chars = [ch for ch in item.phonemes if vocab.get(ch) is not None]
        zeros = torch.zeros(len(token_chars) + 2, dtype=torch.long)
        mask = torch.zeros(len(token_chars) + 2, dtype=torch.bool)
        target_num_samples_24k = int(load_audio_mono(item.target_wav_resolved, target_sr=24_000).numel())
        payload.update(
            {
                "gt_dur_frames": zeros,
                "gt_dur_mask": mask,
                "prosody_enabled": False,
                "alignment_status": "unsupported_language",
                "target_num_samples_24k": target_num_samples_24k,
                "gt_total_duration_frames": 0,
                "gt_total_duration_samples": 0,
                "duration_coverage_ratio": 0.0,
                "coverage_min_ratio": float(min_coverage_ratio),
                "coverage_max_ratio": float(max_coverage_ratio),
                "coverage_accepted": False,
                "coverage_rejection_reason": "unsupported_language",
            }
        )
        return payload
    if not item.textgrid_path.is_file():
        token_chars = [ch for ch in item.phonemes if vocab.get(ch) is not None]
        zeros = torch.zeros(len(token_chars) + 2, dtype=torch.long)
        mask = torch.zeros(len(token_chars) + 2, dtype=torch.bool)
        target_num_samples_24k = int(load_audio_mono(item.target_wav_resolved, target_sr=24_000).numel())
        payload.update(
            {
                "gt_dur_frames": zeros,
                "gt_dur_mask": mask,
                "prosody_enabled": False,
                "alignment_status": "missing_textgrid",
                "target_num_samples_24k": target_num_samples_24k,
                "gt_total_duration_frames": 0,
                "gt_total_duration_samples": 0,
                "duration_coverage_ratio": 0.0,
                "coverage_min_ratio": float(min_coverage_ratio),
                "coverage_max_ratio": float(max_coverage_ratio),
                "coverage_accepted": False,
                "coverage_rejection_reason": "missing_textgrid",
            }
        )
        return payload
    intervals = _parse_textgrid_intervals(item.textgrid_path)
    durations, mask, status = _durations_for_supported_row(
        phonemes=item.phonemes,
        vocab=vocab,
        context_length=context_length,
        intervals=intervals,
    )
    target_num_samples_24k = int(load_audio_mono(item.target_wav_resolved, target_sr=24_000).numel())
    gt_total_duration_frames = int(durations.sum().item())
    gt_total_duration_samples = int(gt_total_duration_frames * _KOKORO_DURATION_FRAME_SAMPLES_24K)
    duration_coverage_ratio = (
        float(gt_total_duration_samples) / float(target_num_samples_24k) if target_num_samples_24k > 0 else 0.0
    )
    coverage_accepted = (
        status == "aligned" and float(min_coverage_ratio) <= duration_coverage_ratio <= float(max_coverage_ratio)
    )
    if status != "aligned":
        coverage_rejection_reason = status
    elif duration_coverage_ratio < float(min_coverage_ratio):
        coverage_rejection_reason = "coverage_below_min"
    elif duration_coverage_ratio > float(max_coverage_ratio):
        coverage_rejection_reason = "coverage_above_max"
    else:
        coverage_rejection_reason = None
    payload.update(
        {
            "gt_dur_frames": durations,
            "gt_dur_mask": mask,
            "prosody_enabled": coverage_accepted,
            "alignment_status": status if coverage_accepted else f"rejected:{coverage_rejection_reason}",
            "target_num_samples_24k": target_num_samples_24k,
            "gt_total_duration_frames": gt_total_duration_frames,
            "gt_total_duration_samples": gt_total_duration_samples,
            "duration_coverage_ratio": duration_coverage_ratio,
            "coverage_min_ratio": float(min_coverage_ratio),
            "coverage_max_ratio": float(max_coverage_ratio),
            "coverage_accepted": coverage_accepted,
            "coverage_rejection_reason": coverage_rejection_reason,
        }
    )
    return payload


def build_alignments_for_manifest(
    manifest_path: Path,
    *,
    manifest_root: Optional[Path] = None,
    alignments_root: Path = Path("alignments"),
    kokoro_repo_id: str = "hexgrad/Kokoro-82M",
    conda_env: str = "aligner",
    jobs: int = 1,
    skip_existing: bool = True,
    clean: bool = False,
    min_coverage_ratio: float = DEFAULT_MIN_COVERAGE_RATIO,
    max_coverage_ratio: float = DEFAULT_MAX_COVERAGE_RATIO,
) -> None:
    manifest_root = manifest_root or manifest_path.parent
    prepared = _write_workspace(
        manifest_path,
        manifest_root=manifest_root,
        alignments_root=alignments_root,
        skip_existing=skip_existing,
    )
    _run_mfa_alignment(prepared, conda_env=conda_env, jobs=jobs, skip_existing=skip_existing, clean=clean)
    vocab, context_length = kokoro_vocab_and_context_length(kokoro_repo_id)
    stats: dict[str, dict[str, int]] = {}
    for item in prepared:
        out_path = default_alignment_row_path(manifest_path, item.index, alignments_root=alignments_root)
        if skip_existing and out_path.is_file():
            continue
        payload = _build_alignment_payload(
            item,
            vocab=vocab,
            context_length=context_length,
            min_coverage_ratio=min_coverage_ratio,
            max_coverage_ratio=max_coverage_ratio,
        )
        lang_stats = stats.setdefault(item.normalized_lang, {"rows": 0, "accepted": 0, "rejected": 0})
        lang_stats["rows"] += 1
        if bool(payload.get("coverage_accepted", False)):
            lang_stats["accepted"] += 1
        else:
            lang_stats["rejected"] += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
    if stats:
        print("[mfa_pipeline] coverage summary by lang_code")
        for lang, lang_stats in sorted(stats.items()):
            print(
                f"[mfa_pipeline] lang={lang} rows={lang_stats['rows']} "
                f"accepted={lang_stats['accepted']} rejected={lang_stats['rejected']}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MFA alignments and per-row duration payloads for a manifest.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, default=None)
    p.add_argument("--alignments-root", type=Path, default=Path("alignments"))
    p.add_argument("--kokoro-repo", type=str, default="hexgrad/Kokoro-82M")
    p.add_argument("--conda-env", type=str, default="aligner")
    p.add_argument("--jobs", type=int, default=max(os.cpu_count() or 1, 1))
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--min-coverage-ratio", type=float, default=DEFAULT_MIN_COVERAGE_RATIO)
    p.add_argument("--max-coverage-ratio", type=float, default=DEFAULT_MAX_COVERAGE_RATIO)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_alignments_for_manifest(
        args.manifest,
        manifest_root=args.manifest_root,
        alignments_root=args.alignments_root,
        kokoro_repo_id=args.kokoro_repo,
        conda_env=args.conda_env,
        jobs=args.jobs,
        skip_existing=args.skip_existing,
        clean=args.clean,
        min_coverage_ratio=args.min_coverage_ratio,
        max_coverage_ratio=args.max_coverage_ratio,
    )


if __name__ == "__main__":
    main()
