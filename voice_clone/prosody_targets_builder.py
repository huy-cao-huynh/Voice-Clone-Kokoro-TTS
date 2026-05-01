"""Precompute per-row duration/F0 supervision for feature-cache building."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import torchaudio

from kokoro.pipeline import KPipeline

from .alignment import default_alignment_row_path
from .config import kokoro_vocab_and_context_length
from .dataset import (
    build_manifest_row_fingerprint,
    load_audio_mono,
    normalize_lang_code,
    phonemes_to_input_ids,
    text_to_phonemes,
)

PROSODY_CACHE_SCHEMA_VERSION = "phase1_mfa_v2"


def default_prosody_row_path(
    manifest_path: Path,
    row_index: int,
    *,
    prosody_cache_root: Path = Path("prosody_cache"),
) -> Path:
    return Path(prosody_cache_root) / manifest_path.stem / f"{int(row_index)}.pt"


def _estimate_total_duration_steps(
    *,
    target_num_samples_24k: int,
    token_count: int,
    duration_steps_per_second: float,
) -> int:
    seconds = float(target_num_samples_24k) / 24_000.0
    total = int(round(seconds * float(duration_steps_per_second)))
    return max(int(token_count), total)


def _allocate_uniform_durations(token_count: int, total_steps: int) -> torch.Tensor:
    total_steps = max(int(total_steps), int(token_count))
    base = total_steps // token_count
    rem = total_steps % token_count
    durations = torch.full((token_count,), float(base), dtype=torch.float32)
    if rem > 0:
        durations[:rem] += 1.0
    return durations


def _extract_f0_targets(
    target_wav_24k: torch.Tensor,
    *,
    target_length: int,
    frame_time_seconds: float,
    sample_rate: int = 24_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_length = max(int(target_length), 1)
    wav = target_wav_24k.unsqueeze(0)
    f0 = torchaudio.functional.detect_pitch_frequency(
        wav,
        sample_rate=sample_rate,
        frame_time=float(frame_time_seconds),
    ).squeeze(0)
    if f0.numel() == 0:
        f0 = torch.zeros(1, dtype=torch.float32)
    f0 = torch.nan_to_num(f0.float(), nan=0.0, posinf=0.0, neginf=0.0)
    voiced = f0 > 1.0
    if int(f0.numel()) != int(target_length):
        f0 = F.interpolate(f0.view(1, 1, -1), size=int(target_length), mode="linear", align_corners=False).view(-1)
        voiced = F.interpolate(voiced.float().view(1, 1, -1), size=int(target_length), mode="nearest").view(-1) > 0.5
    return f0, voiced


def build_prosody_targets_for_manifest(
    manifest_path: Path,
    *,
    kokoro_repo_id: str,
    prosody_cache_root: Path,
    alignments_root: Optional[Path] = None,
    manifest_root: Optional[Path] = None,
    duration_steps_per_second: float = 100.0,
    strict_single_chunk: bool = True,
    device: Optional[torch.device] = None,
    skip_existing: bool = True,
    start_index: int = 0,
) -> None:
    vocab, context_length = kokoro_vocab_and_context_length(kokoro_repo_id)
    manifest_root = manifest_root or manifest_path.parent
    pipelines: Dict[str, KPipeline] = {}

    def pipeline_for_lang(lang_code: str) -> KPipeline:
        lang = normalize_lang_code(lang_code)
        if lang not in pipelines:
            pipelines[lang] = KPipeline(lang_code=lang, repo_id=kokoro_repo_id, model=False)
        return pipelines[lang]

    device = device or torch.device("cpu")
    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if idx < int(start_index):
                continue
            row: Dict[str, Any] = json.loads(line)
            out_path = default_prosody_row_path(manifest_path, idx, prosody_cache_root=prosody_cache_root)
            if skip_existing and out_path.is_file():
                continue
            target_wav_path = Path(row["target_wav"])
            if not target_wav_path.is_absolute():
                target_wav_path = manifest_root / target_wav_path

            try:
                phonemes = row.get("phonemes")
                if phonemes is None:
                    phonemes = text_to_phonemes(
                        pipeline_for_lang(row["lang_code"]),
                        row["text"],
                        strict_single_chunk=strict_single_chunk,
                    )
                input_ids = phonemes_to_input_ids(vocab, str(phonemes), context_length=context_length)
                target_wav_24k = load_audio_mono(target_wav_path, target_sr=24_000).to(device)
                alignment_payload = None
                if alignments_root is not None:
                    alignment_path = default_alignment_row_path(manifest_path, idx, alignments_root=alignments_root)
                    if alignment_path.is_file():
                        alignment_payload = torch.load(alignment_path, map_location="cpu", weights_only=False)
                if alignment_payload is not None and bool(alignment_payload.get("prosody_enabled", False)):
                    gt_dur_frames = torch.as_tensor(alignment_payload["gt_dur_frames"], dtype=torch.long)
                    gt_dur_mask = torch.as_tensor(alignment_payload["gt_dur_mask"], dtype=torch.bool)
                    if gt_dur_frames.numel() != int(input_ids.numel()):
                        raise ValueError(
                            f"{manifest_path}:{idx}: alignment duration length {gt_dur_frames.numel()} "
                            f"!= input token count {int(input_ids.numel())}"
                        )
                    duration_targets = gt_dur_frames.to(dtype=torch.float32)
                    duration_mask = gt_dur_mask
                    prosody_enabled = bool(alignment_payload.get("prosody_enabled", False))
                    alignment_status = str(alignment_payload.get("alignment_status", "unknown"))
                    mfa_acoustic_model_id = alignment_payload.get("mfa_acoustic_model_id")
                    mfa_lexicon_id = alignment_payload.get("mfa_lexicon_id")
                    phoneme_set_version = alignment_payload.get("phoneme_set_version", "unknown")
                    target_num_samples_24k = int(alignment_payload["target_num_samples_24k"])
                    gt_total_duration_frames = int(alignment_payload["gt_total_duration_frames"])
                    gt_total_duration_samples = int(alignment_payload["gt_total_duration_samples"])
                    duration_coverage_ratio = float(alignment_payload["duration_coverage_ratio"])
                    coverage_min_ratio = float(alignment_payload["coverage_min_ratio"])
                    coverage_max_ratio = float(alignment_payload["coverage_max_ratio"])
                    coverage_accepted = bool(alignment_payload["coverage_accepted"])
                    coverage_rejection_reason = alignment_payload.get("coverage_rejection_reason")
                else:
                    total_duration_steps = _estimate_total_duration_steps(
                        target_num_samples_24k=int(target_wav_24k.numel()),
                        token_count=int(input_ids.numel()),
                        duration_steps_per_second=float(duration_steps_per_second),
                    )
                    duration_targets = _allocate_uniform_durations(int(input_ids.numel()), total_duration_steps)
                    duration_mask = torch.ones_like(duration_targets, dtype=torch.bool)
                    gt_dur_frames = torch.zeros(int(input_ids.numel()), dtype=torch.long)
                    gt_dur_mask = torch.zeros(int(input_ids.numel()), dtype=torch.bool)
                    prosody_enabled = False
                    alignment_status = "heuristic_fallback"
                    mfa_acoustic_model_id = None
                    mfa_lexicon_id = None
                    phoneme_set_version = "heuristic_uniform_v1"
                    target_num_samples_24k = int(target_wav_24k.numel())
                    gt_total_duration_frames = 0
                    gt_total_duration_samples = 0
                    duration_coverage_ratio = 0.0
                    coverage_min_ratio = 0.0
                    coverage_max_ratio = 0.0
                    coverage_accepted = False
                    coverage_rejection_reason = "heuristic_fallback"

                f0_target_length = int(duration_targets.sum().item()) * 2
                frame_time_seconds = 1.0 / (float(duration_steps_per_second) * 2.0)
                f0_targets, f0_mask = _extract_f0_targets(
                    target_wav_24k,
                    target_length=f0_target_length,
                    frame_time_seconds=frame_time_seconds,
                )
            except Exception as exc:
                text = str(row.get("text", ""))
                short_text = text[:120].replace("\n", " ")
                raise ValueError(
                    f"{manifest_path}:{idx}: failed to build prosody targets for lang_code={row.get('lang_code')!r} "
                    f"text={short_text!r} target_wav={row.get('target_wav')!r}"
                ) from exc

            payload = {
                "duration_targets": duration_targets.cpu(),
                "duration_mask": duration_mask.cpu(),
                "gt_dur_frames": gt_dur_frames.cpu(),
                "gt_dur_mask": gt_dur_mask.cpu(),
                "prosody_enabled": bool(prosody_enabled),
                "f0_targets": f0_targets.to(dtype=torch.float16).cpu(),
                "f0_mask": f0_mask.cpu(),
                "manifest_fingerprint": build_manifest_row_fingerprint(row, index=idx),
                "row_index": idx,
                "manifest_path": str(manifest_path),
                "prosody_cache_schema_version": PROSODY_CACHE_SCHEMA_VERSION,
                "duration_steps_per_second": float(duration_steps_per_second),
                "alignment_status": alignment_status,
                "mfa_acoustic_model_id": mfa_acoustic_model_id,
                "mfa_lexicon_id": mfa_lexicon_id,
                "phoneme_set_version": phoneme_set_version,
                "target_num_samples_24k": target_num_samples_24k,
                "gt_total_duration_frames": gt_total_duration_frames,
                "gt_total_duration_samples": gt_total_duration_samples,
                "duration_coverage_ratio": duration_coverage_ratio,
                "coverage_min_ratio": coverage_min_ratio,
                "coverage_max_ratio": coverage_max_ratio,
                "coverage_accepted": coverage_accepted,
                "coverage_rejection_reason": coverage_rejection_reason,
                "note": (
                    "Duration targets come from MFA-backed alignments when available, "
                    "falling back to heuristic uniform allocation for unsupported rows. "
                    "F0 targets remain placeholder compatibility fields until explicit Phase-2 prosody supervision."
                ),
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-row duration/F0 supervision caches.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, default=None)
    p.add_argument("--kokoro-repo", type=str, default="hexgrad/Kokoro-82M")
    p.add_argument("--prosody-cache-root", type=Path, default=Path("prosody_cache"))
    p.add_argument("--alignments-root", type=Path, default=None)
    p.add_argument("--duration-steps-per-second", type=float, default=100.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_prosody_targets_for_manifest(
        args.manifest,
        kokoro_repo_id=args.kokoro_repo,
        prosody_cache_root=args.prosody_cache_root,
        alignments_root=args.alignments_root,
        manifest_root=args.manifest_root,
        duration_steps_per_second=args.duration_steps_per_second,
        device=torch.device(args.device),
        skip_existing=args.skip_existing,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
