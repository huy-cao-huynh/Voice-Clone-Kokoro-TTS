"""Offline feature-cache builder for voice-clone manifests."""

from __future__ import annotations

import argparse
import contextlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional

import torch

from .config import kokoro_vocab_and_context_length
from .dataset import (
    VoiceCloneManifestDataset,
    build_manifest_row_fingerprint,
    default_cache_row_path,
    load_cache_row,
    load_audio_mono,
    normalize_lang_code,
    phonemes_to_input_ids,
    text_to_phonemes,
)
from .mhubert_encoder import MHuBERTEncoder
from .wespeaker_sv import WeSpeakerSV
from kokoro.pipeline import KPipeline


def _tensor_1d(values: Any, *, name: str) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.dim() != 1:
        raise ValueError(f"{name} must be 1-D")
    return t


def default_prosody_row_path(
    manifest_path: Path,
    row_index: int,
    *,
    prosody_cache_root: Path = Path("prosody_cache"),
) -> Path:
    return Path(prosody_cache_root) / manifest_path.stem / f"{int(row_index)}.pt"


def _load_row_prosody(
    row: Dict[str, Any],
    *,
    token_count: int,
    prosody_cache_path: Optional[Path] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if row.get("duration_targets") is not None and row.get("f0_targets") is not None:
        payload = {
            "duration_targets": row["duration_targets"],
            "duration_mask": None,
            "f0_targets": row["f0_targets"],
            "f0_mask": None,
        }
    else:
        if prosody_cache_path is None:
            raise ValueError("Prosody cache path is required when duration_targets/f0_targets are not embedded in the manifest.")
        payload = torch.load(prosody_cache_path, map_location="cpu", weights_only=False)
    dur = _tensor_1d(payload["duration_targets"], name="duration_targets")
    if dur.numel() != token_count:
        raise ValueError(f"duration_targets length {dur.numel()} != input token count {token_count}")
    dur_mask = payload.get("duration_mask")
    dur_mask = torch.ones_like(dur, dtype=torch.bool) if dur_mask is None else torch.as_tensor(dur_mask, dtype=torch.bool)
    if dur_mask.shape != dur.shape:
        raise ValueError(f"duration_mask shape {tuple(dur_mask.shape)} != duration_targets shape {tuple(dur.shape)}")
    f0 = _tensor_1d(payload["f0_targets"], name="f0_targets")
    f0_mask = payload.get("f0_mask")
    f0_mask = torch.ones_like(f0, dtype=torch.bool) if f0_mask is None else torch.as_tensor(f0_mask, dtype=torch.bool)
    if f0_mask.shape != f0.shape:
        raise ValueError(f"f0_mask shape {tuple(f0_mask.shape)} != f0_targets shape {tuple(f0.shape)}")
    return dur, dur_mask, f0, f0_mask


def _resolve_audio_path(path_value: str, *, manifest_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = manifest_root / path
    return path


def _pad_waveforms(rows: List[torch.Tensor], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([int(row.numel()) for row in rows], dtype=torch.long, device=device)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(rows), max_len, dtype=rows[0].dtype, device=device)
    for i, row in enumerate(rows):
        padded[i, : row.numel()] = row.to(device=device)
    return padded, lengths


def _iter_pending_rows(
    *,
    manifest_path: Path,
    manifest_root: Path,
    start_index: int,
    skip_existing: bool,
    cache_root: Path,
) -> Iterator[tuple[int, Dict[str, Any], Path]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or idx < int(start_index):
                continue
            row = json.loads(line)
            cache_path = default_cache_row_path(manifest_path, idx, cache_root=cache_root)
            if skip_existing and cache_path.is_file():
                continue
            yield idx, row, cache_path


def _prepare_cache_item(
    *,
    idx: int,
    row: Dict[str, Any],
    cache_path: Path,
    manifest_path: Path,
    manifest_root: Path,
    vocab: Dict[str, int],
    context_length: int,
    strict_single_chunk: bool,
    prosody_cache_root: Optional[Path],
    kokoro_repo_id: str,
) -> Dict[str, Any]:
    ref_path = _resolve_audio_path(str(row["ref_wav"]), manifest_root=manifest_root)
    tgt_path = _resolve_audio_path(str(row["target_wav"]), manifest_root=manifest_root)

    phonemes = row.get("phonemes")
    if phonemes is None:
        raise ValueError(
            "Feature cache building requires manifest phonemes to be precomputed. "
            "Populate `phonemes` ahead of time to avoid per-row G2P during cache building."
        )
    del strict_single_chunk, kokoro_repo_id
    input_ids = phonemes_to_input_ids(vocab, str(phonemes), context_length=context_length)
    prosody_path = None
    if prosody_cache_root is not None:
        prosody_path = default_prosody_row_path(manifest_path, idx, prosody_cache_root=prosody_cache_root)
    duration_targets, duration_mask, f0_targets, f0_mask = _load_row_prosody(
        row,
        token_count=int(input_ids.numel()),
        prosody_cache_path=prosody_path,
    )
    return {
        "idx": idx,
        "row": row,
        "cache_path": cache_path,
        "ref_wav": load_audio_mono(ref_path, target_sr=16_000),
        "tgt_wav": load_audio_mono(tgt_path, target_sr=16_000),
        "duration_targets": duration_targets,
        "duration_mask": duration_mask,
        "f0_targets": f0_targets,
        "f0_mask": f0_mask,
    }


def _iter_prepared_items(
    *,
    pending_rows: Iterator[tuple[int, Dict[str, Any], Path]],
    manifest_path: Path,
    manifest_root: Path,
    vocab: Dict[str, int],
    context_length: int,
    strict_single_chunk: bool,
    prosody_cache_root: Optional[Path],
    kokoro_repo_id: str,
    prefetch_workers: int,
    prefetch_count: int,
) -> Iterator[Dict[str, Any]]:
    if prefetch_workers <= 0:
        for idx, row, cache_path in pending_rows:
            yield _prepare_cache_item(
                idx=idx,
                row=row,
                cache_path=cache_path,
                manifest_path=manifest_path,
                manifest_root=manifest_root,
                vocab=vocab,
                context_length=context_length,
                strict_single_chunk=strict_single_chunk,
                prosody_cache_root=prosody_cache_root,
                kokoro_repo_id=kokoro_repo_id,
            )
        return

    with ThreadPoolExecutor(max_workers=prefetch_workers) as ex:
        queue: List[Future] = []

        def submit_next() -> bool:
            try:
                idx, row, cache_path = next(pending_rows)
            except StopIteration:
                return False
            fut = ex.submit(
                _prepare_cache_item,
                idx=idx,
                row=row,
                cache_path=cache_path,
                manifest_path=manifest_path,
                manifest_root=manifest_root,
                vocab=vocab,
                context_length=context_length,
                strict_single_chunk=strict_single_chunk,
                prosody_cache_root=prosody_cache_root,
                kokoro_repo_id=kokoro_repo_id,
            )
            queue.append(fut)
            return True

        for _ in range(max(prefetch_count, prefetch_workers)):
            if not submit_next():
                break

        while queue:
            fut = queue.pop(0)
            yield fut.result()
            submit_next()


def build_feature_cache_for_manifest(
    manifest_path: Path,
    *,
    kokoro_repo_id: str,
    cache_root: Path,
    prosody_cache_root: Optional[Path],
    mhubert: MHuBERTEncoder,
    wespeaker: WeSpeakerSV,
    manifest_root: Optional[Path] = None,
    strict_single_chunk: bool = True,
    skip_existing: bool = True,
    start_index: int = 0,
    batch_size: int = 8,
    prefetch_workers: int = 8,
    prefetch_count: int = 32,
    progress_interval: int = 250,
    use_amp: bool = True,
) -> None:
    vocab, context_length = kokoro_vocab_and_context_length(kokoro_repo_id)
    manifest_root = manifest_root or manifest_path.parent
    pipelines: Dict[str, KPipeline] = {}

    def pipeline_for_lang(lang_code: str) -> KPipeline:
        lang = normalize_lang_code(lang_code)
        if lang not in pipelines:
            pipelines[lang] = KPipeline(lang_code=lang, repo_id=kokoro_repo_id, model=False)
        return pipelines[lang]

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    model_device = next(mhubert.parameters()).device
    pending = _iter_pending_rows(
        manifest_path=manifest_path,
        manifest_root=manifest_root,
        start_index=start_index,
        skip_existing=skip_existing,
        cache_root=cache_root,
    )
    prepared_items = _iter_prepared_items(
        pending_rows=pending,
        manifest_path=manifest_path,
        manifest_root=manifest_root,
        vocab=vocab,
        context_length=context_length,
        strict_single_chunk=strict_single_chunk,
        prosody_cache_root=prosody_cache_root,
        kokoro_repo_id=kokoro_repo_id,
        prefetch_workers=prefetch_workers,
        prefetch_count=prefetch_count,
    )

    batch_items: List[Dict[str, Any]] = []
    total_written = 0
    total_decode_s = 0.0
    total_encode_s = 0.0
    total_save_s = 0.0
    wall_start = time.perf_counter()
    batch_ready_start = time.perf_counter()
    for item in prepared_items:
        batch_items.append(item)

        if len(batch_items) < batch_size:
            continue
        decode_s = time.perf_counter() - batch_ready_start
        encode_s, save_s = _flush_feature_cache_batch(
            batch_items,
            manifest_path=manifest_path,
            mhubert=mhubert,
            wespeaker=wespeaker,
            model_device=model_device,
            use_amp=use_amp,
        )
        total_written += len(batch_items)
        total_decode_s += decode_s
        total_encode_s += encode_s
        total_save_s += save_s
        batch_items.clear()
        batch_ready_start = time.perf_counter()
        if progress_interval > 0 and total_written % progress_interval == 0:
            elapsed = time.perf_counter() - wall_start
            rows_per_sec = total_written / max(elapsed, 1e-6)
            print(
                f"[cache_builder] wrote={total_written} rows elapsed={elapsed/60.0:.1f}m "
                f"rows_per_sec={rows_per_sec:.2f} "
                f"decode={total_decode_s/max(total_written,1):.4f}s/row "
                f"encode={total_encode_s/max(total_written,1):.4f}s/row "
                f"save={total_save_s/max(total_written,1):.4f}s/row"
            )

    if batch_items:
        decode_s = time.perf_counter() - batch_ready_start
        encode_s, save_s = _flush_feature_cache_batch(
            batch_items,
            manifest_path=manifest_path,
            mhubert=mhubert,
            wespeaker=wespeaker,
            model_device=model_device,
            use_amp=use_amp,
        )
        total_written += len(batch_items)
        total_decode_s += decode_s
        total_encode_s += encode_s
        total_save_s += save_s
    if total_written > 0:
        elapsed = time.perf_counter() - wall_start
        print(
            f"[cache_builder] finished rows={total_written} elapsed={elapsed/60.0:.1f}m "
            f"rows_per_sec={total_written/max(elapsed,1e-6):.2f} "
            f"decode={total_decode_s/max(total_written,1):.4f}s/row "
            f"encode={total_encode_s/max(total_written,1):.4f}s/row "
            f"save={total_save_s/max(total_written,1):.4f}s/row"
        )


def _flush_feature_cache_batch(
    batch_items: List[Dict[str, Any]],
    *,
    manifest_path: Path,
    mhubert: MHuBERTEncoder,
    wespeaker: WeSpeakerSV,
    model_device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    ref_batch, ref_lengths = _pad_waveforms([item["ref_wav"] for item in batch_items], device=model_device)
    ref_attn = torch.arange(ref_batch.size(1), device=model_device).unsqueeze(0) < ref_lengths.unsqueeze(1)
    tgt_batch, tgt_lengths = _pad_waveforms([item["tgt_wav"] for item in batch_items], device=model_device)

    encode_start = time.perf_counter()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and model_device.type == "cuda")
        if model_device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode():
        with amp_ctx:
            ref_out = mhubert(ref_batch, attention_mask=ref_attn.long())
            tgt_embed = wespeaker(
                tgt_batch,
                sampling_rate=wespeaker.sample_rate,
                waveform_lengths=tgt_lengths,
                grad_through_input=False,
                return_frame_features=False,
            ).pooled_embedding.detach()
    encode_s = time.perf_counter() - encode_start

    frame_lengths = ref_out.frame_mask.sum(dim=1).tolist()
    save_start = time.perf_counter()
    for i, item in enumerate(batch_items):
        n_frames = int(frame_lengths[i])
        payload = {
            "ref_hidden_states": ref_out.hidden_states[i, :n_frames].detach().to(dtype=torch.float16).cpu(),
            "ref_frame_mask": ref_out.frame_mask[i, :n_frames].detach().bool().cpu(),
            "target_wespeaker_embedding": tgt_embed[i].to(dtype=torch.float16).cpu(),
            "duration_targets": item["duration_targets"].cpu(),
            "duration_mask": item["duration_mask"].cpu(),
            "f0_targets": item["f0_targets"].to(dtype=torch.float16).cpu(),
            "f0_mask": item["f0_mask"].cpu(),
            "manifest_fingerprint": build_manifest_row_fingerprint(item["row"], index=item["idx"]),
            "row_index": item["idx"],
            "manifest_path": str(manifest_path),
        }
        item["cache_path"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, item["cache_path"])
    save_s = time.perf_counter() - save_start
    return encode_s, save_s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline feature caches for a manifest.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--kokoro-repo", type=str, default="hexgrad/Kokoro-82M")
    p.add_argument("--cache-root", type=Path, default=Path("cache"))
    p.add_argument("--prosody-cache-root", type=Path, default=Path("prosody_cache"))
    p.add_argument("--manifest-root", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prefetch-workers", type=int, default=8)
    p.add_argument("--prefetch-count", type=int, default=32)
    p.add_argument("--progress-interval", type=int, default=250)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    mhubert = MHuBERTEncoder().to(device)
    wespeaker = WeSpeakerSV.from_checkpoint(
        "voice_clone/encoder-ckpts/wespeaker-ckpt/models/avg_model.pt",
        embedding_dim=256,
        sample_rate=16_000,
        device=device,
    )
    build_feature_cache_for_manifest(
        args.manifest,
        kokoro_repo_id=args.kokoro_repo,
        cache_root=args.cache_root,
        prosody_cache_root=args.prosody_cache_root,
        mhubert=mhubert,
        wespeaker=wespeaker,
        manifest_root=args.manifest_root,
        skip_existing=args.skip_existing,
        start_index=args.start_index,
        batch_size=args.batch_size,
        prefetch_workers=args.prefetch_workers,
        prefetch_count=args.prefetch_count,
        progress_interval=args.progress_interval,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
