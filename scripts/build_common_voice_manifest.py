#!/usr/bin/env python3
"""Build a voice-clone JSONL manifest from Mozilla Common Voice (validated + clip durations).

Joins ``validated.tsv`` with ``clip_durations.tsv``, optionally caps total audio with
stratified sampling (by accents / gender / age), filters lines that do not phonemize as
a single Kokoro segment, and writes JSONL plus a metadata sidecar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

# CV ``accents`` substrings that suggest British English for generic ``en`` locale.
_UK_ACCENT_HINTS = (
    "england",
    "scotland",
    "scottish",
    "welsh",
    "irish",
    "british",
    "uk",
    "wales",
    "northern ireland",
    "received pronunciation",
    "rp",
)

# ---------------------------------------------------------------------------
# Common Voice ``locale`` column → Kokoro ``lang_code`` (single-char keys in
# ``kokoro.pipeline.LANG_CODES``). Mirrors typical Mozilla Common Voice locale
# ids (folder names use the same strings, lowercased with ``_`` → ``-``).
#
# English: generic ``en`` is *not* listed here; use ``_infer_en_kokoro_code``.
# Portuguese: Kokoro ``p`` is **Brazilian** (``pt-br``). Generic ``pt`` is
# mapped to ``p`` because many CV dumps label Brazilian data as ``pt``; use
# ``--lang-code`` if your tree is European. **European Portuguese** locale ids
# below are blocked unless ``--lang-code`` overrides inference.
# ---------------------------------------------------------------------------

# CV locale strings that must not auto-map to Kokoro ``p`` (no EU G2P in Kokoro-82M).
_EU_PORTUGUESE_CV_LOCALES = frozenset({"pt-pt", "pt_pt"})

# Direct mappings for CV ``locale`` after ``lower()`` and ``_`` → ``-``.
CV_LOCALE_TO_KOKORO: Dict[str, str] = {
    "hi": "h",
    "es": "e",
    "fr": "f",
    "fr-fr": "f",
    "it": "i",
    "ja": "j",
    "zh": "z",
    "zh-cn": "z",
    "zh-tw": "z",
    "pt": "p",
    "pt-br": "p",
    "pt_br": "p",
    "en-us": "a",
    "en-gb": "b",
    # Other ``en-*`` (en-au, en-ca, …) handled in ``infer_kokoro_lang_code``.
}


def _repo_sys_path() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "kokoro"))


# Cached copies of ``kokoro.pipeline.ALIASES`` / ``LANG_CODES`` keys (stay in sync at runtime).
_pipeline_aliases: Optional[Dict[str, str]] = None
_pipeline_lang_keys: Optional[frozenset] = None


def _kokoro_pipeline_maps() -> Tuple[Dict[str, str], frozenset]:
    global _pipeline_aliases, _pipeline_lang_keys
    if _pipeline_aliases is None:
        _repo_sys_path()
        from kokoro.pipeline import ALIASES, LANG_CODES

        _pipeline_aliases = dict(ALIASES)
        _pipeline_lang_keys = frozenset(LANG_CODES.keys())
    assert _pipeline_aliases is not None and _pipeline_lang_keys is not None
    return _pipeline_aliases, _pipeline_lang_keys


def stratum_key(row: Dict[str, str], *, include_variant: bool) -> str:
    parts = [
        (row.get("accents") or "").strip() or "__empty__",
        (row.get("gender") or "").strip() or "__empty__",
        (row.get("age") or "").strip() or "__empty__",
    ]
    if include_variant:
        parts.append((row.get("variant") or "").strip() or "__empty__")
    return "|".join(parts)


def sub_seed(main_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{main_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def normalize_kokoro_lang_code(lang_code: str) -> str:
    aliases, lang_keys = _kokoro_pipeline_maps()
    c = lang_code.lower().strip()
    c = aliases.get(c, c)
    if c not in lang_keys:
        raise ValueError(
            f"Unknown lang_code {lang_code!r}; expected one of {sorted(lang_keys)} "
            f"(or aliases like en-us). See kokoro.pipeline.ALIASES / LANG_CODES."
        )
    return c


def _infer_en_kokoro_code(*, loc: str, accents: str) -> str:
    """Map English CV locale + accent hints to Kokoro ``a`` (US) or ``b`` (UK)."""
    acc = accents.strip().lower()
    if loc == "en":
        if any(h in acc for h in _UK_ACCENT_HINTS):
            return normalize_kokoro_lang_code("b")
        return normalize_kokoro_lang_code("a")
    if loc.startswith("en-"):
        if "gb" in loc or loc.endswith("uk"):
            return normalize_kokoro_lang_code("b")
        return normalize_kokoro_lang_code("a")
    raise AssertionError("internal: _infer_en_kokoro_code expects loc == 'en' or en-*")


def infer_kokoro_lang_code(
    *,
    locale: str,
    accents: str,
    lang_code_override: Optional[str],
    fallback_locale_from_dir: str,
) -> str:
    if lang_code_override:
        return normalize_kokoro_lang_code(lang_code_override)

    loc = (locale or "").strip().lower().replace("_", "-")
    acc = (accents or "").strip().lower()
    if not loc:
        loc = fallback_locale_from_dir.strip().lower().replace("_", "-")

    if loc in _EU_PORTUGUESE_CV_LOCALES:
        raise ValueError(
            "European Portuguese (pt-PT) has no Kokoro G2P in LANG_CODES; Kokoro ``p`` is Brazilian. "
            "Use --lang-code p only if you accept Brazilian phonemization, or train without this locale."
        )

    if loc in CV_LOCALE_TO_KOKORO:
        return normalize_kokoro_lang_code(CV_LOCALE_TO_KOKORO[loc])
    if loc == "en" or loc.startswith("en-"):
        return _infer_en_kokoro_code(loc=loc, accents=acc)

    try:
        return normalize_kokoro_lang_code(loc)
    except ValueError:
        fb = fallback_locale_from_dir.strip().lower().replace("_", "-")
        return normalize_kokoro_lang_code(fb)


def load_clip_durations(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip = (row.get("clip") or "").strip()
            if not clip:
                continue
            raw = (row.get("duration[ms]") or row.get("duration") or "").strip()
            if not raw:
                continue
            try:
                ms = float(raw)
            except ValueError:
                continue
            out[clip] = ms / 1000.0
    return out


def load_test_paths(test_tsv: Path) -> set[str]:
    paths: set[str] = set()
    with test_tsv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            p = (row.get("path") or "").strip()
            if p:
                paths.add(p)
    return paths


@dataclass
class JoinedRow:
    path: str
    client_id: str
    sentence: str
    duration_sec: float
    stratum: str
    locale: str
    accents: str


def join_validated(
    validated_path: Path,
    durations: Dict[str, float],
    *,
    include_variant: bool,
    exclude_paths: Optional[set[str]],
) -> Tuple[List[JoinedRow], Dict[str, int]]:
    stats = {
        "validated_rows": 0,
        "missing_duration": 0,
        "excluded_test": 0,
        "duplicate_path_skipped": 0,
    }
    seen: set[str] = set()
    joined: List[JoinedRow] = []
    with validated_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["validated_rows"] += 1
            path = (row.get("path") or "").strip()
            if not path:
                continue
            if exclude_paths is not None and path in exclude_paths:
                stats["excluded_test"] += 1
                continue
            if path in seen:
                stats["duplicate_path_skipped"] += 1
                continue
            dur = durations.get(path)
            if dur is None:
                stats["missing_duration"] += 1
                continue
            seen.add(path)
            joined.append(
                JoinedRow(
                    path=path,
                    client_id=(row.get("client_id") or "").strip(),
                    sentence=(row.get("sentence") or "").strip(),
                    duration_sec=dur,
                    stratum=stratum_key(row, include_variant=include_variant),
                    locale=(row.get("locale") or "").strip(),
                    accents=(row.get("accents") or "").strip(),
                )
            )
    return joined, stats


def stratum_durations(rows: Iterable[JoinedRow]) -> Dict[str, float]:
    acc: Dict[str, float] = defaultdict(float)
    for r in rows:
        acc[r.stratum] += r.duration_sec
    return dict(acc)


def stratified_sample(
    rows: List[JoinedRow],
    *,
    cap_sec: float,
    seed: int,
) -> Tuple[List[JoinedRow], Dict[str, Any]]:
    """When total duration exceeds ``cap_sec``, pick clips per stratum with proportional quotas.

    Phase 1 adds clips within each stratum only while ``running_sum + clip_dur <= stratum_target``,
    so stratum targets are not blown past by a long last clip. Phase 2 shuffles the remaining
    pool and greedily adds clips while ``total <= cap_sec``, so the final duration never exceeds
    the cap (aside from floating noise).
    """
    by_s: Dict[str, List[JoinedRow]] = defaultdict(list)
    for r in rows:
        by_s[r.stratum].append(r)

    total_dur = sum(r.duration_sec for r in rows)
    targets: Dict[str, float] = {}
    for s, rs in by_s.items():
        t_s = sum(x.duration_sec for x in rs)
        targets[s] = cap_sec * (t_s / total_dur)

    selected: List[JoinedRow] = []
    selected_paths: set[str] = set()
    phase1_per_stratum: Dict[str, Tuple[int, float]] = {}

    for s in sorted(by_s.keys()):
        pool = list(by_s[s])
        rng = random.Random(sub_seed(seed, s))
        rng.shuffle(pool)
        goal = targets[s]
        got = 0.0
        n = 0
        for r in pool:
            if got >= goal - 1e-9:
                break
            if got + r.duration_sec <= goal + 1e-9:
                selected.append(r)
                selected_paths.add(r.path)
                got += r.duration_sec
                n += 1
        phase1_per_stratum[s] = (n, got)

    total_sel = sum(r.duration_sec for r in selected)
    budget = max(0.0, cap_sec - total_sel)
    remainder = [r for r in rows if r.path not in selected_paths]
    rng2 = random.Random(sub_seed(seed, "__phase2__"))
    rng2.shuffle(remainder)
    phase2_count = 0
    for r in remainder:
        if budget <= 1e-9:
            break
        if r.duration_sec <= budget + 1e-9:
            selected.append(r)
            selected_paths.add(r.path)
            budget -= r.duration_sec
            phase2_count += 1

    stratum_actual: Dict[str, float] = defaultdict(float)
    stratum_counts: Dict[str, int] = defaultdict(int)
    for r in selected:
        stratum_actual[r.stratum] += r.duration_sec
        stratum_counts[r.stratum] += 1

    audit: Dict[str, Any] = {
        "stratum_target_seconds": {k: round(v, 3) for k, v in sorted(targets.items())},
        "stratum_actual_seconds": {k: round(v, 3) for k, v in sorted(stratum_actual.items())},
        "stratum_row_counts": dict(sorted(stratum_counts.items())),
        "phase1_rows_per_stratum": {k: v[0] for k, v in sorted(phase1_per_stratum.items())},
        "phase2_rows_added": phase2_count,
    }
    return selected, audit


def precheck_text_single_kokoro_segment(
    pipeline: Any,
    text: str,
    *,
    max_phoneme_chars: int = 510,
) -> None:
    """Quiet ``KPipeline`` (``model=False``) + same rules as training-time G2P.

    Drops rows that would multi-chunk or exceed ``max_phoneme_chars`` phoneme characters
    (Kokoro / ``voice_clone.dataset.text_to_phonemes`` contract; see DATASET.md).
    """
    from voice_clone.dataset import text_to_phonemes

    text_to_phonemes(
        pipeline,
        text,
        strict_single_chunk=True,
        max_phoneme_chars=max_phoneme_chars,
    )


def g2p_filter_rows(
    rows: List[JoinedRow],
    *,
    kokoro_repo_id: str,
    fallback_locale_from_dir: str,
    lang_code_override: Optional[str],
) -> Tuple[List[JoinedRow], Dict[str, int]]:
    from kokoro.pipeline import KPipeline

    stats: Dict[str, int] = {
        "g2p_ok": 0,
        "g2p_dropped": 0,
        "g2p_lang_infer_failed": 0,
        "g2p_empty_text": 0,
        "g2p_multi_chunk": 0,
        "g2p_phoneme_too_long": 0,
        "g2p_other": 0,
    }
    pipelines: Dict[str, KPipeline] = {}
    kept: List[JoinedRow] = []

    def pipe_for(lc: str) -> KPipeline:
        if lc not in pipelines:
            pipelines[lc] = KPipeline(lang_code=lc, repo_id=kokoro_repo_id, model=False)
        return pipelines[lc]

    def _bump_drop(reason: str) -> None:
        stats["g2p_dropped"] += 1
        stats[reason] += 1

    for r in rows:
        try:
            lc = infer_kokoro_lang_code(
                locale=r.locale,
                accents=r.accents,
                lang_code_override=lang_code_override,
                fallback_locale_from_dir=fallback_locale_from_dir,
            )
        except ValueError:
            _bump_drop("g2p_lang_infer_failed")
            continue
        if not r.sentence:
            _bump_drop("g2p_empty_text")
            continue
        try:
            precheck_text_single_kokoro_segment(pipe_for(lc), r.sentence)
        except ValueError as err:
            msg = str(err)
            if "chunks" in msg:
                _bump_drop("g2p_multi_chunk")
            elif "max_phoneme_chars" in msg or "exceeds max_phoneme" in msg:
                _bump_drop("g2p_phoneme_too_long")
            else:
                _bump_drop("g2p_other")
            continue
        except Exception:
            _bump_drop("g2p_other")
            continue
        kept.append(r)
        stats["g2p_ok"] += 1

    return kept, stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--locale-dir",
        type=Path,
        required=True,
        help="Common Voice locale directory (contains validated.tsv, clip_durations.tsv, clips/).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL manifest path.",
    )
    p.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Metadata JSON path (default: alongside --out with .meta.json suffix).",
    )
    p.add_argument(
        "--hours",
        type=float,
        default=20.0,
        help="Target hours of audio when the joined corpus is larger (default: 20).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for stratified sampling.")
    p.add_argument(
        "--exclude-test",
        action="store_true",
        help="Exclude clip paths that appear in test.tsv (same locale directory).",
    )
    p.add_argument(
        "--lang-code",
        default=None,
        help="Force Kokoro lang_code for every row (e.g. h for Hindi).",
    )
    p.add_argument(
        "--kokoro-repo",
        default="hexgrad/Kokoro-82M",
        help="HF repo_id for quiet KPipeline G2P checks.",
    )
    p.add_argument(
        "--skip-g2p-check",
        action="store_true",
        help="Do not run KPipeline; skip multi-chunk / length validation.",
    )
    p.add_argument(
        "--strata-include-variant",
        action="store_true",
        help="Include Common Voice ``variant`` in stratification key.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _repo_sys_path()

    locale_dir = args.locale_dir.resolve()
    validated_tsv = locale_dir / "validated.tsv"
    durations_tsv = locale_dir / "clip_durations.tsv"
    if not validated_tsv.is_file():
        print(f"Missing {validated_tsv}", file=sys.stderr)
        return 1
    if not durations_tsv.is_file():
        print(f"Missing {durations_tsv}", file=sys.stderr)
        return 1

    fallback_locale = locale_dir.name.lower().replace("_", "-")

    exclude: Optional[set[str]] = None
    if args.exclude_test:
        test_tsv = locale_dir / "test.tsv"
        if not test_tsv.is_file():
            print(f"--exclude-test but missing {test_tsv}", file=sys.stderr)
            return 1
        exclude = load_test_paths(test_tsv)

    durations = load_clip_durations(durations_tsv)
    joined, join_stats = join_validated(
        validated_tsv,
        durations,
        include_variant=args.strata_include_variant,
        exclude_paths=exclude,
    )

    total_sec = sum(r.duration_sec for r in joined)
    cap_sec = min(total_sec, max(0.0, float(args.hours)) * 3600.0)

    sample_audit: Dict[str, Any]
    if total_sec <= cap_sec or not joined:
        pool = list(joined)
        rng = random.Random(args.seed)
        rng.shuffle(pool)
        selected = pool
        sample_audit = {
            "mode": "use_all",
            "stratum_target_seconds": {},
            "stratum_actual_seconds": {k: round(v, 3) for k, v in sorted(stratum_durations(selected).items())},
            "stratum_row_counts": {
                k: sum(1 for r in selected if r.stratum == k) for k in sorted(set(r.stratum for r in selected))
            },
        }
    else:
        selected, sample_audit = stratified_sample(joined, cap_sec=cap_sec, seed=args.seed)
        sample_audit["mode"] = "stratified_cap"
        rng = random.Random(args.seed + 1)
        rng.shuffle(selected)

    g2p_stats: Dict[str, int] = {"g2p_ok": len(selected), "g2p_dropped": 0}
    if not args.skip_g2p_check:
        _repo_sys_path()
        selected, g2p_stats = g2p_filter_rows(
            selected,
            kokoro_repo_id=args.kokoro_repo,
            fallback_locale_from_dir=fallback_locale,
            lang_code_override=args.lang_code,
        )
        rng = random.Random(args.seed + 2)
        rng.shuffle(selected)

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.metadata_out
    if meta_path is None:
        meta_path = out_path.parent / f"{out_path.stem}.meta.json"
    else:
        meta_path = meta_path.resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[Dict[str, Any]] = []
    lang_infer_failures = 0
    for r in selected:
        try:
            lc = infer_kokoro_lang_code(
                locale=r.locale,
                accents=r.accents,
                lang_code_override=args.lang_code,
                fallback_locale_from_dir=fallback_locale,
            )
        except ValueError as e:
            lang_infer_failures += 1
            print(f"Skipping row {r.path!r}: {e}", file=sys.stderr)
            continue
        rel_audio = f"clips/{r.path}"
        lines.append(
            {
                "_dur": r.duration_sec,
                "rec": {
                    "ref_wav": rel_audio,
                    "target_wav": rel_audio,
                    "text": r.sentence,
                    "lang_code": lc,
                    "speaker_id": r.client_id,
                },
            }
        )

    hours_manifest = sum(item["_dur"] for item in lines) / 3600.0
    rows_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item["rec"], ensure_ascii=False) + "\n")
            rows_written += 1

    meta: Dict[str, Any] = {
        "seed": args.seed,
        "locale_dir": str(locale_dir),
        "fallback_locale_from_dir": fallback_locale,
        "hours_requested": float(args.hours),
        "hours_total_joined": round(total_sec / 3600.0, 6),
        "hours_cap_applied": round(cap_sec / 3600.0, 6),
        "hours_in_manifest": round(hours_manifest, 6),
        "row_count": rows_written,
        "joined_unique_paths": len(joined),
        "sampling": sample_audit,
        "join_stats": join_stats,
        "g2p_stats": g2p_stats,
        "lang_infer_failures": lang_infer_failures,
        "kokoro_repo_id": args.kokoro_repo,
        "skip_g2p_check": bool(args.skip_g2p_check),
        "lang_code_override": args.lang_code,
        "strata_include_variant": bool(args.strata_include_variant),
        "exclude_test": bool(args.exclude_test),
        "note_pt_br": "Kokoro lang_code p is Brazilian Portuguese; European Portuguese may need a separate policy.",
        "kokoro_locale_policy": (
            "Locale→lang_code: see CV_LOCALE_TO_KOKORO and _EU_PORTUGUESE_CV_LOCALES in this script; "
            "generic en uses _UK_ACCENT_HINTS for a vs b; pt-PT is rejected unless --lang-code. "
            "G2P precheck: quiet KPipeline + precheck_text_single_kokoro_segment (single chunk, ≤510 phoneme chars)."
        ),
    }
    with meta_path.open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)
        mf.write("\n")

    print(f"Wrote {rows_written} rows to {out_path}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
