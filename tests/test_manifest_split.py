"""Tests for speaker-level val split and manifest line builder."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "scripts" / "build_common_voice_manifest.py"
_spec = importlib.util.spec_from_file_location("build_common_voice_manifest", _SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_SCRIPT_PATH}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[str(_spec.name)] = _mod

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "kokoro"))
_spec.loader.exec_module(_mod)

JoinedRow = _mod.JoinedRow
split_speakers_for_val = _mod.split_speakers_for_val
build_manifest_lines = _mod.build_manifest_lines
_derive_val_path = _mod._derive_val_path
filter_rows_with_client_id = _mod.filter_rows_with_client_id
join_validated = _mod.join_validated
load_existing_clip_relpaths = _mod.load_existing_clip_relpaths
main = _mod.main

# Pre-seed the cached pipeline maps so kokoro.pipeline is never imported
# (avoids loguru / heavy model dependencies in unit tests).
_FAKE_LANG_KEYS = frozenset({"a", "b", "e", "f", "h", "i", "j", "p", "z"})
_FAKE_ALIASES: Dict[str, str] = {"en-us": "a", "en-gb": "b", "hi": "h"}


def _patch_pipeline_maps():
    """Context manager that patches the lazy-loaded pipeline maps."""
    return patch.multiple(
        _mod,
        _pipeline_aliases=_FAKE_ALIASES,
        _pipeline_lang_keys=_FAKE_LANG_KEYS,
    )


def _make_rows(speakers: Dict[str, int], *, locale: str = "hi", accents: str = "") -> List[JoinedRow]:
    """Build synthetic rows: ``speakers`` maps client_id to clip count."""
    rows: List[JoinedRow] = []
    clip_idx = 0
    for spk, n in speakers.items():
        for _ in range(n):
            rows.append(
                JoinedRow(
                    path=f"clip_{clip_idx}.mp3",
                    client_id=spk,
                    sentence=f"sentence {clip_idx}",
                    duration_sec=4.0,
                    stratum="__empty__|__empty__|__empty__",
                    locale=locale,
                    accents=accents,
                )
            )
            clip_idx += 1
    return rows


# ---------------------------------------------------------------------------
# split_speakers_for_val
# ---------------------------------------------------------------------------

class TestSplitSpeakersForVal:
    def test_no_speaker_leakage(self) -> None:
        rows = _make_rows({"A": 5, "B": 3, "C": 4, "D": 2, "E": 6})
        train, val = split_speakers_for_val(rows, val_fraction=0.4, seed=42)
        train_spk = set(r.client_id for r in train)
        val_spk = set(r.client_id for r in val)
        assert train_spk & val_spk == set(), "speakers must not appear in both train and val"

    def test_all_rows_preserved(self) -> None:
        rows = _make_rows({"A": 5, "B": 3, "C": 4})
        train, val = split_speakers_for_val(rows, val_fraction=0.3, seed=7)
        assert len(train) + len(val) == len(rows)

    def test_deterministic(self) -> None:
        rows = _make_rows({"A": 5, "B": 3, "C": 4, "D": 2})
        t1, v1 = split_speakers_for_val(rows, val_fraction=0.25, seed=99)
        t2, v2 = split_speakers_for_val(rows, val_fraction=0.25, seed=99)
        assert [r.path for r in t1] == [r.path for r in t2]
        assert [r.path for r in v1] == [r.path for r in v2]

    def test_different_seed_different_split(self) -> None:
        rows = _make_rows({f"s{i}": 3 for i in range(20)})
        _, v1 = split_speakers_for_val(rows, val_fraction=0.3, seed=1)
        _, v2 = split_speakers_for_val(rows, val_fraction=0.3, seed=2)
        assert set(r.client_id for r in v1) != set(r.client_id for r in v2)

    def test_minimum_one_val_speaker(self) -> None:
        rows = _make_rows({"A": 5, "B": 3})
        _, val = split_speakers_for_val(rows, val_fraction=0.01, seed=42)
        assert len(set(r.client_id for r in val)) >= 1


# ---------------------------------------------------------------------------
# build_manifest_lines
# ---------------------------------------------------------------------------

class TestBuildManifestLines:
    def test_ref_wav_same_speaker(self) -> None:
        rows = _make_rows({"A": 4, "B": 3})
        with _patch_pipeline_maps():
            lines, _, _ = build_manifest_lines(
                rows, lang_code_override="h", fallback_locale="hi", seed=42,
            )
        for item in lines:
            rec = item["rec"]
            assert rec["speaker_id"] in ("A", "B")
            target_spk = rec["speaker_id"]
            ref_path = rec["ref_wav"]
            target_path = rec["target_wav"]
            assert ref_path != target_path, "ref_wav must differ from target_wav"
            ref_spk = _find_speaker_for_path(rows, ref_path.replace("clips/", ""))
            assert ref_spk == target_spk, "ref_wav must come from the same speaker"

    def test_single_clip_speakers_dropped(self) -> None:
        rows = _make_rows({"A": 1, "B": 3})
        with _patch_pipeline_maps():
            lines, dropped, _ = build_manifest_lines(
                rows, lang_code_override="h", fallback_locale="hi", seed=42,
            )
        assert dropped == 1
        spk_ids = {item["rec"]["speaker_id"] for item in lines}
        assert "A" not in spk_ids

    def test_deterministic_output(self) -> None:
        rows = _make_rows({"A": 4, "B": 3})
        with _patch_pipeline_maps():
            l1, _, _ = build_manifest_lines(
                rows, lang_code_override="h", fallback_locale="hi", seed=42,
            )
            l2, _, _ = build_manifest_lines(
                rows, lang_code_override="h", fallback_locale="hi", seed=42,
            )
        assert l1 == l2

    def test_blank_client_id_rows_are_excluded(self) -> None:
        rows = _make_rows({"A": 2})
        rows.extend(
            [
                JoinedRow(
                    path="blank_0.mp3",
                    client_id="",
                    sentence="blank 0",
                    duration_sec=4.0,
                    stratum="__empty__|__empty__|__empty__",
                    locale="hi",
                    accents="",
                ),
                JoinedRow(
                    path="blank_1.mp3",
                    client_id="",
                    sentence="blank 1",
                    duration_sec=4.0,
                    stratum="__empty__|__empty__|__empty__",
                    locale="hi",
                    accents="",
                ),
            ]
        )
        with _patch_pipeline_maps():
            lines, dropped, _ = build_manifest_lines(
                rows, lang_code_override="h", fallback_locale="hi", seed=42,
            )
        assert len(lines) == 2
        assert dropped == 0
        assert all(item["rec"]["speaker_id"] == "A" for item in lines)
        assert all("blank_" not in item["rec"]["target_wav"] for item in lines)


class TestFilterRowsWithClientId:
    def test_drops_blank_client_ids(self) -> None:
        rows = _make_rows({"A": 2})
        rows.append(
            JoinedRow(
                path="blank.mp3",
                client_id="   ",
                sentence="blank",
                duration_sec=4.0,
                stratum="__empty__|__empty__|__empty__",
                locale="hi",
                accents="",
            )
        )
        kept, dropped = filter_rows_with_client_id(rows)
        assert dropped == 1
        assert [row.client_id for row in kept] == ["A", "A"]


class TestJoinValidated:
    def test_requires_exact_relative_clip_path(self, tmp_path: Path) -> None:
        locale_dir = tmp_path / "hi"
        clips_dir = locale_dir / "clips"
        clips_dir.mkdir(parents=True)
        (clips_dir / "clip_0.mp3").write_bytes(b"audio")

        _write_tsv(
            locale_dir / "validated.tsv",
            fieldnames=["client_id", "path", "sentence", "locale", "accents", "gender", "age"],
            rows=[
                {
                    "client_id": "speaker-a",
                    "path": "nested/clip_0.mp3",
                    "sentence": "hello",
                    "locale": "hi",
                    "accents": "",
                    "gender": "",
                    "age": "",
                }
            ],
        )

        rows, stats = join_validated(
            locale_dir / "validated.tsv",
            {"nested/clip_0.mp3": 1.0},
            existing_clip_relpaths=load_existing_clip_relpaths(clips_dir),
            include_variant=False,
            exclude_paths=None,
        )

        assert rows == []
        assert stats["missing_clip_file"] == 1

    def test_accepts_nested_relative_clip_path_when_present(self, tmp_path: Path) -> None:
        locale_dir = tmp_path / "hi"
        clips_dir = locale_dir / "clips"
        (clips_dir / "nested").mkdir(parents=True)
        (clips_dir / "nested" / "clip_0.mp3").write_bytes(b"audio")

        _write_tsv(
            locale_dir / "validated.tsv",
            fieldnames=["client_id", "path", "sentence", "locale", "accents", "gender", "age"],
            rows=[
                {
                    "client_id": "speaker-a",
                    "path": "nested/clip_0.mp3",
                    "sentence": "hello",
                    "locale": "hi",
                    "accents": "",
                    "gender": "",
                    "age": "",
                }
            ],
        )

        rows, stats = join_validated(
            locale_dir / "validated.tsv",
            {"nested/clip_0.mp3": 1.0},
            existing_clip_relpaths=load_existing_clip_relpaths(clips_dir),
            include_variant=False,
            exclude_paths=None,
        )

        assert [row.path for row in rows] == ["nested/clip_0.mp3"]
        assert stats["missing_clip_file"] == 0


# ---------------------------------------------------------------------------
# _derive_val_path
# ---------------------------------------------------------------------------

class TestDeriveValPath:
    def test_replaces_train_with_val(self) -> None:
        p = Path("manifests/hi_train.jsonl")
        assert _derive_val_path(p) == Path("manifests/hi_val.jsonl")

    def test_appends_val_when_no_train(self) -> None:
        p = Path("manifests/hi.jsonl")
        assert _derive_val_path(p) == Path("manifests/hi_val.jsonl")


class TestMainValidationSplit:
    def test_requires_two_non_empty_speakers_for_val_split(self, tmp_path: Path) -> None:
        locale_dir = tmp_path / "hi"
        locale_dir.mkdir()
        _write_tsv(
            locale_dir / "validated.tsv",
            fieldnames=["client_id", "path", "sentence", "locale", "accents", "gender", "age"],
            rows=[
                {
                    "client_id": "speaker-a",
                    "path": "clip_0.mp3",
                    "sentence": "hello",
                    "locale": "hi",
                    "accents": "",
                    "gender": "",
                    "age": "",
                },
                {
                    "client_id": "speaker-a",
                    "path": "clip_1.mp3",
                    "sentence": "world",
                    "locale": "hi",
                    "accents": "",
                    "gender": "",
                    "age": "",
                },
            ],
        )
        _write_tsv(
            locale_dir / "clip_durations.tsv",
            fieldnames=["clip", "duration[ms]"],
            rows=[
                {"clip": "clip_0.mp3", "duration[ms]": "1000"},
                {"clip": "clip_1.mp3", "duration[ms]": "1000"},
            ],
        )

        with _patch_pipeline_maps():
            rc = main(
                [
                    "--locale-dir",
                    str(locale_dir),
                    "--out",
                    str(tmp_path / "train.jsonl"),
                    "--skip-g2p-check",
                    "--val-fraction",
                    "0.5",
                ]
            )

        assert rc == 1
        assert not (tmp_path / "train.jsonl").exists()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_speaker_for_path(rows: List[JoinedRow], path: str) -> str:
    for r in rows:
        if r.path == path:
            return r.client_id
    raise ValueError(f"path {path!r} not found in rows")


def _write_tsv(path: Path, *, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
