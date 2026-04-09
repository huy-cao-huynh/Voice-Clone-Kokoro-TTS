"""Tests for multilingual manifest merging."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "scripts" / "merge_manifests.py"
_spec = importlib.util.spec_from_file_location("merge_manifests", _SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_SCRIPT_PATH}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[str(_spec.name)] = _mod
_spec.loader.exec_module(_mod)

main = _mod.main
merge_manifest_files = _mod.merge_manifest_files


def test_merge_manifest_files_rewrites_paths_and_builds_summary(tmp_path: Path) -> None:
    manifests_dir = tmp_path / "manifests"
    data_hi = tmp_path / "data" / "hi"
    data_es = tmp_path / "data" / "es"
    manifests_dir.mkdir(parents=True)
    data_hi.mkdir(parents=True)
    data_es.mkdir(parents=True)

    hi_manifest = manifests_dir / "hi_train.jsonl"
    es_manifest = manifests_dir / "es_train.jsonl"
    _write_manifest(
        hi_manifest,
        {"ref_wav": "clips/hi_ref.wav", "target_wav": "clips/hi_tgt.wav", "text": "namaste", "lang_code": "h"},
        {"ref_wav": "clips/hi_ref2.wav", "target_wav": "clips/hi_tgt2.wav", "text": "duniya", "lang_code": "h"},
    )
    _write_manifest(
        es_manifest,
        {"ref_wav": "clips/es_ref.wav", "target_wav": "clips/es_tgt.wav", "text": "hola", "lang_code": "e"},
    )
    _write_meta(hi_manifest, locale_dir=data_hi, hours_in_manifest=1.25, split="train")
    _write_meta(es_manifest, locale_dir=data_es, hours_in_manifest=2.5, split="train")

    rows, meta = merge_manifest_files(
        [es_manifest, hi_manifest],
        output_manifest_path=manifests_dir / "multilingual_train.jsonl",
        seed=7,
        shuffle=False,
        split="train",
    )

    assert [row["lang_code"] for row in rows] == ["e", "h", "h"]
    assert rows[0]["ref_wav"] == "../data/es/clips/es_ref.wav"
    assert rows[1]["target_wav"] == "../data/hi/clips/hi_tgt.wav"
    assert meta["lang_code_totals"] == {"e": 1, "h": 2}
    assert meta["source_row_totals"] == {"es_train.jsonl": 1, "hi_train.jsonl": 2}
    assert meta["hours_in_manifest_total"] == 3.75


def test_main_merges_train_and_val_manifests(tmp_path: Path) -> None:
    manifests_dir = tmp_path / "manifests"
    data_hi = tmp_path / "data" / "hi"
    data_es = tmp_path / "data" / "es"
    manifests_dir.mkdir(parents=True)
    data_hi.mkdir(parents=True)
    data_es.mkdir(parents=True)

    hi_train = manifests_dir / "hi_train.jsonl"
    es_train = manifests_dir / "es_train.jsonl"
    hi_val = manifests_dir / "hi_val.jsonl"
    es_val = manifests_dir / "es_val.jsonl"

    _write_manifest(hi_train, {"ref_wav": "clips/a.wav", "target_wav": "clips/b.wav", "text": "one", "lang_code": "h"})
    _write_manifest(es_train, {"ref_wav": "clips/c.wav", "target_wav": "clips/d.wav", "text": "dos", "lang_code": "e"})
    _write_manifest(hi_val, {"ref_wav": "clips/e.wav", "target_wav": "clips/f.wav", "text": "three", "lang_code": "h"})
    _write_manifest(es_val, {"ref_wav": "clips/g.wav", "target_wav": "clips/h.wav", "text": "cuatro", "lang_code": "e"})

    _write_meta(hi_train, locale_dir=data_hi, hours_in_manifest=1.0, split="train")
    _write_meta(es_train, locale_dir=data_es, hours_in_manifest=1.5, split="train")
    _write_meta(hi_val, locale_dir=data_hi, hours_in_manifest=0.25, split="val")
    _write_meta(es_val, locale_dir=data_es, hours_in_manifest=0.5, split="val")

    rc = main(["--manifests-dir", str(manifests_dir), "--seed", "123"])
    assert rc == 0

    train_rows = _read_jsonl(manifests_dir / "multilingual_train.jsonl")
    val_rows = _read_jsonl(manifests_dir / "multilingual_val.jsonl")
    train_meta = _read_json(manifests_dir / "multilingual_train.meta.json")
    val_meta = _read_json(manifests_dir / "multilingual_val.meta.json")

    assert len(train_rows) == 2
    assert len(val_rows) == 2
    assert {row["ref_wav"] for row in train_rows} == {"../data/hi/clips/a.wav", "../data/es/clips/c.wav"}
    assert {row["target_wav"] for row in val_rows} == {"../data/hi/clips/f.wav", "../data/es/clips/h.wav"}
    assert train_meta["lang_code_totals"] == {"e": 1, "h": 1}
    assert val_meta["lang_code_totals"] == {"e": 1, "h": 1}
    assert train_meta["hours_in_manifest_total"] == 2.5
    assert val_meta["hours_in_manifest_total"] == 0.75


def _write_manifest(path: Path, *rows: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_meta(manifest_path: Path, *, locale_dir: Path, hours_in_manifest: float, split: str) -> None:
    meta_path = manifest_path.parent / f"{manifest_path.stem}.meta.json"
    payload = {
        "locale_dir": str(locale_dir),
        "hours_in_manifest": hours_in_manifest,
        "split": split,
        "fallback_locale_from_dir": locale_dir.name,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
