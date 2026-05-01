from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "scripts" / "build_single_manifest_same_speaker_subset.py"
_spec = importlib.util.spec_from_file_location("build_single_manifest_same_speaker_subset", _SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_SCRIPT_PATH}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[str(_spec.name)] = _mod
_spec.loader.exec_module(_mod)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_build_subset_pair_uses_single_manifest_and_shortest_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    out_train = tmp_path / "memorization_en_6x1_min.phonemes.jsonl"
    out_val = tmp_path / "memorization_en_6x1_min_val.phonemes.jsonl"
    rows = [
        {"ref_wav": "r0.wav", "target_wav": "t0.wav", "text": "zzzz", "lang_code": "a", "phonemes": "p", "speaker_id": "s1"},
        {"ref_wav": "r1.wav", "target_wav": "t1.wav", "text": "a", "lang_code": "a", "phonemes": "p", "speaker_id": "s1"},
        {"ref_wav": "r2.wav", "target_wav": "t2.wav", "text": "bb", "lang_code": "a", "phonemes": "p", "speaker_id": "s1"},
        {"ref_wav": "r3.wav", "target_wav": "t3.wav", "text": "cccc", "lang_code": "b", "phonemes": "p", "speaker_id": "s2"},
        {"ref_wav": "r4.wav", "target_wav": "t4.wav", "text": "c", "lang_code": "b", "phonemes": "p", "speaker_id": "s2"},
        {"ref_wav": "r5.wav", "target_wav": "t5.wav", "text": "dd", "lang_code": "b", "phonemes": "p", "speaker_id": "s2"},
        {"ref_wav": "r6.wav", "target_wav": "t6.wav", "text": "skip", "lang_code": "e", "phonemes": "p", "speaker_id": "s3"},
        {"ref_wav": "r7.wav", "target_wav": "t7.wav", "text": "skip2", "lang_code": "e", "phonemes": "p", "speaker_id": "s3"},
    ]
    _write_manifest(source, rows)

    meta = _mod.build_subset_pair(
        source_manifest=source,
        output_train_manifest=out_train,
        output_val_manifest=out_val,
        speaker_count=2,
        allowed_lang_codes={"a", "b"},
    )

    train_selected = [json.loads(line) for line in out_train.read_text(encoding="utf-8").splitlines()]
    val_selected = [json.loads(line) for line in out_val.read_text(encoding="utf-8").splitlines()]
    assert [row["speaker_id"] for row in train_selected] == ["s1", "s2"]
    assert [row["speaker_id"] for row in val_selected] == ["s1", "s2"]
    assert [row["text"] for row in train_selected] == ["a", "c"]
    assert [row["text"] for row in val_selected] == ["bb", "dd"]
    assert meta["selected_speaker_ids"] == ["s1", "s2"]
    assert out_train.with_suffix(".meta.json").is_file()
    assert out_val.with_suffix(".meta.json").is_file()


def test_build_subset_pair_requires_two_rows_per_speaker(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_manifest(
        source,
        [
            {"ref_wav": "r0.wav", "target_wav": "t0.wav", "text": "a", "lang_code": "a", "phonemes": "p", "speaker_id": "s1"},
            {"ref_wav": "r1.wav", "target_wav": "t1.wav", "text": "b", "lang_code": "a", "phonemes": "p", "speaker_id": "s2"},
        ],
    )
    with pytest.raises(ValueError, match="at least 2 rows"):
        _mod.build_subset_pair(
            source_manifest=source,
            output_train_manifest=tmp_path / "out_train.jsonl",
            output_val_manifest=tmp_path / "out_val.jsonl",
            speaker_count=1,
            allowed_lang_codes={"a"},
        )
