"""MFA alignment helpers for cache-backed duration supervision."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_MFA_LANGUAGES = frozenset({"a", "b", "e", "f", "i", "j", "z"})


def default_alignment_workspace(manifest_path: Path, *, alignments_root: Path = Path("alignments")) -> Path:
    return Path(alignments_root) / Path(manifest_path).stem


def default_alignment_row_path(
    manifest_path: Path,
    row_index: int,
    *,
    alignments_root: Path = Path("alignments"),
) -> Path:
    return default_alignment_workspace(manifest_path, alignments_root=alignments_root) / "rows" / f"{int(row_index)}.pt"


__all__ = ["SUPPORTED_MFA_LANGUAGES", "default_alignment_row_path", "default_alignment_workspace"]
