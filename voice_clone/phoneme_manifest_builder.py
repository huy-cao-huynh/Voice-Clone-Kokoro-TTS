"""Add precomputed Kokoro phoneme strings to an existing manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from kokoro.pipeline import KPipeline

from .dataset import normalize_lang_code, text_to_phonemes


def build_phoneme_manifest(
    manifest_path: Path,
    *,
    output_path: Path,
    kokoro_repo_id: str,
    strict_single_chunk: bool = True,
    skip_existing_phonemes: bool = True,
) -> None:
    pipelines: Dict[str, KPipeline] = {}

    def pipeline_for_lang(lang_code: str) -> KPipeline:
        lang = normalize_lang_code(lang_code)
        if lang not in pipelines:
            pipelines[lang] = KPipeline(lang_code=lang, repo_id=kokoro_repo_id, model=False)
        return pipelines[lang]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if skip_existing_phonemes and row.get("phonemes"):
                phonemes = str(row["phonemes"])
            else:
                try:
                    phonemes = text_to_phonemes(
                        pipeline_for_lang(str(row["lang_code"])),
                        str(row["text"]),
                        strict_single_chunk=strict_single_chunk,
                    )
                except Exception as exc:
                    text = str(row.get("text", ""))[:120].replace("\n", " ")
                    raise ValueError(
                        f"{manifest_path}:{idx}: failed to phonemize lang_code={row.get('lang_code')!r} text={text!r}"
                    ) from exc
            row["phonemes"] = phonemes
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add `phonemes` to a voice-clone manifest.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kokoro-repo", type=str, default="hexgrad/Kokoro-82M")
    p.add_argument("--strict-single-chunk", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_phoneme_manifest(
        args.manifest,
        output_path=args.output,
        kokoro_repo_id=args.kokoro_repo,
        strict_single_chunk=args.strict_single_chunk,
    )


if __name__ == "__main__":
    main()
