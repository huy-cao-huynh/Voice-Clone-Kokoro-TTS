#!/usr/bin/env python3
"""Quick dependency check for training/inference (run from repo root in WSL/Linux)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ok = True
    for name in (
        "torch",
        "torchaudio",
        "soundfile",
        "transformers",
        "huggingface_hub",
        "numpy",
    ):
        try:
            m = importlib.import_module(name)
            ver = getattr(m, "__version__", "?")
            print(f"OK  {name:16} {ver}")
        except Exception as e:
            ok = False
            print(f"FAIL {name:16} {e}")

    for name in ("misaki", "loguru"):
        try:
            importlib.import_module(name)
            print(f"OK  {name:16} (Kokoro G2P / logging)")
        except Exception as e:
            ok = False
            print(f"FAIL {name:16} {e}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "kokoro"))
    try:
        import kokoro  # noqa: F401

        print("OK  kokoro          (on PYTHONPATH or editable install)")
    except Exception as e:
        ok = False
        print(f"FAIL kokoro          {e}")

    try:
        from voice_clone.train_adapters import build_models  # noqa: F401

        print("OK  voice_clone.train_adapters")
    except Exception as e:
        ok = False
        print(f"FAIL voice_clone     {e}")

    if not ok:
        print("\nInstall steps: see docs/environment.md", file=sys.stderr)
        return 1
    t = sys.modules.get("torch")
    if t is not None:
        print(f"\nTorch CUDA available: {t.cuda.is_available()}")
        if t.cuda.is_available():
            print(f"Device: {t.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
