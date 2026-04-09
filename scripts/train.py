"""Voice-clone adapter training launcher with Weights & Biases.

Usage (from repo root):
    python scripts/train.py
    MANIFEST=manifests/other.jsonl python scripts/train.py
    python scripts/train.py --max-steps 75          # smoke run (extra args forwarded)

All knobs are controlled via environment variables with sensible defaults.
Override any of them before invoking the script:

    MANIFEST          manifests/hi_train.jsonl   JSONL manifest path
    MANIFEST_ROOT     data/hi                    resolve relative manifest paths
    CKPT_DIR          ckpt                       periodic checkpoint directory
    RESUME            (empty)                    checkpoint to resume from
    KOKORO_REPO       hexgrad/Kokoro-82M         HF model repo
    EPOCHS            1
    MAX_STEPS         2000                       cap training steps (e.g. 75 for smoke)
    DEVICE            (auto)                     cuda | cpu | mps
    NUM_WORKERS       auto                       dataloader workers (auto = cpu_count or 0 on Windows)
    WANDB_PROJECT     Voice-Clone-Kokoro-TTS
    WANDB_RUN_NAME    hi-train-<timestamp>
    VAL_MANIFEST      manifests/hi_val.jsonl      validation JSONL manifest (skipped if missing)
    VAL_MANIFEST_ROOT data/hi                    resolve relative val manifest paths
    PROFILE_BREAKDOWN       0                    1 to enable coarse step timers
    PROFILE_BREAKDOWN_STEPS 3                    steps before printing timing summary
    TORCH_PROFILER_TRACE    (empty)              path for Chrome trace JSON
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_num_workers() -> int:
    if platform.system() == "Windows":
        return 0
    return os.cpu_count() or 0


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    manifest = os.environ.get("MANIFEST", "manifests/hi_train.jsonl")
    manifest_root = os.environ.get("MANIFEST_ROOT", "data/hi")
    ckpt_dir = os.environ.get("CKPT_DIR", "ckpt")
    resume = os.environ.get("RESUME", "")
    kokoro_repo = os.environ.get("KOKORO_REPO", "hexgrad/Kokoro-82M")
    epochs = os.environ.get("EPOCHS", "1")
    max_steps = os.environ.get("MAX_STEPS", "")
    device = os.environ.get("DEVICE")
    num_workers = os.environ.get("NUM_WORKERS", str(_default_num_workers()))
    wandb_project = os.environ.get("WANDB_PROJECT", "Voice-Clone-Kokoro-TTS")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", f"hi-train-{timestamp}")
    val_manifest = os.environ.get("VAL_MANIFEST", "manifests/hi_val.jsonl")
    val_manifest_root = os.environ.get("VAL_MANIFEST_ROOT", "data/hi")
    if val_manifest and not (REPO_ROOT / val_manifest).is_file():
        val_manifest = ""
        val_manifest_root = ""
    profile_breakdown = os.environ.get("PROFILE_BREAKDOWN", "0")
    profile_breakdown_steps = os.environ.get("PROFILE_BREAKDOWN_STEPS", "3")
    torch_profiler_trace = os.environ.get("TORCH_PROFILER_TRACE", "")

    cmd: list[str] = [
        sys.executable, "-m", "voice_clone.train_adapters",
        "--manifest", manifest,
        "--kokoro-repo", kokoro_repo,
        "--epochs", epochs,
        "--wandb",
        "--wandb-project", wandb_project,
        "--wandb-run-name", wandb_run_name,
        "--amp",
        "--num-workers", num_workers,
    ]

    if device:
        cmd += ["--device", device]
    if manifest_root:
        cmd += ["--manifest-root", manifest_root]
    if val_manifest:
        cmd += ["--val-manifest", val_manifest]
    if val_manifest_root:
        cmd += ["--val-manifest-root", val_manifest_root]
    if ckpt_dir:
        cmd += ["--ckpt-dir", ckpt_dir]
    if resume:
        cmd += ["--resume", resume]
    if max_steps:
        cmd += ["--max-steps", max_steps]
    if profile_breakdown == "1":
        cmd += ["--profile-breakdown", "--profile-breakdown-steps", profile_breakdown_steps]
    if torch_profiler_trace:
        cmd += ["--torch-profiler-trace", torch_profiler_trace]

    cmd += sys.argv[1:]

    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(REPO_ROOT)
    if existing_pythonpath:
        pythonpath = os.pathsep.join([pythonpath, existing_pythonpath])
    env = {**os.environ, "PYTHONPATH": pythonpath}
    sys.exit(subprocess.run(cmd, env=env, cwd=str(REPO_ROOT)).returncode)


if __name__ == "__main__":
    main()
