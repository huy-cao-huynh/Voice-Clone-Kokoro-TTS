"""Voice-clone adapter training launcher with Weights & Biases.

Usage (from repo root):
    python scripts/train.py
    MANIFEST=manifests/other.jsonl python scripts/train.py
    python scripts/train.py --max-steps 75          # smoke run (extra args forwarded)
    MANIFEST=manifests/memorize_train.jsonl VAL_MANIFEST=manifests/memorize_val.jsonl EPOCHS=200 CHECKPOINT_INTERVAL=25 python scripts/train.py

All knobs are controlled via environment variables with sensible defaults.
Override any of them before invoking the script:

    MANIFEST          manifests/multilingual_train.jsonl   JSONL manifest path
    MANIFEST_ROOT     (empty)                             optional root for relative manifest paths;
                                                         omit for merged multilingual manifests
    CKPT_DIR          ckpt                       periodic checkpoint directory
    RESUME            (empty)                    checkpoint to resume from
    KOKORO_REPO       hexgrad/Kokoro-82M         HF model repo
    EPOCHS            1
    MAX_STEPS         2000                       cap training steps (e.g. 75 for smoke)
    DEVICE            (auto)                     cuda | cpu | mps
    NUM_WORKERS       auto                       dataloader workers (auto = cpu_count or 0 on Windows)
    WANDB_PROJECT     Voice-Clone-Kokoro-TTS
    WANDB_RUN_NAME    multilingual-train-<timestamp>
    VAL_MANIFEST      manifests/multilingual_val.jsonl    validation JSONL manifest (skipped if missing)
    VAL_MANIFEST_ROOT (empty)                             optional root for relative val manifest paths
    SAVE_FINAL_CHECKPOINT   (unset)                1 to force a final off-interval checkpoint, 0 for interval-only
    BATCH_SIZE              (from TrainConfig)     batch size per micro-step
    GRAD_ACCUM_STEPS        (from TrainConfig)     micro-steps per optimizer step
    WARMUP_STEPS            (unset)                LR warmup steps; omit to use TrainConfig default
    DISC_START_STEP         (from TrainConfig)     step to activate discriminator
    CHECKPOINT_INTERVAL     (from TrainConfig)     save checkpoint every N steps
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

    manifest = os.environ.get("MANIFEST", "manifests/memorization_en_1024.phonemes.jsonl")
    manifest_root = os.environ.get("MANIFEST_ROOT", "")
    ckpt_dir = os.environ.get("CKPT_DIR", "ckpt/memorization_en_1024")
    resume = os.environ.get("RESUME", "")
    kokoro_repo = os.environ.get("KOKORO_REPO", "hexgrad/Kokoro-82M")
    epochs = os.environ.get("EPOCHS", "300")
    max_steps = os.environ.get("MAX_STEPS", "")
    device = os.environ.get("DEVICE")
    num_workers = os.environ.get("NUM_WORKERS", str(_default_num_workers()))
    wandb_project = os.environ.get("WANDB_PROJECT", "Voice-Clone-Kokoro-TTS")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", f"memorization_en_1024-train-{timestamp}")
    val_manifest = os.environ.get("VAL_MANIFEST", "manifests/memorization_en_val.phonemes.jsonl")
    val_manifest_root = os.environ.get("VAL_MANIFEST_ROOT", "")
    if val_manifest and not (REPO_ROOT / val_manifest).is_file():
        val_manifest = ""
        val_manifest_root = ""
    batch_size = os.environ.get("BATCH_SIZE", "4")
    grad_accum_steps = os.environ.get("GRAD_ACCUM_STEPS", "")
    warmup_steps = os.environ.get("WARMUP_STEPS", "")
    disc_start_step = os.environ.get("DISC_START_STEP", "")
    checkpoint_interval = os.environ.get("CHECKPOINT_INTERVAL", "")
    save_final_checkpoint = os.environ.get("SAVE_FINAL_CHECKPOINT", "0")
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
    if batch_size:
        cmd += ["--batch-size", batch_size]
    if grad_accum_steps:
        cmd += ["--grad-accum-steps", grad_accum_steps]
    if warmup_steps:
        cmd += ["--warmup-steps", warmup_steps]
    if disc_start_step:
        cmd += ["--disc-start-step", disc_start_step]
    if checkpoint_interval:
        cmd += ["--checkpoint-interval", checkpoint_interval]
    if save_final_checkpoint == "1":
        cmd += ["--save-final-checkpoint"]
    elif save_final_checkpoint == "0":
        cmd += ["--no-save-final-checkpoint"]

    cmd += sys.argv[1:]

    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(REPO_ROOT)
    if existing_pythonpath:
        pythonpath = os.pathsep.join([pythonpath, existing_pythonpath])
    env = {**os.environ, "PYTHONPATH": pythonpath}
    sys.exit(subprocess.run(cmd, env=env, cwd=str(REPO_ROOT)).returncode)


if __name__ == "__main__":
    main()
