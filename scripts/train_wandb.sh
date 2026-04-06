#!/usr/bin/env bash
# Voice-clone adapter training with Weights & Biases (WSL).
# Usage: MANIFEST=/path/to/train.jsonl bash scripts/train_wandb.sh
# Optional: MAX_STEPS=75 for a short smoke run (passed as --max-steps).
# Profiling: PROFILE_BREAKDOWN=1 for coarse step timers; TORCH_PROFILER_TRACE=./prof/trace.json for Chrome trace
#   (use MAX_STEPS>=5 with breakdown, MAX_STEPS>=4 with torch profiler default schedule).

set -euo pipefail

# --- edit these ---
CONDA_ENV="${CONDA_ENV:-voice-kokoro}"                                  # your conda env name
MANIFEST="${MANIFEST:-manifests/hi_train.jsonl}"                        # required: JSONL manifest path
MANIFEST_ROOT="${MANIFEST_ROOT:-data/hi}"                               # optional: resolve relative paths in manifest
KOKORO_REPO="${KOKORO_REPO:-hexgrad/Kokoro-82M}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-20}"                                            # optional: e.g. 75 → --max-steps 75 (smoke runs)
DEVICE="${DEVICE:-cuda}"                                                # cuda | cpu | mps
WANDB_PROJECT="${WANDB_PROJECT:-Voice-Clone-Kokoro-TTS}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-hi-train-$(date +%Y%m%d-%H%M%S)}"
PROFILE_BREAKDOWN="${PROFILE_BREAKDOWN:-0}"                             # 1 → --profile-breakdown
PROFILE_BREAKDOWN_STEPS="${PROFILE_BREAKDOWN_STEPS:-3}"                 # steps before printing timing summary
TORCH_PROFILER_TRACE="${TORCH_PROFILER_TRACE:-}"                        # e.g. ./wandb/prof/trace.json → --torch-profiler-trace
# ------------------

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_THIS_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. In WSL, install Miniconda/Anaconda or add conda to PATH." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ -z "${MANIFEST}" ]]; then
  echo "error: MANIFEST is required (path to JSONL manifest)." >&2
  echo "  Example: MANIFEST=manifests/hi_train.jsonl bash scripts/train_wandb.sh" >&2
  echo "  Smoke run: MANIFEST=... MAX_STEPS=75 bash scripts/train_wandb.sh" >&2
  exit 1
fi

CMD=(
  python -m voice_clone.train_adapters
  --manifest "$MANIFEST"
  --kokoro-repo "$KOKORO_REPO"
  --epochs "$EPOCHS"
  --device "$DEVICE"
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$WANDB_RUN_NAME"
  --amp
  --num-workers 8
)
if [[ -n "${MANIFEST_ROOT}" ]]; then
  CMD+=(--manifest-root "$MANIFEST_ROOT")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  CMD+=(--max-steps "$MAX_STEPS")
fi
if [[ "${PROFILE_BREAKDOWN}" == "1" ]]; then
  CMD+=(--profile-breakdown --profile-breakdown-steps "$PROFILE_BREAKDOWN_STEPS")
fi
if [[ -n "${TORCH_PROFILER_TRACE}" ]]; then
  CMD+=(--torch-profiler-trace "$TORCH_PROFILER_TRACE")
fi
"${CMD[@]}" "$@"
