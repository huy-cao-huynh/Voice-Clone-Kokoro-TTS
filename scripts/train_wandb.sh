#!/usr/bin/env bash
# Voice-clone adapter training with Weights & Biases (WSL).
# Usage: bash scripts/train_wandb.sh
#        MANIFEST=/path/to/train.jsonl bash scripts/train_wandb.sh

set -euo pipefail

# --- edit these ---
CONDA_ENV="${CONDA_ENV:-voice-kokoro}"   # your conda env name
MANIFEST="${MANIFEST:-}"                # JSONL path, or leave empty for dummy smoke test
MANIFEST_ROOT="${MANIFEST_ROOT:-}"      # optional: resolve relative paths in manifest
KOKORO_REPO="${KOKORO_REPO:-hexgrad/Kokoro-82M}"
EPOCHS="${EPOCHS:-5}"
DEVICE="${DEVICE:-cuda}"                # cuda | cpu | mps
WANDB_PROJECT="${WANDB_PROJECT:-Voice-Clone-Kokoro-TTS}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"
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
  echo "MANIFEST unset: running dummy-data smoke test with wandb."
  python -m voice_clone.train_adapters \
    --dummy-steps 20 \
    --device "$DEVICE" \
    --kokoro-repo "$KOKORO_REPO" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    "$@"
else
  CMD=(
    python -m voice_clone.train_adapters
    --manifest "$MANIFEST"
    --kokoro-repo "$KOKORO_REPO"
    --epochs "$EPOCHS"
    --device "$DEVICE"
    --wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$WANDB_RUN_NAME"
  )
  if [[ -n "${MANIFEST_ROOT}" ]]; then
    CMD+=(--manifest-root "$MANIFEST_ROOT")
  fi
  "${CMD[@]}" "$@"
fi
