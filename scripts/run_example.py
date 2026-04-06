"""Quick offline inference using a trained checkpoint — no wandb needed.

Usage (from repo root):
    python scripts/run_example.py
    python scripts/run_example.py --checkpoint ckpt/checkpoint_2000.pt --text "your text here"
    python scripts/run_example.py --ref-wav path/to/speaker.mp3 --text "custom text" --lang h
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from voice_clone.infer import KOKORO_OUTPUT_SR, infer_waveform


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run voice-clone inference on a checkpoint with a reference speaker clip.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "ckpt" / "checkpoint_2000.pt",
        help="Path to .pt checkpoint (default: ckpt/checkpoint_2000.pt).",
    )
    p.add_argument(
        "--ref-wav",
        type=Path,
        default=REPO_ROOT / "data" / "hi" / "clips" / "common_voice_hi_23795238.mp3",
        help="Reference speaker audio file.",
    )
    p.add_argument(
        "--text",
        type=str,
        default="भारत एक विविधताओं से भरा देश है।",
        help="Text to synthesize (default: a short Hindi sentence).",
    )
    p.add_argument("--lang", type=str, default="h", help="Kokoro lang code (default: h for Hindi).")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "output" / "example.wav",
        help="Output WAV path (default: output/example.wav).",
    )
    p.add_argument("--device", type=str, default=None, help="cuda | cpu | mps (default: auto).")
    p.add_argument("--speed", type=float, default=None, help="Speed factor (default: from checkpoint config).")
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Reference  : {args.ref_wav}")
    print(f"Text       : {args.text}")
    print(f"Lang       : {args.lang}")
    print(f"Device     : {device}")
    print()

    wav = infer_waveform(
        ckpt_path=args.checkpoint,
        ref_wav_path=args.ref_wav,
        text=args.text,
        lang_code=args.lang,
        device=device,
        speed=args.speed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    wav_np = wav.numpy()
    wav_int16 = np.clip(wav_np * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(args.out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(KOKORO_OUTPUT_SR)
        wf.writeframes(wav_int16.tobytes())

    duration_s = wav.numel() / KOKORO_OUTPUT_SR
    print(f"Saved {args.out}  ({wav.numel()} samples, {duration_s:.2f}s @ {KOKORO_OUTPUT_SR} Hz)")


if __name__ == "__main__":
    main()
