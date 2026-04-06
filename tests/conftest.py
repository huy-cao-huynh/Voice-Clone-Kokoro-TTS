"""Shared fixtures and pytest configuration for the voice-clone test suite."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import wave
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "slow: integration tests that download HF models or build the full stack")


# ---------------------------------------------------------------------------
# Lightweight module loaders (avoid pulling the full voice_clone package)
# ---------------------------------------------------------------------------

def _load_module(name: str, rel_path: str):
    """Load a single .py file into sys.modules without triggering voice_clone.__init__."""
    path = _ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def adapters_mod():
    return _load_module("voice_clone.adapters", "voice_clone/adapters.py")


@pytest.fixture(scope="session")
def segment_gst_mod():
    return _load_module("voice_clone.segment_gst", "voice_clone/segment_gst.py")


@pytest.fixture(scope="session")
def losses_mod():
    return _load_module("voice_clone.losses", "voice_clone/losses.py")


@pytest.fixture(scope="session")
def config_mod():
    return _load_module("voice_clone.config", "voice_clone/config.py")


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

def write_wav(path: Path, samples: int, sr: int = 16_000, channels: int = 1) -> None:
    """Write a tiny valid WAV file with silent (zero) PCM-16 data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * samples * channels)


def write_sine_wav(path: Path, samples: int, sr: int = 16_000, freq: float = 440.0) -> None:
    """Write a WAV with a sine tone so mel spectrogram is not degenerate."""
    import math
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = []
        for i in range(samples):
            v = int(16000 * math.sin(2 * math.pi * freq * i / sr))
            frames.append(struct.pack("<h", max(-32768, min(32767, v))))
        wf.writeframes(b"".join(frames))


def make_manifest_and_wavs(
    tmp_path: Path,
    *,
    num_rows: int = 2,
    ref_sr: int = 16_000,
    tgt_sr: int = 24_000,
    samples: int = 4800,
) -> Dict[str, Any]:
    """Create a tiny JSONL manifest and matching WAV files. Returns dict with paths."""
    clips = tmp_path / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for i in range(num_rows):
        ref_name = f"ref_{i}.wav"
        tgt_name = f"tgt_{i}.wav"
        write_sine_wav(clips / ref_name, samples, sr=ref_sr)
        write_sine_wav(clips / tgt_name, samples, sr=tgt_sr)
        row = {
            "ref_wav": f"clips/{ref_name}",
            "target_wav": f"clips/{tgt_name}",
            "text": "hello world",
            "lang_code": "a",
            "phonemes": "hɛˈloʊ wˈɜːld",
        }
        rows.append(row)
    with open(manifest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"manifest": manifest, "root": tmp_path, "rows": rows, "clips": clips}
