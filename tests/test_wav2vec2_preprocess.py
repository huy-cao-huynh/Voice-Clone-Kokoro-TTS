"""Parity tests: Hugging Face ``Wav2Vec2FeatureExtractor`` vs ``wav2vec2_preprocess_batch``.

We compare ``input_values`` and ``attention_mask`` for variable-length batches.

**Transformers version:** ``test_transformers_version_is_noted`` emits a ``UserWarning`` with
``transformers==<version>`` so CI logs and ``pytest -W default`` show the exact wheel. When
upgrading ``transformers``, re-run these tests; if they fail, compare HF
``Wav2Vec2FeatureExtractor`` / ``_np_normalize`` and adjust ``voice_clone/wavlm_sv.py``.

Import note: ``wavlm_sv`` is loaded with ``importlib`` and registered in ``sys.modules`` (so
``@dataclass`` resolves ``__module__``) without importing the ``voice_clone`` package (which
pulls ``torchaudio`` via ``dataset``).
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import transformers
from transformers import Wav2Vec2FeatureExtractor

_ROOT = Path(__file__).resolve().parents[1]
_WAVLM_SV_PATH = _ROOT / "voice_clone" / "wavlm_sv.py"
_spec = importlib.util.spec_from_file_location("voice_clone.wavlm_sv", _WAVLM_SV_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_WAVLM_SV_PATH}")
_wavlm_sv = importlib.util.module_from_spec(_spec)
# Required so ``@dataclass`` can resolve ``sys.modules[cls.__module__]`` during import.
sys.modules[str(_spec.name)] = _wavlm_sv
_spec.loader.exec_module(_wavlm_sv)
wav2vec2_preprocess_batch = _wavlm_sv.wav2vec2_preprocess_batch

MODEL_ID = "microsoft/wavlm-base-plus-sv"
SAMPLING_RATE = 16_000


@pytest.fixture(scope="module")
def feature_extractor() -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)


def test_transformers_version_is_noted() -> None:
    """Record installed ``transformers`` in warning output (visible under default pytest)."""
    v = transformers.__version__
    assert v and v[0].isdigit()
    warnings.warn(
        f"[test_wav2vec2_preprocess] transformers=={v}",
        UserWarning,
        stacklevel=1,
    )


def _variable_length_waveforms(
    generator: torch.Generator, batch_size: int = 4
) -> list[torch.Tensor]:
    lengths = [400 + int(torch.randint(100, 2500, (1,), generator=generator).item()) for _ in range(batch_size)]
    return [torch.randn(n, generator=generator, dtype=torch.float32) * 0.1 for n in lengths]


def _hf_batch(fe: Wav2Vec2FeatureExtractor, rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    np_rows = [r.detach().cpu().numpy().astype(np.float32, copy=False) for r in rows]
    enc = fe(
        np_rows,
        padding=True,
        return_tensors="pt",
        sampling_rate=SAMPLING_RATE,
    )
    return enc["input_values"], enc["attention_mask"]


def _feature_extractor_with_normalize() -> Wav2Vec2FeatureExtractor:
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    fe.do_normalize = True
    return fe


@pytest.mark.parametrize("do_normalize", [False, True])
@pytest.mark.parametrize("seed", [0, 42])
def test_wav2vec2_preprocess_matches_hf_variable_lengths(
    feature_extractor: Wav2Vec2FeatureExtractor,
    do_normalize: bool,
    seed: int,
) -> None:
    """HF NumPy path vs torch batch helper; batch size >= 2, unequal lengths."""
    fe = feature_extractor if not do_normalize else _feature_extractor_with_normalize()
    gen = torch.Generator().manual_seed(seed)
    rows = _variable_length_waveforms(gen, batch_size=4)
    assert len({r.shape[0] for r in rows}) > 1, "expected unequal lengths"

    iv_hf, am_hf = _hf_batch(fe, rows)
    iv_t, am_t = wav2vec2_preprocess_batch(
        fe,
        rows,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sampling_rate=SAMPLING_RATE,
    )

    assert am_t.dtype == torch.long
    assert torch.equal(am_hf.to(dtype=torch.long), am_t)

    rtol, atol = (1e-5, 1e-6) if do_normalize else (1e-6, 1e-6)
    torch.testing.assert_close(iv_hf, iv_t, rtol=rtol, atol=atol)


def test_wav2vec2_preprocess_matches_hf_batched_2d_tensor(
    feature_extractor: Wav2Vec2FeatureExtractor,
) -> None:
    """``(batch, samples)`` tensor input uses one shared length (still exercises tensor branch)."""
    fe = feature_extractor
    gen = torch.Generator().manual_seed(7)
    b, t = 3, 900
    w = torch.randn(b, t, generator=gen, dtype=torch.float32) * 0.05
    rows = [w[i] for i in range(b)]

    iv_hf, am_hf = _hf_batch(fe, rows)
    iv_t, am_t = wav2vec2_preprocess_batch(
        fe,
        w,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sampling_rate=SAMPLING_RATE,
    )
    assert am_t.dtype == torch.long
    assert torch.equal(am_hf.to(dtype=torch.long), am_t)
    torch.testing.assert_close(iv_hf, iv_t, rtol=1e-6, atol=1e-6)
