"""Unit tests for training step profiling helpers (no full Kokoro/WavLM stack)."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_TP_PATH = _ROOT / "voice_clone" / "train_profiling.py"
_spec = importlib.util.spec_from_file_location("voice_clone.train_profiling", _TP_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_TP_PATH}")
_train_profiling = importlib.util.module_from_spec(_spec)
sys.modules[str(_spec.name)] = _train_profiling
_spec.loader.exec_module(_train_profiling)
BreakdownAggregator = _train_profiling.BreakdownAggregator
StepStopwatch = _train_profiling.StepStopwatch


def test_step_stopwatch_cpu_records_intervals() -> None:
    sw = StepStopwatch(torch.device("cpu"))
    sw.begin_step()
    sw.start("dataloader")
    time.sleep(0.01)
    sw.end("dataloader", sync_cuda=False)
    sw.start("h2d")
    sw.end("h2d", sync_cuda=False)
    row = sw.finish_step()
    assert row["dataloader"] >= 0.005
    assert row["h2d"] >= 0.0


def test_step_stopwatch_rejects_nested_start() -> None:
    sw = StepStopwatch(torch.device("cpu"))
    sw.start("a")
    with pytest.raises(RuntimeError, match="already timing"):
        sw.start("b")
    sw.discard_active()


def test_step_stopwatch_span_context() -> None:
    sw = StepStopwatch(torch.device("cpu"))
    sw.begin_step()
    with sw.span("x", sync_cuda=False):
        time.sleep(0.005)
    row = sw.finish_step()
    assert row["x"] >= 0.002


def test_step_stopwatch_span_discards_on_exception() -> None:
    sw = StepStopwatch(torch.device("cpu"))
    sw.begin_step()
    with pytest.raises(ValueError):
        with sw.span("bad", sync_cuda=False):
            raise ValueError("x")
    assert sw._active is None
    row = sw.finish_step()
    assert "bad" not in row


def test_step_stopwatch_cuda_sync_on_end() -> None:
    with patch.object(_train_profiling.torch.cuda, "is_available", return_value=True):
        with patch.object(_train_profiling.torch.cuda, "synchronize") as mock_sync:
            sw = StepStopwatch(torch.device("cuda"))
            sw.begin_step()
            sw.start("g")
            sw.end("g", sync_cuda=True)
            mock_sync.assert_called_once()


def test_step_stopwatch_no_sync_when_flag_false() -> None:
    with patch.object(_train_profiling.torch.cuda, "is_available", return_value=True):
        with patch.object(_train_profiling.torch.cuda, "synchronize") as mock_sync:
            sw = StepStopwatch(torch.device("cuda"))
            sw.begin_step()
            sw.start("g")
            sw.end("g", sync_cuda=False)
            mock_sync.assert_not_called()


def test_breakdown_aggregator_len_and_summary_skip_warmup() -> None:
    agg = BreakdownAggregator()
    agg.add_step({"a": 10.0, "b": 1.0})
    agg.add_step({"a": 2.0, "b": 3.0})
    agg.add_step({"a": 4.0, "b": 5.0})
    assert len(agg) == 3
    lines = agg.summary_lines(skip_first_step=True, labels=("a", "b"))
    text = "\n".join(lines)
    assert "excluded step 1 as warmup" in text
    # mean of rows 2–3: a=(2+4)/2=3s -> 3000 ms, b=(3+5)/2=4s -> 4000 ms
    assert "3000.00" in text
    assert "4000.00" in text


def test_breakdown_aggregator_single_row_no_skip() -> None:
    agg = BreakdownAggregator()
    agg.add_step({"a": 1.0})
    lines = agg.summary_lines(skip_first_step=True, labels=("a",))
    assert len(lines) >= 2


def test_dataloader_stopiteration_discard() -> None:
    sw = StepStopwatch(torch.device("cpu"))
    sw.start("dataloader")
    sw.discard_active()
    assert sw._active is None
