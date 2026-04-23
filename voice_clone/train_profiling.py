"""Optional coarse step timing and torch.profiler Chrome traces for adapter training."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

# Order used for printed tables and tests.
# Keep aligned with timer span names emitted in train_adapters.py.
BREAKDOWN_LABELS = (
    "dataloader",
    "h2d",
    "mhubert_ref",
    "sv_ref",
    "kokoro_fwd",
    "sv_gen",
    "disc",
    "gen_backward",
    "wandb_log",
)


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


class StepStopwatch:
    """Named intervals with ``perf_counter``; optional CUDA sync at interval end for GPU spans."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._active: Optional[str] = None
        self._t0: float = 0.0
        self._current_step: Dict[str, float] = {}

    def start(self, name: str) -> None:
        if self._active is not None:
            raise RuntimeError(f"StepStopwatch: already timing {self._active!r}, cannot start {name!r}")
        import time

        self._active = name
        self._t0 = time.perf_counter()

    def end(self, name: str, *, sync_cuda: bool = False) -> None:
        if self._active != name:
            raise RuntimeError(f"StepStopwatch: end({name!r}) but active is {self._active!r}")
        import time

        if sync_cuda:
            _cuda_sync_if_needed(self.device)
        dt = time.perf_counter() - self._t0
        self._active = None
        self._t0 = 0.0
        self._current_step[name] = self._current_step.get(name, 0.0) + dt

    def discard_active(self) -> None:
        """Clear an unfinished interval (e.g. ``StopIteration`` on dataloader)."""
        self._active = None
        self._t0 = 0.0

    def begin_step(self) -> None:
        self._current_step = {}

    def finish_step(self) -> Dict[str, float]:
        out = dict(self._current_step)
        self._current_step = {}
        return out

    @contextmanager
    def span(self, name: str, *, sync_cuda: bool = False) -> Iterator[None]:
        self.start(name)
        try:
            yield
        except BaseException:
            self.discard_active()
            raise
        self.end(name, sync_cuda=sync_cuda)


class BreakdownAggregator:
    """Collect per-step timings and print mean/std (optionally excluding step 1 as warmup)."""

    def __init__(self) -> None:
        self._rows: List[Dict[str, float]] = []

    def add_step(self, timings: Dict[str, float]) -> None:
        self._rows.append(dict(timings))

    def __len__(self) -> int:
        return len(self._rows)

    def summary_lines(
        self,
        *,
        skip_first_step: bool = True,
        labels: tuple[str, ...] = BREAKDOWN_LABELS,
    ) -> List[str]:
        if not self._rows:
            return ["[profile-breakdown] no steps recorded."]
        rows = self._rows[1:] if skip_first_step and len(self._rows) > 1 else list(self._rows)
        if not rows:
            rows = self._rows
        lines = [
            f"[profile-breakdown] steps in aggregate: {len(rows)}"
            + (" (excluded step 1 as warmup)" if skip_first_step and len(self._rows) > 1 else "")
        ]
        for lab in labels:
            vals = [r.get(lab, 0.0) for r in rows]
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            lines.append(f"  {lab:16s}  mean={mean * 1000:8.2f} ms  std={std * 1000:8.2f} ms")
        total_mean = sum(sum(r.get(l, 0.0) for l in labels) for r in rows) / len(rows)
        lines.append(f"  {'total(labeled)':16s}  mean={total_mean * 1000:8.2f} ms")
        return lines

    def print_summary(self, *, skip_first_step: bool = True, labels: tuple[str, ...] = BREAKDOWN_LABELS) -> None:
        for line in self.summary_lines(skip_first_step=skip_first_step, labels=labels):
            print(line)


def build_torch_profiler(
    trace_path: Path,
    device: torch.device,
    *,
    skip_first: int = 1,
    wait: int = 0,
    warmup: int = 1,
    active: int = 2,
    repeat: int = 1,
) -> Any:
    """Create a ``torch.profiler.profile`` configured for Chrome trace export."""
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    activities: List[torch.profiler.ProfilerActivity] = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        skip_first=skip_first,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )

    def _on_trace_ready(prof: torch.profiler.profile) -> None:
        prof.export_chrome_trace(str(trace_path))

    return torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=_on_trace_ready,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    )
