"""Precise timing utilities using CUDA events and wall-clock fallback."""

import time
from dataclasses import dataclass

import torch


@dataclass
class TimingResult:
    elapsed_ms: float
    method: str  # "cuda_event" or "wall_clock"


class CudaTimer:
    """Measures GPU kernel time using torch.cuda.Event (sub-millisecond accuracy).

    Falls back to wall-clock timing when CUDA is unavailable.
    """

    def __init__(self, device: torch.device):
        self.use_cuda = device.type == "cuda" and torch.cuda.is_available()
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._wall_start: float = 0.0

    def start(self) -> None:
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._wall_start = time.perf_counter()

    def stop(self) -> TimingResult:
        if self.use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            elapsed = self._start_event.elapsed_time(self._end_event)
            return TimingResult(elapsed_ms=elapsed, method="cuda_event")
        else:
            elapsed = (time.perf_counter() - self._wall_start) * 1000.0
            return TimingResult(elapsed_ms=elapsed, method="wall_clock")


class WallTimer:
    """Simple wall-clock timer using time.perf_counter()."""

    def __init__(self):
        self._start: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> TimingResult:
        elapsed = (time.perf_counter() - self._start) * 1000.0
        return TimingResult(elapsed_ms=elapsed, method="wall_clock")
