"""Unit tests for timing and metrics utilities."""

import time
import pytest
import torch

from src.metrics.timer import CudaTimer, WallTimer, TimingResult


class TestWallTimer:
    def test_measures_positive_time(self):
        t = WallTimer()
        t.start()
        time.sleep(0.01)
        result = t.stop()
        assert isinstance(result, TimingResult)
        assert result.elapsed_ms > 0
        assert result.method == "wall_clock"

    def test_method_label(self):
        t = WallTimer()
        t.start()
        result = t.stop()
        assert result.method == "wall_clock"


class TestCudaTimer:
    def test_fallback_to_wall_clock_on_cpu(self):
        t = CudaTimer(torch.device("cpu"))
        t.start()
        time.sleep(0.01)
        result = t.stop()
        assert result.elapsed_ms > 0
        assert result.method == "wall_clock"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_cuda_event_timing(self):
        t = CudaTimer(torch.device("cuda"))
        t.start()
        x = torch.randn(1000, 1000, device="cuda")
        _ = x @ x
        result = t.stop()
        assert result.elapsed_ms > 0
        assert result.method == "cuda_event"
