"""Partial benchmarking — short controlled runs with convergence detection.

Instead of running the full iteration matrix (100 iters x 3 repeats x all
batch sizes), the partial profiler runs a small number of iterations and
stops as soon as throughput stabilises.  This cuts cloud spend by 5-10x
while still producing reliable estimates with confidence intervals.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from ..benchmark_config import resolve_workload_specs
from ..workloads import get_workload
from ..metrics.timer import CudaTimer
from ..metrics.gpu_collector import GpuCollector
from ..reproducibility.seed_manager import set_deterministic

logger = logging.getLogger(__name__)


@dataclass
class PartialResult:
    """Output of a single partial-benchmark run."""
    workload: str
    mode: str
    batch_size: int
    gpu_type: str
    device: str

    estimated_throughput: float
    confidence_low: float
    confidence_high: float
    converged: bool
    iterations_run: int
    convergence_cv: float

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    avg_gpu_util_pct: float = 0.0
    avg_gpu_mem_mb: float = 0.0
    wall_time_sec: float = 0.0
    throughput_unit: str = ""
    param_count: int = 0

    raw_throughputs: list[float] = field(default_factory=list)


class PartialProfiler:
    """Run short benchmark bursts with on-the-fly convergence detection."""

    def __init__(
        self,
        max_iterations: int = 30,
        warmup_iterations: int = 5,
        convergence_window: int = 8,
        convergence_cv_threshold: float = 0.05,
        time_budget_seconds: float = 300,
    ):
        self.max_iterations = max_iterations
        self.warmup_iterations = warmup_iterations
        self.convergence_window = convergence_window
        self.convergence_cv_threshold = convergence_cv_threshold
        self.time_budget_seconds = time_budget_seconds

    @classmethod
    def from_config(cls, config_path: str | Path) -> "PartialProfiler":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        pcfg = cfg.get("partial_benchmark", {})
        return cls(
            max_iterations=pcfg.get("max_iterations", 30),
            warmup_iterations=pcfg.get("warmup_iterations", 5),
            convergence_window=pcfg.get("convergence_window", 8),
            convergence_cv_threshold=pcfg.get("convergence_cv_threshold", 0.05),
            time_budget_seconds=pcfg.get("time_budget_seconds", 300),
        )

    def _check_convergence(self, throughputs: list[float]) -> tuple[bool, float]:
        if len(throughputs) < self.convergence_window:
            return False, float("inf")
        window = np.array(throughputs[-self.convergence_window :])
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        cv = std / mean if mean > 0 else float("inf")
        return cv <= self.convergence_cv_threshold, cv

    def _estimate(
        self, throughputs: list[float]
    ) -> tuple[float, float, float, float]:
        """Return (mean, ci_low, ci_high, cv) from the stable window."""
        if len(throughputs) >= self.convergence_window:
            window = np.array(throughputs[-self.convergence_window :])
        else:
            window = np.array(throughputs)
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
        n = len(window)
        margin = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        cv = std / mean if mean > 0 else 0.0
        return mean, float(mean - margin), float(mean + margin), cv

    def _detect_gpu_type(self) -> str:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            for tag in ("H100", "A100", "A10G", "V100", "T4", "L4", "L40", "A6000"):
                if tag in name:
                    return tag
            return name.replace(" ", "_")
        return "CPU"

    def run(
        self,
        workload_name: str,
        batch_size: int,
        mode: str = "inference",
        device: str | None = None,
        seed: int = 42,
    ) -> PartialResult:
        """Execute a partial benchmark with early stopping on convergence."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        gpu_type = self._detect_gpu_type()
        set_deterministic(seed)

        workload = get_workload(workload_name, batch_size=batch_size, device=device, mode=mode)
        workload.setup()
        meta = workload.get_metadata()
        samples_per_batch = workload.samples_per_batch()

        logger.info(
            "Partial benchmark: %s | bs=%d | mode=%s | device=%s | max_iters=%d | budget=%ds",
            meta.model_name, batch_size, mode, device,
            self.max_iterations, self.time_budget_seconds,
        )

        workload.warmup(self.warmup_iterations)

        collector = GpuCollector(interval_sec=0.5)
        if device != "cpu":
            collector.start()

        timer = CudaTimer(workload.device)
        batch = workload.generate_batch()

        latencies: list[float] = []
        throughputs: list[float] = []
        wall_start = time.perf_counter()
        converged = False
        final_cv = float("inf")

        for i in range(self.max_iterations):
            elapsed_wall = time.perf_counter() - wall_start
            if elapsed_wall >= self.time_budget_seconds:
                logger.info("Time budget exhausted after %d iterations (%.1fs)", i, elapsed_wall)
                break

            timer.start()
            workload.run_iteration(batch)
            result = timer.stop()
            latencies.append(result.elapsed_ms)

            iter_throughput = samples_per_batch / (result.elapsed_ms / 1000.0) if result.elapsed_ms > 0 else 0.0
            throughputs.append(iter_throughput)

            converged, final_cv = self._check_convergence(throughputs)
            if converged and i >= self.warmup_iterations + self.convergence_window:
                logger.info(
                    "Converged at iteration %d (CV=%.4f <= %.4f)",
                    i, final_cv, self.convergence_cv_threshold,
                )
                break

        gpu_snapshots = collector.stop() if device != "cpu" else []
        wall_time = time.perf_counter() - wall_start

        est_mean, ci_low, ci_high, final_cv = self._estimate(throughputs)

        lat_arr = np.array(latencies)
        avg_util = float(np.mean([s.utilization_pct for s in gpu_snapshots])) if gpu_snapshots else 0.0
        avg_mem = float(np.mean([s.memory_used_mb for s in gpu_snapshots])) if gpu_snapshots else 0.0

        result = PartialResult(
            workload=meta.name,
            mode=mode,
            batch_size=batch_size,
            gpu_type=gpu_type,
            device=device,
            estimated_throughput=round(est_mean, 2),
            confidence_low=round(ci_low, 2),
            confidence_high=round(ci_high, 2),
            converged=converged,
            iterations_run=len(latencies),
            convergence_cv=round(final_cv, 6),
            latency_p50_ms=round(float(np.percentile(lat_arr, 50)), 4),
            latency_p95_ms=round(float(np.percentile(lat_arr, 95)), 4),
            latency_p99_ms=round(float(np.percentile(lat_arr, 99)), 4),
            avg_gpu_util_pct=round(avg_util, 2),
            avg_gpu_mem_mb=round(avg_mem, 2),
            wall_time_sec=round(wall_time, 2),
            throughput_unit=meta.throughput_unit,
            param_count=meta.param_count,
            raw_throughputs=throughputs,
        )

        workload.cleanup()
        logger.info(
            "Partial result: %s %.1f %s [%.1f, %.1f] (converged=%s, %d iters, %.1fs)",
            workload_name, est_mean, meta.throughput_unit,
            ci_low, ci_high, converged, len(latencies), wall_time,
        )
        return result

    def run_suite(
        self,
        config_path: str | Path,
        device: str | None = None,
    ) -> list[PartialResult]:
        """Run partial benchmarks for all workloads/modes/batch_sizes in config."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        rec_cfg = {}
        rec_config_path = Path(config_path).parent / "recommendation_config.yaml"
        if rec_config_path.exists():
            with open(rec_config_path) as f:
                rec_cfg = yaml.safe_load(f).get("partial_benchmark", {})

        workload_specs = resolve_workload_specs(
            cfg.get("workloads", ["resnet50"]),
            default_batch_sizes=rec_cfg.get("batch_sizes", cfg.get("batch_sizes", [1, 32])),
            default_modes=rec_cfg.get("modes", cfg.get("modes", ["inference"])),
        )
        seed = cfg.get("seed", 42)

        results: list[PartialResult] = []
        for workload_spec in workload_specs:
            for m in workload_spec.modes:
                for bs in workload_spec.batch_sizes:
                    try:
                        r = self.run(workload_spec.name, bs, m, device=device, seed=seed)
                        results.append(r)
                    except Exception as e:
                        logger.error("Partial benchmark failed: %s/%s/bs%d: %s", workload_spec.name, m, bs, e)
        return results
