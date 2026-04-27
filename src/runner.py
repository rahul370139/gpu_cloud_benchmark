"""Main benchmark runner — orchestrates workload execution and metric collection."""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .benchmark_config import resolve_workload_specs
from .workloads import get_workload, register_custom_workloads
from .metrics.timer import CudaTimer
from .metrics.gpu_collector import GpuCollector
from .metrics.prometheus_exporter import init_prometheus, push_benchmark_metrics
from .reproducibility.seed_manager import set_deterministic
from .reproducibility.env_capture import capture_environment
from .reproducibility.checksum import checksum_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _detect_gpu_type() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        for tag in ("H100", "A100", "A10G", "V100", "T4", "L4", "L40", "A6000"):
            if tag in name:
                return tag
        return name.replace(" ", "_")
    return "CPU"


def _latency_percentiles(latencies: list[float]) -> dict[str, float]:
    arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
    }


def run_single_benchmark(
    workload_name: str,
    batch_size: int,
    mode: str,
    device: str,
    warmup_iters: int,
    benchmark_iters: int,
    seed: int,
    gpu_poll_interval: float = 0.5,
) -> dict:
    """Run one (workload, batch_size, mode) experiment. Returns a results dict."""

    set_deterministic(seed)

    workload = get_workload(workload_name, batch_size=batch_size, device=device, mode=mode)
    workload.setup()
    meta = workload.get_metadata()

    logger.info(
        "Benchmarking %s | batch=%d | mode=%s | device=%s | iters=%d",
        meta.model_name, batch_size, mode, device, benchmark_iters,
    )

    workload.warmup(warmup_iters)

    collector = GpuCollector(interval_sec=gpu_poll_interval)
    if device != "cpu":
        collector.start()

    timer = CudaTimer(workload.device)
    latencies: list[float] = []

    batch = workload.generate_batch()
    for i in range(benchmark_iters):
        timer.start()
        workload.run_iteration(batch)
        result = timer.stop()
        latencies.append(result.elapsed_ms)

    gpu_snapshots = collector.stop() if device != "cpu" else []

    total_time_sec = sum(latencies) / 1000.0
    samples_processed = workload.samples_per_batch() * benchmark_iters
    throughput = samples_processed / total_time_sec if total_time_sec > 0 else 0.0

    pctiles = _latency_percentiles(latencies)

    avg_util = 0.0
    avg_mem = 0.0
    if gpu_snapshots:
        avg_util = np.mean([s.utilization_pct for s in gpu_snapshots])
        avg_mem = np.mean([s.memory_used_mb for s in gpu_snapshots])

    result = {
        "workload": meta.name,
        "model_name": meta.model_name,
        "param_count": meta.param_count,
        "mode": mode,
        "batch_size": batch_size,
        "input_shape": str(meta.input_shape),
        "throughput_unit": meta.throughput_unit,
        "device": device,
        "benchmark_iters": benchmark_iters,
        "warmup_iters": warmup_iters,
        "total_time_sec": round(total_time_sec, 4),
        "throughput": round(throughput, 2),
        "latency_p50_ms": round(pctiles["p50_ms"], 4),
        "latency_p95_ms": round(pctiles["p95_ms"], 4),
        "latency_p99_ms": round(pctiles["p99_ms"], 4),
        "latency_mean_ms": round(pctiles["mean_ms"], 4),
        "latency_std_ms": round(pctiles["std_ms"], 4),
        "avg_gpu_utilization_pct": round(avg_util, 2),
        "avg_gpu_memory_used_mb": round(avg_mem, 2),
        "seed": seed,
        "timing_method": "cuda_event" if device != "cpu" else "wall_clock",
    }

    workload.cleanup()
    return result, latencies, gpu_snapshots


def _register_cli_workload_target(
    workload_target: str | None,
    workload_name: str | None,
    config: dict,
) -> None:
    if not workload_target:
        return
    alias = workload_name or workload_target.rsplit(":", 1)[-1]
    custom = dict(config.get("custom_workloads") or {})
    custom[alias] = workload_target
    config["custom_workloads"] = custom
    config["workloads"] = [alias]


def run_full_benchmark(
    config_path: str,
    device: str | None = None,
    workload_target: str | None = None,
    workload_name: str | None = None,
) -> Path:
    """Execute the complete benchmark suite defined in a config YAML."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _register_cli_workload_target(workload_target, workload_name, config)

    pushgateway_override = os.environ.get("BENCHMARK_PROMETHEUS_PUSHGATEWAY")
    if pushgateway_override:
        config["prometheus_pushgateway"] = pushgateway_override

    output_dir = Path(os.environ.get("BENCHMARK_RESULTS_DIR", config.get("output_dir", "results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_type = _detect_gpu_type()
    logger.info("Detected GPU type: %s (device=%s)", gpu_type, device)

    pushgateway = config.get("prometheus_pushgateway", "")
    prom_enabled = init_prometheus(pushgateway)

    env_info = capture_environment()
    seed_base = config.get("seed", 42)

    register_custom_workloads(config.get("custom_workloads"))
    workload_specs = resolve_workload_specs(
        config.get("workloads", ["resnet50"]),
        default_batch_sizes=config.get("batch_sizes", [1, 8, 32]),
        default_modes=config.get("modes", ["inference"]),
    )
    num_repeats = config.get("num_repeats", 3)
    warmup_iters = config.get("warmup_iters", 10)
    benchmark_iters = config.get("benchmark_iters", 100)

    all_results: list[dict] = []
    summary_csv = output_dir / f"benchmark_summary_{gpu_type}.csv"

    for workload_spec in workload_specs:
        for mode in workload_spec.modes:
            for batch_size in workload_spec.batch_sizes:
                for repeat in range(1, num_repeats + 1):
                    run_seed = seed_base + repeat - 1
                    logger.info(
                        "=== Run: %s / %s / bs=%d / repeat=%d / seed=%d ===",
                        workload_spec.name, mode, batch_size, repeat, run_seed,
                    )
                    try:
                        result, latencies, gpu_snaps = run_single_benchmark(
                            workload_name=workload_spec.name,
                            batch_size=batch_size,
                            mode=mode,
                            device=device,
                            warmup_iters=warmup_iters,
                            benchmark_iters=benchmark_iters,
                            seed=run_seed,
                        )
                        result["workload"] = workload_spec.name
                        result["gpu_type"] = gpu_type
                        result["repeat"] = repeat

                        # Per-run latency CSV
                        latency_csv = output_dir / f"{gpu_type}_{workload_spec.name}_{mode}_bs{batch_size}_r{repeat}_latencies.csv"
                        with open(latency_csv, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["iteration", "latency_ms"])
                            for i, lat in enumerate(latencies):
                                writer.writerow([i, round(lat, 4)])

                        # Per-run GPU metrics CSV
                        if gpu_snaps:
                            collector = GpuCollector()
                            collector.snapshots = gpu_snaps
                            collector.save_csv(
                                output_dir / f"{gpu_type}_{workload_spec.name}_{mode}_bs{batch_size}_r{repeat}_gpu_metrics.csv"
                            )

                        all_results.append(result)

                        if prom_enabled:
                            push_benchmark_metrics(
                                job_name=f"benchmark_{gpu_type}",
                                gpu_type=gpu_type,
                                workload=workload_spec.name,
                                batch_size=batch_size,
                                throughput=result["throughput"],
                                latency_p50=result["latency_p50_ms"],
                                latency_p95=result["latency_p95_ms"],
                                latency_p99=result["latency_p99_ms"],
                                gpu_utilization=result["avg_gpu_utilization_pct"],
                                gpu_memory_used_mb=result["avg_gpu_memory_used_mb"],
                            )

                    except Exception as e:
                        logger.error("FAILED: %s/%s/bs%d/r%d: %s", workload_spec.name, mode, batch_size, repeat, e)
                        all_results.append({
                            "gpu_type": gpu_type,
                            "workload": workload_spec.name,
                            "mode": mode,
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "error": str(e),
                        })

    # Write summary CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        for r in all_results[1:]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("Summary CSV written to %s (%d runs)", summary_csv, len(all_results))

    # Write run manifest
    result_checksums = checksum_directory(output_dir, pattern="*.csv")
    manifest = {
        "gpu_type": gpu_type,
        "device": device,
        "config": config,
        "custom_workloads": config.get("custom_workloads", {}),
        "environment": env_info,
        "total_runs": len(all_results),
        "successful_runs": sum(1 for r in all_results if "error" not in r),
        "failed_runs": sum(1 for r in all_results if "error" in r),
        "result_checksums": result_checksums,
    }
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Run manifest written to %s", manifest_path)

    # Auto-log to history database for the recommendation engine
    try:
        from .recommender.history import HistoryStore
        db_path = Path(config.get("history_db", "data/benchmark_history.db"))
        store = HistoryStore(db_path)
        store.log_benchmark_results(all_results, gpu_type)
        store.close()
        logger.info("Results logged to history database: %s", db_path)
    except Exception as e:
        logger.debug("History logging skipped: %s", e)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="GPU Cloud Benchmark Runner")
    parser.add_argument("--config", type=str, default="config/benchmark_config.yaml")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu)")
    parser.add_argument(
        "--workload-target",
        type=str,
        default=None,
        help="Benchmark a custom workload class via import path: module.path:ClassName",
    )
    parser.add_argument(
        "--workload-name",
        type=str,
        default=None,
        help="Optional alias to use with --workload-target; defaults to the class name",
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="Run recommendation engine after benchmark completes",
    )
    args = parser.parse_args()

    output_dir = run_full_benchmark(
        args.config,
        args.device,
        workload_target=args.workload_target,
        workload_name=args.workload_name,
    )
    logger.info("Benchmark complete. Results in %s", output_dir)

    if args.recommend:
        try:
            from .recommender.engine import RecommendationEngine, format_recommendation
            engine = RecommendationEngine()
            result = engine.recommend(results_dir=output_dir)
            print(format_recommendation(result))
        except Exception as e:
            logger.error("Recommendation failed: %s", e)


if __name__ == "__main__":
    main()
