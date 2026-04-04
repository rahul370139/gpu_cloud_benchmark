"""Aggregate raw benchmark CSVs into a single analysis-ready DataFrame."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CV_NOISY_THRESHOLD = 0.10  # 10%


def load_summary_csvs(results_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all benchmark_summary_*.csv files."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("benchmark_summary_*.csv"))
    if not files:
        raise FileNotFoundError(f"No benchmark_summary_*.csv files in {results_dir}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d summary rows from %d files", len(df), len(files))
    return df


def compute_aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, CV across repeats for each (gpu_type, workload, mode, batch_size).

    Returns a DataFrame with one row per unique combination.
    """
    group_cols = ["gpu_type", "workload", "mode", "batch_size"]
    available_cols = [c for c in group_cols if c in df.columns]

    # Only aggregate rows without errors
    if "error" in df.columns:
        clean = df[df["error"].isna() | (df["error"] == "")].copy()
    else:
        clean = df.copy()

    agg = clean.groupby(available_cols).agg(
        mean_throughput=("throughput", "mean"),
        std_throughput=("throughput", "std"),
        min_throughput=("throughput", "min"),
        max_throughput=("throughput", "max"),
        num_repeats=("throughput", "count"),
        mean_latency_p50=("latency_p50_ms", "mean"),
        mean_latency_p95=("latency_p95_ms", "mean"),
        mean_latency_p99=("latency_p99_ms", "mean"),
        mean_latency_mean=("latency_mean_ms", "mean"),
        mean_gpu_util=("avg_gpu_utilization_pct", "mean"),
        mean_gpu_mem=("avg_gpu_memory_used_mb", "mean"),
    ).reset_index()

    agg["cv_throughput"] = agg["std_throughput"] / agg["mean_throughput"]
    agg["cv_throughput"] = agg["cv_throughput"].fillna(0)
    agg["is_noisy"] = agg["cv_throughput"] > CV_NOISY_THRESHOLD

    # Carry forward metadata from first row of each group
    meta_cols = ["model_name", "param_count", "throughput_unit"]
    for col in meta_cols:
        if col in clean.columns:
            first_vals = clean.groupby(available_cols)[col].first().reset_index()
            agg = agg.merge(first_vals, on=available_cols, how="left")

    noisy = agg[agg["is_noisy"]]
    if len(noisy) > 0:
        logger.warning(
            "%d groups flagged as noisy (CV > %.0f%%): %s",
            len(noisy), CV_NOISY_THRESHOLD * 100,
            noisy[available_cols + ["cv_throughput"]].to_string(index=False),
        )

    logger.info("Aggregated %d groups from %d runs", len(agg), len(clean))
    return agg


def load_latency_csvs(results_dir: str | Path) -> pd.DataFrame:
    """Load all per-run latency CSVs into a single DataFrame with run metadata."""
    results_dir = Path(results_dir)
    frames = []
    for f in sorted(results_dir.glob("*_latencies.csv")):
        parts = f.stem.split("_")
        # Filename: {gpu}_{workload}_{mode}_bs{batch}_r{repeat}_latencies
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_gpu_metrics_csvs(results_dir: str | Path) -> pd.DataFrame:
    """Load all GPU metric time-series CSVs."""
    results_dir = Path(results_dir)
    frames = []
    for f in sorted(results_dir.glob("*_gpu_metrics.csv")):
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
