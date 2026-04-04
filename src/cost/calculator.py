"""Compute cost-efficiency metrics: throughput-per-dollar, cost-per-1K-samples."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_gpu_rates(rates_path: str | Path) -> dict[str, dict]:
    """Load GPU cost rates from YAML.  Returns {gpu_name: {cost_per_hour, ...}}."""
    with open(rates_path) as f:
        data = yaml.safe_load(f)
    rates = {}
    for gpu, info in data.get("gpu_rates", {}).items():
        cost = info.get("cost_per_gpu_hour", info.get("cost_per_hour", 0.0))
        rates[gpu] = {
            "instance_type": info.get("instance_type", "unknown"),
            "cost_per_hour": cost,
            "gpu_memory_gb": info.get("gpu_memory_gb", 0),
        }
    return rates


def compute_cost_metrics(
    results_df: pd.DataFrame,
    gpu_rates: dict[str, dict],
) -> pd.DataFrame:
    """Add cost-efficiency columns to a benchmark results DataFrame.

    Expects columns: gpu_type, workload, batch_size, mean_throughput.
    Adds: cost_per_hour, throughput_per_dollar, cost_per_1k_samples, cost_efficiency_rank.
    """
    df = results_df.copy()

    df["cost_per_hour"] = df["gpu_type"].map(
        lambda g: gpu_rates.get(g, {}).get("cost_per_hour", 0.0)
    )
    df["gpu_memory_gb"] = df["gpu_type"].map(
        lambda g: gpu_rates.get(g, {}).get("gpu_memory_gb", 0)
    )

    # throughput_per_dollar = samples processed per dollar spent
    # = throughput (samples/sec) * 3600 (sec/hr) / cost_per_hour ($/hr)
    df["throughput_per_dollar"] = df.apply(
        lambda r: (r["mean_throughput"] * 3600.0 / r["cost_per_hour"])
        if r["cost_per_hour"] > 0 else 0.0,
        axis=1,
    )

    # cost to process 1000 samples
    df["cost_per_1k_samples"] = df.apply(
        lambda r: (1000.0 / r["mean_throughput"]) * (r["cost_per_hour"] / 3600.0)
        if r["mean_throughput"] > 0 else float("inf"),
        axis=1,
    )

    # Rank within each workload+batch_size group (1 = most cost-efficient)
    df["cost_efficiency_rank"] = (
        df.groupby(["workload", "batch_size"])["throughput_per_dollar"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    logger.info("Cost metrics computed for %d rows", len(df))
    return df


def save_cost_report(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write cost-comparison CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "gpu_type", "workload", "batch_size",
        "mean_throughput", "cost_per_hour", "gpu_memory_gb",
        "throughput_per_dollar", "cost_per_1k_samples", "cost_efficiency_rank",
    ]
    existing = [c for c in cols if c in df.columns]
    df[existing].to_csv(output_path, index=False)
    logger.info("Cost report saved to %s", output_path)
    return output_path
