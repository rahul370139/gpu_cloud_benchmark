"""Multi-criteria GPU scoring engine.

Normalises throughput, cost-efficiency, and latency to [0, 1], applies
configurable weights, and produces a ranked list of GPUs with composite
scores and human-readable reasoning.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "throughput": 0.40,
    "cost_efficiency": 0.35,
    "latency": 0.25,
}


@dataclass
class GpuScore:
    gpu_type: str
    composite_score: float
    rank: int

    throughput: float
    throughput_score: float
    throughput_unit: str

    cost_per_hour: float
    throughput_per_dollar: float
    cost_score: float

    latency_p95_ms: float
    latency_score: float

    avg_gpu_util_pct: float
    avg_gpu_mem_mb: float
    confidence_note: str  # "full run" or "partial, CI=[x, y]"

    reasoning: str = ""
    detail_lines: list[str] = field(default_factory=list)


def _min_max_normalise(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    normed = (series - lo) / (hi - lo)
    return normed if higher_is_better else 1.0 - normed


def _build_reasoning(row: pd.Series, rank: int, total: int) -> tuple[str, list[str]]:
    lines = []
    if rank == 1:
        lines.append("Highest composite score across all feasible GPUs")
    else:
        lines.append(f"Ranked #{rank} of {total} GPUs")

    if row.get("cost_per_hour", 0) > 0:
        tpd = row.get("throughput_per_dollar", 0)
        tu = row.get("throughput_unit", "samples")
        lines.append(f"Throughput-per-dollar: {tpd:,.0f} {tu}/$")

    lat = row.get("latency_p95_ms", row.get("mean_latency_p95", 0))
    if lat > 0:
        lines.append(f"P95 latency: {lat:.1f} ms")

    util = row.get("avg_gpu_util_pct", row.get("mean_gpu_util", 0))
    if util > 0:
        lines.append(f"GPU utilization: {util:.0f}%")

    headline = lines[0] if lines else ""
    return headline, lines


def score_gpus(
    df: pd.DataFrame,
    gpu_rates: dict[str, dict] | None = None,
    weights: dict[str, float] | None = None,
) -> list[GpuScore]:
    """Score and rank GPUs from a DataFrame of benchmark results.

    Expected columns (at minimum):
        gpu_type, throughput (or mean_throughput),
        latency_p95_ms (or mean_latency_p95), cost_per_hour (optional).
    Adds throughput_per_dollar if cost data is available.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = w["throughput"] + w["cost_efficiency"] + w["latency"]
    for k in w:
        w[k] /= total

    df = df.copy()

    tp_col = "mean_throughput" if "mean_throughput" in df.columns else "throughput"
    lat_col = (
        "mean_latency_p95"
        if "mean_latency_p95" in df.columns
        else "latency_p95_ms"
    )
    tu_col = "throughput_unit" if "throughput_unit" in df.columns else None

    if "cost_per_hour" not in df.columns and gpu_rates:
        df["cost_per_hour"] = df["gpu_type"].map(
            lambda g: gpu_rates.get(g, {}).get("cost_per_hour", 0.0)
        )

    if "cost_per_hour" not in df.columns:
        df["cost_per_hour"] = 0.0

    if "throughput_per_dollar" not in df.columns:
        df["throughput_per_dollar"] = df.apply(
            lambda r: r[tp_col] * 3600.0 / r["cost_per_hour"]
            if r.get("cost_per_hour", 0) > 0
            else 0.0,
            axis=1,
        )

    df["_tp_norm"] = _min_max_normalise(df[tp_col], higher_is_better=True)
    df["_lat_norm"] = _min_max_normalise(df[lat_col], higher_is_better=False)

    has_cost = (df["cost_per_hour"] > 0).any()
    if has_cost:
        df["_cost_norm"] = _min_max_normalise(df["throughput_per_dollar"], higher_is_better=True)
    else:
        df["_cost_norm"] = 1.0

    df["composite_score"] = (
        w["throughput"] * df["_tp_norm"]
        + w["cost_efficiency"] * df["_cost_norm"]
        + w["latency"] * df["_lat_norm"]
    )

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    scores: list[GpuScore] = []
    for _, row in df.iterrows():
        is_partial = bool(row.get("is_partial", 0))
        if is_partial:
            ci_lo = row.get("confidence_low", 0)
            ci_hi = row.get("confidence_high", 0)
            conf = f"partial run, 95% CI=[{ci_lo:.0f}, {ci_hi:.0f}]"
        else:
            conf = "full benchmark run"

        headline, detail = _build_reasoning(row, int(row["rank"]), len(df))

        util_col = "avg_gpu_util_pct" if "avg_gpu_util_pct" in row.index else "mean_gpu_util"
        mem_col = "avg_gpu_mem_mb" if "avg_gpu_mem_mb" in row.index else "mean_gpu_mem"

        scores.append(
            GpuScore(
                gpu_type=row["gpu_type"],
                composite_score=round(float(row["composite_score"]), 4),
                rank=int(row["rank"]),
                throughput=round(float(row[tp_col]), 2),
                throughput_score=round(float(row["_tp_norm"]), 4),
                throughput_unit=row.get(tu_col, "") if tu_col else "",
                cost_per_hour=round(float(row.get("cost_per_hour", 0)), 4),
                throughput_per_dollar=round(float(row.get("throughput_per_dollar", 0)), 2),
                cost_score=round(float(row["_cost_norm"]), 4),
                latency_p95_ms=round(float(row.get(lat_col, 0)), 4),
                latency_score=round(float(row["_lat_norm"]), 4),
                avg_gpu_util_pct=round(float(row.get(util_col, 0)), 2),
                avg_gpu_mem_mb=round(float(row.get(mem_col, 0)), 2),
                confidence_note=conf,
                reasoning=headline,
                detail_lines=detail,
            )
        )
    return scores
