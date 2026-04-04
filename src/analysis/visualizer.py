"""Generate publication-quality benchmark charts with matplotlib and seaborn."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGSIZE = (10, 6)
DPI = 150
PALETTE = "viridis"

sns.set_theme(style="whitegrid", font_scale=1.1)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", path)


def throughput_bar_chart(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: throughput by GPU, colored by workload."""
    path = output_dir / "throughput_by_gpu.png"
    fig, ax = plt.subplots(figsize=FIGSIZE)

    chart_df = df.copy()
    chart_df["label"] = chart_df["workload"] + " (bs=" + chart_df["batch_size"].astype(str) + ")"

    pivot = chart_df.pivot_table(
        index="gpu_type", columns="label", values="mean_throughput", aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax, colormap=PALETTE, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_xlabel("GPU Type")
    ax.set_title("Throughput by GPU and Workload")
    ax.legend(title="Workload", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.xticks(rotation=0)

    _save(fig, path)
    return path


def latency_percentile_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart showing p50, p95, p99 latencies across GPUs."""
    path = output_dir / "latency_percentiles.png"
    fig, ax = plt.subplots(figsize=FIGSIZE)

    melted = df.melt(
        id_vars=["gpu_type", "workload", "batch_size"],
        value_vars=["mean_latency_p50", "mean_latency_p95", "mean_latency_p99"],
        var_name="percentile", value_name="latency_ms",
    )
    melted["percentile"] = melted["percentile"].str.replace("mean_latency_", "").str.upper()
    melted["group"] = melted["gpu_type"] + "\n" + melted["workload"]

    sns.barplot(data=melted, x="group", y="latency_ms", hue="percentile", ax=ax, palette="mako")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("")
    ax.set_title("Latency Percentiles by GPU and Workload")
    ax.legend(title="Percentile")

    _save(fig, path)
    return path


def throughput_vs_cost_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    """Scatter: x = $/hr, y = throughput, bubble size = GPU memory."""
    path = output_dir / "throughput_vs_cost.png"
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "cost_per_hour" not in df.columns or "gpu_memory_gb" not in df.columns:
        logger.warning("Cost/memory columns missing — skipping throughput_vs_cost")
        plt.close(fig)
        return path

    for workload in df["workload"].unique():
        subset = df[df["workload"] == workload]
        sizes = subset["gpu_memory_gb"].clip(lower=1) * 8
        ax.scatter(
            subset["cost_per_hour"], subset["mean_throughput"],
            s=sizes, alpha=0.7, label=workload, edgecolors="black", linewidths=0.5,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["gpu_type"], (row["cost_per_hour"], row["mean_throughput"]),
                fontsize=7, ha="center", va="bottom",
            )

    ax.set_xlabel("Cost ($/hr)")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Throughput vs. Cost (bubble size = GPU memory)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    _save(fig, path)
    return path


def cost_efficiency_bar_chart(df: pd.DataFrame, output_dir: Path) -> Path:
    """Horizontal bar chart ranking GPUs by throughput-per-dollar."""
    path = output_dir / "cost_efficiency.png"
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "throughput_per_dollar" not in df.columns:
        logger.warning("throughput_per_dollar column missing — skipping chart")
        plt.close(fig)
        return path

    chart_df = df.copy()
    chart_df["label"] = chart_df["gpu_type"] + " / " + chart_df["workload"]
    chart_df = chart_df.sort_values("throughput_per_dollar", ascending=True)

    colors = sns.color_palette(PALETTE, len(chart_df))
    ax.barh(chart_df["label"], chart_df["throughput_per_dollar"], color=colors, edgecolor="white")
    ax.set_xlabel("Throughput per Dollar (samples/$)")
    ax.set_title("Cost Efficiency Ranking")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())

    _save(fig, path)
    return path


def gpu_utilization_timeseries(gpu_metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    """Line plot of GPU utilization over time from collector snapshots."""
    path = output_dir / "gpu_utilization_timeseries.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if gpu_metrics_df.empty:
        logger.warning("No GPU metrics data — skipping utilization timeseries")
        plt.close(fig)
        return path

    for source, group in gpu_metrics_df.groupby("source_file"):
        t = group["timestamp"] - group["timestamp"].iloc[0]
        label = source.replace("_gpu_metrics.csv", "")
        axes[0].plot(t, group["utilization_pct"], label=label, alpha=0.8)
        axes[1].plot(t, group["memory_used_mb"], label=label, alpha=0.8)

    axes[0].set_ylabel("GPU Utilization (%)")
    axes[0].set_title("GPU Utilization Over Time")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel("Memory Used (MB)")
    axes[1].set_xlabel("Time (sec)")
    axes[1].legend(fontsize=7, ncol=2)

    _save(fig, path)
    return path


def batch_size_scaling_curve(df: pd.DataFrame, output_dir: Path) -> Path:
    """Line chart: throughput vs batch_size, one line per (gpu_type, workload)."""
    path = output_dir / "batch_size_scaling.png"
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for (gpu, wl), group in df.groupby(["gpu_type", "workload"]):
        sorted_g = group.sort_values("batch_size")
        ax.plot(
            sorted_g["batch_size"], sorted_g["mean_throughput"],
            marker="o", label=f"{gpu} / {wl}", linewidth=2,
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Throughput Scaling with Batch Size")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    ax.set_xscale("log", base=2)

    _save(fig, path)
    return path


def cv_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Heatmap of coefficient of variation across (gpu_type, workload+batch_size)."""
    path = output_dir / "cv_heatmap.png"
    fig, ax = plt.subplots(figsize=FIGSIZE)

    chart_df = df.copy()
    chart_df["config"] = chart_df["workload"] + " bs=" + chart_df["batch_size"].astype(str)

    pivot = chart_df.pivot_table(
        index="gpu_type", columns="config", values="cv_throughput", aggfunc="mean",
    )
    sns.heatmap(
        pivot, annot=True, fmt=".2%", cmap="YlOrRd", ax=ax,
        linewidths=0.5, cbar_kws={"label": "CV (Throughput)"},
    )
    ax.set_title("Reproducibility: Coefficient of Variation")

    _save(fig, path)
    return path


def generate_all_charts(
    agg_df: pd.DataFrame,
    gpu_metrics_df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Run all chart generators. Returns list of saved file paths."""
    output_dir = Path(output_dir)
    paths = []

    paths.append(throughput_bar_chart(agg_df, output_dir))
    paths.append(latency_percentile_plot(agg_df, output_dir))
    paths.append(throughput_vs_cost_scatter(agg_df, output_dir))
    paths.append(cost_efficiency_bar_chart(agg_df, output_dir))
    paths.append(gpu_utilization_timeseries(gpu_metrics_df, output_dir))
    paths.append(batch_size_scaling_curve(agg_df, output_dir))
    paths.append(cv_heatmap(agg_df, output_dir))

    logger.info("Generated %d charts in %s", len(paths), output_dir)
    return paths
