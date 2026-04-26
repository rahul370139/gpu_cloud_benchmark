"""Generate a self-contained HTML benchmark report with embedded charts."""

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Template

logger = logging.getLogger(__name__)


def _scenario_label(row: pd.Series) -> str:
    return f"{row['workload']} | {row['mode']} | bs={row['batch_size']}"


def _format_numeric(value: object, decimals: int = 1, empty: str = "—") -> str:
    if pd.isna(value):
        return empty
    return f"{float(value):,.{decimals}f}"


def _format_int(value: object, empty: str = "—") -> str:
    if pd.isna(value):
        return empty
    return f"{int(value)}"


def _to_html_table(df: pd.DataFrame, classes: str = "matrix-table") -> str:
    if df.empty:
        return "<p>No data available.</p>"
    display_df = df.fillna("—")
    return display_df.to_html(index=False, classes=classes, border=0)


def _build_gpu_overview(agg_df: pd.DataFrame, cost_df: pd.DataFrame | None) -> pd.DataFrame:
    overview = (
        agg_df.groupby("gpu_type")
        .agg(
            scenarios=("workload", "count"),
            workloads=("workload", "nunique"),
            mean_throughput=("mean_throughput", "mean"),
            median_p95_ms=("mean_latency_p95", "median"),
            mean_gpu_util_pct=("mean_gpu_util", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "gpu_type": "GPU",
                "scenarios": "Scenarios",
                "workloads": "Workloads",
                "mean_throughput": "Avg Throughput",
                "median_p95_ms": "Median P95 (ms)",
                "mean_gpu_util_pct": "Avg GPU Util (%)",
            }
        )
    )

    if cost_df is not None and not cost_df.empty:
        cost_overview = (
            cost_df.groupby("gpu_type")
            .agg(
                hourly_cost=("cost_per_hour", "first"),
                median_throughput_per_dollar=("throughput_per_dollar", "median"),
            )
            .reset_index()
            .rename(
                columns={
                    "gpu_type": "GPU",
                    "hourly_cost": "$/hr",
                    "median_throughput_per_dollar": "Median Throughput/$",
                }
            )
        )
        overview = overview.merge(cost_overview, on="GPU", how="left")

    for col in overview.columns:
        if col in {"GPU"}:
            continue
        if col in {"Scenarios", "Workloads"}:
            overview[col] = overview[col].map(_format_int)
        elif col == "$/hr":
            overview[col] = overview[col].map(lambda x: f"${float(x):.3f}" if not pd.isna(x) else "—")
        else:
            overview[col] = overview[col].map(_format_numeric)
    return overview


def _build_scenario_leaderboard(agg_df: pd.DataFrame, cost_df: pd.DataFrame | None) -> pd.DataFrame:
    base = agg_df.copy()
    base["Scenario"] = base.apply(_scenario_label, axis=1)

    rows: list[dict] = []
    grouped = base.groupby(["workload", "mode", "batch_size"], sort=True)
    cost_lookup = None
    if cost_df is not None and not cost_df.empty:
        cost_lookup = cost_df.copy()
        cost_lookup["Scenario"] = cost_lookup.apply(_scenario_label, axis=1)

    for _, group in grouped:
        scenario = _scenario_label(group.iloc[0])
        best_thr = group.loc[group["mean_throughput"].idxmax()]
        best_lat = group.loc[group["mean_latency_p95"].idxmin()]
        row = {
            "Scenario": scenario,
            "Best Throughput GPU": best_thr["gpu_type"],
            "Best Throughput": _format_numeric(best_thr["mean_throughput"]),
            "Lowest P95 GPU": best_lat["gpu_type"],
            "Lowest P95 (ms)": _format_numeric(best_lat["mean_latency_p95"], decimals=2),
        }
        if cost_lookup is not None:
            scenario_cost = cost_lookup[cost_lookup["Scenario"] == scenario]
            if not scenario_cost.empty:
                best_cost = scenario_cost.loc[scenario_cost["throughput_per_dollar"].idxmax()]
                row["Best Value GPU"] = best_cost["gpu_type"]
                row["Best Throughput/$"] = _format_numeric(best_cost["throughput_per_dollar"], decimals=0)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_winner_summary(agg_df: pd.DataFrame, cost_df: pd.DataFrame | None) -> pd.DataFrame:
    base = agg_df.copy()
    counts: dict[str, dict[str, int]] = {}

    for _, group in base.groupby(["workload", "mode", "batch_size"], sort=True):
        thr_gpu = group.loc[group["mean_throughput"].idxmax(), "gpu_type"]
        lat_gpu = group.loc[group["mean_latency_p95"].idxmin(), "gpu_type"]
        counts.setdefault(thr_gpu, {"Throughput Wins": 0, "Latency Wins": 0, "Value Wins": 0})
        counts.setdefault(lat_gpu, {"Throughput Wins": 0, "Latency Wins": 0, "Value Wins": 0})
        counts[thr_gpu]["Throughput Wins"] += 1
        counts[lat_gpu]["Latency Wins"] += 1

    if cost_df is not None and not cost_df.empty:
        for _, group in cost_df.groupby(["workload", "mode", "batch_size"], sort=True):
            val_gpu = group.loc[group["throughput_per_dollar"].idxmax(), "gpu_type"]
            counts.setdefault(val_gpu, {"Throughput Wins": 0, "Latency Wins": 0, "Value Wins": 0})
            counts[val_gpu]["Value Wins"] += 1

    rows = [{"GPU": gpu, **metrics} for gpu, metrics in sorted(counts.items())]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["Throughput Wins", "Latency Wins", "Value Wins"]:
        df[col] = df[col].map(_format_int)
    return df


def _build_metric_matrix(
    df: pd.DataFrame,
    value_col: str,
    value_label: str,
    decimals: int = 1,
) -> pd.DataFrame:
    matrix_df = df.copy()
    matrix_df["Scenario"] = matrix_df.apply(_scenario_label, axis=1)
    pivot = (
        matrix_df.pivot_table(
            index="Scenario",
            columns="gpu_type",
            values=value_col,
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in pivot.columns:
        if col == "Scenario":
            continue
        pivot[col] = pivot[col].map(lambda x: _format_numeric(x, decimals=decimals))
    pivot.columns = ["Scenario", *[f"{col} {value_label}" for col in pivot.columns[1:]]]
    return pivot


REPORT_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GPU Benchmark Report — {{ title }}</title>
<style>
  :root { --bg: #fafafa; --text: #222; --accent: #2563eb; --border: #e5e7eb; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; padding: 2rem; max-width: 1200px; margin: auto; }
  h1 { font-size: 1.8rem; border-bottom: 3px solid var(--accent); padding-bottom: 0.4rem; margin-bottom: 1.5rem; }
  h2 { font-size: 1.3rem; color: var(--accent); margin: 2rem 0 0.8rem; }
  h3 { font-size: 1.1rem; margin: 1.2rem 0 0.5rem; }
  table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }
  th, td { padding: 0.5rem 0.8rem; border: 1px solid var(--border); text-align: right; }
  th { background: var(--accent); color: white; text-align: center; }
  tr:nth-child(even) { background: #f1f5f9; }
  .chart { margin: 1.5rem 0; text-align: center; }
  .chart img { max-width: 100%; border: 1px solid var(--border); border-radius: 4px; }
  .env-block { background: #f8fafc; border: 1px solid var(--border); border-radius: 4px; padding: 1rem; font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; overflow-x: auto; }
  .noisy { background: #fef3c7; }
  .meta { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px; font-size: 0.75rem; font-weight: 600; }
  .badge-ok { background: #d1fae5; color: #065f46; }
  .badge-warn { background: #fef3c7; color: #92400e; }
  .subtle { color: #475569; margin-bottom: 1rem; }
  .matrix-table td:first-child, .matrix-table th:first-child { text-align: left; }
</style>
</head>
<body>
<h1>GPU Cloud Benchmark Report</h1>
<p class="meta">Generated {{ generated_at }} | GPUs: {{ gpu_types|join(", ") }} | Scenarios: {{ scenario_count }} | Runs: {{ total_runs }} ({{ failed_runs }} failed)</p>

<h2>GPU Overview</h2>
<p class="subtle">A single side-by-side view of coverage, average throughput, latency, utilization, and cost metrics for each GPU.</p>
{{ gpu_overview_table }}

<h2>Winner Summary</h2>
<p class="subtle">Count of scenario wins by throughput, latency, and value.</p>
{{ winner_summary_table }}

<h2>Scenario Leaders</h2>
<p class="subtle">For each workload and batch size, this shows which GPU wins on throughput, latency, and value.</p>
{{ scenario_leaderboard_table }}

<h2>Comparison Matrices</h2>
<h3>Throughput Matrix</h3>
{{ throughput_matrix_table }}

<h3>P95 Latency Matrix</h3>
{{ latency_matrix_table }}

{% if value_matrix_table %}
<h3>Throughput per Dollar Matrix</h3>
{{ value_matrix_table }}
{% endif %}

<h2>Summary</h2>
<table>
<tr>
  <th>GPU</th><th>Workload</th><th>Mode</th><th>Batch Size</th>
  <th>Throughput</th><th>Unit</th>
  <th>P50 (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th>
  <th>CV</th><th>Status</th>
</tr>
{% for row in summary_rows %}
<tr class="{{ 'noisy' if row.is_noisy else '' }}">
  <td>{{ row.gpu_type }}</td>
  <td>{{ row.workload }}</td>
  <td>{{ row.mode }}</td>
  <td>{{ row.batch_size }}</td>
  <td>{{ "%.1f"|format(row.mean_throughput) }}</td>
  <td>{{ row.throughput_unit | default("samples/sec") }}</td>
  <td>{{ "%.2f"|format(row.mean_latency_p50) }}</td>
  <td>{{ "%.2f"|format(row.mean_latency_p95) }}</td>
  <td>{{ "%.2f"|format(row.mean_latency_p99) }}</td>
  <td>{{ "%.1f%%"|format(row.cv_throughput * 100) }}</td>
  <td>
    {% if row.is_noisy %}
      <span class="badge badge-warn">NOISY</span>
    {% else %}
      <span class="badge badge-ok">OK</span>
    {% endif %}
  </td>
</tr>
{% endfor %}
</table>

{% if cost_rows %}
<h2>Cost Efficiency</h2>
<table>
<tr>
  <th>Rank</th><th>GPU</th><th>Workload</th><th>Batch Size</th>
  <th>Throughput</th><th>$/hr</th><th>Throughput/$</th><th>$/1K samples</th>
</tr>
{% for row in cost_rows %}
<tr>
  <td>{{ row.cost_efficiency_rank }}</td>
  <td>{{ row.gpu_type }}</td>
  <td>{{ row.workload }}</td>
  <td>{{ row.batch_size }}</td>
  <td>{{ "%.1f"|format(row.mean_throughput) }}</td>
  <td>{{ "$%.3f"|format(row.cost_per_hour) }}</td>
  <td>{{ "%.0f"|format(row.throughput_per_dollar) }}</td>
  <td>{{ "$%.6f"|format(row.cost_per_1k_samples) }}</td>
</tr>
{% endfor %}
</table>
{% endif %}

<h2>Charts</h2>
{% for chart in charts %}
<div class="chart">
  <h3>{{ chart.title }}</h3>
  <img src="data:image/png;base64,{{ chart.b64 }}" alt="{{ chart.title }}">
</div>
{% endfor %}

<h2>Environment</h2>
<div class="env-block">{{ env_json }}</div>

{% if checksums %}
<h2>Reproducibility Checksums</h2>
<div class="env-block">{{ checksums_json }}</div>
{% endif %}

</body>
</html>
""")


def _img_to_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _chart_title(filename: str) -> str:
    return filename.replace("_", " ").replace(".png", "").title()


def generate_html_report(
    agg_df: pd.DataFrame,
    cost_df: pd.DataFrame | None,
    figures_dir: str | Path,
    manifest: dict | None,
    output_path: str | Path,
) -> Path:
    """Render a self-contained HTML report.

    Args:
        agg_df: Aggregated benchmark stats (from preprocessor.compute_aggregate_stats).
        cost_df: Cost-enriched DataFrame (from cost.calculator), or None.
        figures_dir: Directory containing chart PNGs.
        manifest: run_manifest.json dict, or None.
        output_path: Where to write the HTML file.
    """
    output_path = Path(output_path)
    figures_dir = Path(figures_dir)

    charts = []
    for png in sorted(figures_dir.glob("*.png")):
        b64 = _img_to_b64(png)
        if b64:
            charts.append({"title": _chart_title(png.stem), "b64": b64})

    summary_rows = agg_df.to_dict("records") if not agg_df.empty else []

    cost_rows = []
    if cost_df is not None and not cost_df.empty:
        cost_rows = cost_df.sort_values(
            ["workload", "batch_size", "cost_efficiency_rank"]
        ).to_dict("records")

    env_json = json.dumps(manifest.get("environment", {}), indent=2) if manifest else "{}"
    checksums = manifest.get("result_checksums", {}) if manifest else {}
    checksums_json = json.dumps(checksums, indent=2) if checksums else ""

    gpu_types = sorted(agg_df["gpu_type"].dropna().unique().tolist()) if not agg_df.empty else []
    total_runs = manifest.get("total_runs", 0) if manifest else len(summary_rows)
    failed_runs = manifest.get("failed_runs", 0) if manifest else 0
    scenario_count = len(agg_df[["workload", "mode", "batch_size"]].drop_duplicates()) if not agg_df.empty else 0

    gpu_overview_table = _to_html_table(_build_gpu_overview(agg_df, cost_df))
    winner_summary_table = _to_html_table(_build_winner_summary(agg_df, cost_df))
    scenario_leaderboard_table = _to_html_table(_build_scenario_leaderboard(agg_df, cost_df))
    throughput_matrix_table = _to_html_table(
        _build_metric_matrix(agg_df, "mean_throughput", value_label="", decimals=1)
    )
    latency_matrix_table = _to_html_table(
        _build_metric_matrix(agg_df, "mean_latency_p95", value_label="P95 (ms)", decimals=2)
    )
    value_matrix_table = ""
    if cost_df is not None and not cost_df.empty:
        value_matrix_table = _to_html_table(
            _build_metric_matrix(cost_df, "throughput_per_dollar", value_label="Throughput/$", decimals=0)
        )

    html = REPORT_TEMPLATE.render(
        title="GPU Comparison",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        gpu_types=gpu_types or ["Unknown"],
        scenario_count=scenario_count,
        total_runs=total_runs,
        failed_runs=failed_runs,
        gpu_overview_table=gpu_overview_table,
        winner_summary_table=winner_summary_table,
        scenario_leaderboard_table=scenario_leaderboard_table,
        throughput_matrix_table=throughput_matrix_table,
        latency_matrix_table=latency_matrix_table,
        value_matrix_table=value_matrix_table,
        summary_rows=summary_rows,
        cost_rows=cost_rows,
        charts=charts,
        env_json=env_json,
        checksums=bool(checksums),
        checksums_json=checksums_json,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s (%d KB)", output_path, len(html) // 1024)
    return output_path
