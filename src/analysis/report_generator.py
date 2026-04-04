"""Generate a self-contained HTML benchmark report with embedded charts."""

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Template

logger = logging.getLogger(__name__)

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
</style>
</head>
<body>
<h1>GPU Cloud Benchmark Report</h1>
<p class="meta">Generated {{ generated_at }} | GPU: {{ gpu_type }} | Runs: {{ total_runs }} ({{ failed_runs }} failed)</p>

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

    gpu_type = manifest.get("gpu_type", "Unknown") if manifest else "Unknown"
    total_runs = manifest.get("total_runs", 0) if manifest else len(summary_rows)
    failed_runs = manifest.get("failed_runs", 0) if manifest else 0

    html = REPORT_TEMPLATE.render(
        title=f"{gpu_type} Benchmark",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        gpu_type=gpu_type,
        total_runs=total_runs,
        failed_runs=failed_runs,
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
