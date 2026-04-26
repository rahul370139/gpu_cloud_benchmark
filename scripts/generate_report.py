#!/usr/bin/env python3
"""CLI to generate an HTML benchmark report from collected results."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.preprocessor import load_summary_csvs, compute_aggregate_stats, load_gpu_metrics_csvs
from src.analysis.visualizer import generate_all_charts
from src.analysis.report_generator import generate_html_report
from src.cost.calculator import load_gpu_rates, compute_cost_metrics, save_cost_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory with benchmark CSVs")
    parser.add_argument("--cost-rates", type=str, default="config/gpu_cost_rates.yaml", help="GPU cost rates YAML")
    parser.add_argument("--output", type=str, default="results/report.html", help="Output HTML path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"

    # Load and aggregate
    logger.info("Loading results from %s", results_dir)
    try:
        raw_df = load_summary_csvs(results_dir)
    except FileNotFoundError as e:
        logger.error("No benchmark results found: %s", e)
        sys.exit(1)

    agg_df = compute_aggregate_stats(raw_df)

    # Cost metrics
    cost_df = None
    cost_rates_path = Path(args.cost_rates)
    if cost_rates_path.exists():
        gpu_rates = load_gpu_rates(cost_rates_path)
        cost_df = compute_cost_metrics(agg_df, gpu_rates)
        save_cost_report(cost_df, results_dir / "cost_comparison.csv")
    else:
        logger.warning("Cost rates file not found at %s — skipping cost analysis", cost_rates_path)

    # Load GPU time-series metrics
    gpu_metrics_df = load_gpu_metrics_csvs(results_dir)

    # Use cost-enriched df for charts if available
    chart_df = cost_df if cost_df is not None else agg_df

    # Generate charts
    generate_all_charts(chart_df, gpu_metrics_df, figures_dir)

    # Load manifest if available. When aggregating multi-GPU results, there may
    # be multiple nested manifests rather than one root-level file, so fall back
    # to a synthetic summary based on the merged raw results.
    manifest = None
    manifest_path = results_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "environment": {},
            "total_runs": len(raw_df),
            "failed_runs": int(raw_df["error"].notna().sum()) if "error" in raw_df.columns else 0,
        }

    # Generate HTML report
    output_path = generate_html_report(
        agg_df=chart_df,
        cost_df=cost_df,
        figures_dir=figures_dir,
        manifest=manifest,
        output_path=args.output,
    )
    logger.info("Report generated: %s", output_path)


if __name__ == "__main__":
    main()
