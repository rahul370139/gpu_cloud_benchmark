"""Recommendation engine — the brain that ties scoring, constraints,
history, partial profiling, and prediction together.

Three operating modes:
    recommend  — analyse existing results, produce ranked GPU advice
    partial    — run short benchmarks with early stopping, then recommend
    predict    — estimate performance from history (zero cost, zero GPU)
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from ..analysis.preprocessor import load_summary_csvs, compute_aggregate_stats
from ..cost.calculator import load_gpu_rates
from .scorer import score_gpus, GpuScore
from .constraints import UserConstraints, apply_constraints, ExcludedGpu
from .history import HistoryStore
from .partial import PartialProfiler, PartialResult
from .predictor import WorkloadPredictor

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Orchestrates the full recommendation pipeline."""

    def __init__(
        self,
        config_path: str | Path = "config/recommendation_config.yaml",
        cost_rates_path: str | Path = "config/gpu_cost_rates.yaml",
        history_db_path: str | Path | None = None,
    ):
        self.config_path = Path(config_path)
        self.cost_rates_path = Path(cost_rates_path)

        self.cfg = {}
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.cfg = yaml.safe_load(f) or {}

        self.gpu_rates = {}
        if self.cost_rates_path.exists():
            self.gpu_rates = load_gpu_rates(self.cost_rates_path)

        db = history_db_path or self.cfg.get("history", {}).get(
            "database_path", "data/benchmark_history.db"
        )
        self.history = HistoryStore(db)
        self.predictor = WorkloadPredictor.from_config(self.cfg)
        self.weights = self.cfg.get("scoring", {}).get("weights")

    # ------------------------------------------------------------------
    # Mode 1: Recommend from existing benchmark results
    # ------------------------------------------------------------------

    def recommend(
        self,
        results_dir: str | Path = "results",
        constraints: UserConstraints | None = None,
        workload: str | None = None,
        mode: str | None = None,
        batch_size: int | None = None,
    ) -> dict:
        """Analyse results directory and produce GPU recommendation."""
        results_dir = Path(results_dir)

        raw_df = load_summary_csvs(results_dir)
        agg_df = compute_aggregate_stats(raw_df)

        if workload:
            agg_df = agg_df[agg_df["workload"] == workload]
        if mode:
            agg_df = agg_df[agg_df["mode"] == mode]
        if batch_size is not None:
            agg_df = agg_df[agg_df["batch_size"] == batch_size]

        if agg_df.empty:
            return self._empty_result("No matching benchmark results found.")

        scores = score_gpus(agg_df, gpu_rates=self.gpu_rates, weights=self.weights)
        feasible, excluded = apply_constraints(scores, constraints or UserConstraints())

        result = self._build_output(feasible, excluded, constraints or UserConstraints(), source="benchmark")
        self._log_to_history(result, workload, mode, batch_size, constraints)
        return result

    # ------------------------------------------------------------------
    # Mode 2: Partial benchmark then recommend
    # ------------------------------------------------------------------

    def partial_and_recommend(
        self,
        benchmark_config_path: str | Path = "config/benchmark_config.yaml",
        constraints: UserConstraints | None = None,
        device: str | None = None,
    ) -> dict:
        """Run short convergence-checked benchmarks, then recommend."""
        profiler = PartialProfiler.from_config(self.config_path)
        partial_results = profiler.run_suite(benchmark_config_path, device=device)

        if not partial_results:
            return self._empty_result("Partial benchmarks produced no results.")

        for pr in partial_results:
            cost = self.gpu_rates.get(pr.gpu_type, {}).get("cost_per_hour", 0)
            self.history.log_partial_result(pr, cost_per_hour=cost)

        rows = []
        for pr in partial_results:
            cost = self.gpu_rates.get(pr.gpu_type, {}).get("cost_per_hour", 0)
            tpd = pr.estimated_throughput * 3600 / cost if cost > 0 else 0
            rows.append({
                "gpu_type": pr.gpu_type,
                "workload": pr.workload,
                "mode": pr.mode,
                "batch_size": pr.batch_size,
                "mean_throughput": pr.estimated_throughput,
                "mean_latency_p95": pr.latency_p95_ms,
                "avg_gpu_util_pct": pr.avg_gpu_util_pct,
                "avg_gpu_mem_mb": pr.avg_gpu_mem_mb,
                "cost_per_hour": cost,
                "throughput_per_dollar": round(tpd, 2),
                "throughput_unit": pr.throughput_unit,
                "is_partial": 1,
                "confidence_low": pr.confidence_low,
                "confidence_high": pr.confidence_high,
                "converged": pr.converged,
            })
        df = pd.DataFrame(rows)

        scores = score_gpus(df, gpu_rates=self.gpu_rates, weights=self.weights)
        feasible, excluded = apply_constraints(scores, constraints or UserConstraints())

        result = self._build_output(feasible, excluded, constraints or UserConstraints(), source="partial")
        result["partial_details"] = [
            {
                "workload": pr.workload,
                "mode": pr.mode,
                "batch_size": pr.batch_size,
                "gpu_type": pr.gpu_type,
                "estimated_throughput": pr.estimated_throughput,
                "ci_95": [pr.confidence_low, pr.confidence_high],
                "converged": pr.converged,
                "iterations": pr.iterations_run,
                "wall_time_sec": pr.wall_time_sec,
            }
            for pr in partial_results
        ]
        return result

    # ------------------------------------------------------------------
    # Mode 3: Predict from history (no benchmark needed)
    # ------------------------------------------------------------------

    def predict_and_recommend(
        self,
        param_count: int,
        batch_size: int,
        mode: str = "inference",
        family: str = "vision",
        constraints: UserConstraints | None = None,
    ) -> dict:
        """Predict GPU performance from historical data and recommend."""
        history_df = self.history.get_all_runs()

        if history_df.empty:
            return self._empty_result(
                "No historical data available. Run a benchmark first to populate history."
            )

        pred_df = self.predictor.predict_with_cost(
            history_df, self.gpu_rates, param_count, batch_size, mode, family,
        )
        if pred_df.empty:
            return self._empty_result(
                f"Insufficient history ({len(history_df)} entries). "
                f"Need at least {self.predictor.min_entries} to predict."
            )

        scores = score_gpus(pred_df, gpu_rates=self.gpu_rates, weights=self.weights)
        feasible, excluded = apply_constraints(scores, constraints or UserConstraints())

        result = self._build_output(feasible, excluded, constraints or UserConstraints(), source="predicted")
        result["prediction_metadata"] = {
            "query_param_count": param_count,
            "query_batch_size": batch_size,
            "query_mode": mode,
            "query_family": family,
            "history_entries_used": len(history_df),
            "gpus_in_history": history_df["gpu_type"].nunique(),
        }
        return result

    # ------------------------------------------------------------------
    # Output builders
    # ------------------------------------------------------------------

    def _build_output(
        self,
        feasible: list[GpuScore],
        excluded: list[ExcludedGpu],
        constraints: UserConstraints,
        source: str,
    ) -> dict:
        recommended = feasible[0] if feasible else None
        return {
            "status": "ok" if recommended else "no_feasible_gpus",
            "source": source,
            "recommended_gpu": recommended.gpu_type if recommended else None,
            "composite_score": recommended.composite_score if recommended else 0,
            "reasoning": recommended.reasoning if recommended else "",
            "detail_lines": recommended.detail_lines if recommended else [],
            "constraints_applied": constraints.describe(),
            "rankings": [asdict(s) for s in feasible],
            "excluded": [asdict(e) for e in excluded],
            "history_stats": self.history.summary_stats(),
        }

    def _empty_result(self, message: str) -> dict:
        return {
            "status": "error",
            "message": message,
            "recommended_gpu": None,
            "rankings": [],
            "excluded": [],
        }

    def _log_to_history(self, result, workload, mode, batch_size, constraints):
        try:
            self.history.log_recommendation(
                workload=workload or "all",
                mode=mode or "all",
                batch_size=batch_size or 0,
                constraints_json=json.dumps(constraints.to_dict() if constraints else {}),
                recommended_gpu=result.get("recommended_gpu", ""),
                composite_score=result.get("composite_score", 0),
                reasoning=result.get("reasoning", ""),
                all_scores_json=json.dumps(result.get("rankings", [])),
            )
        except Exception as e:
            logger.warning("Could not log recommendation to history: %s", e)

    # ------------------------------------------------------------------
    # Import existing results into history
    # ------------------------------------------------------------------

    def import_results_to_history(
        self, results_dir: str | Path, gpu_type: str = ""
    ) -> int:
        """Import benchmark_summary CSVs into the history database."""
        raw_df = load_summary_csvs(results_dir)
        results = raw_df.to_dict("records")
        count = 0
        for r in results:
            if "error" in r:
                continue
            gt = gpu_type or r.get("gpu_type", "unknown")
            cost = self.gpu_rates.get(gt, {}).get("cost_per_hour", 0)
            self.history.log_run(
                gpu_type=gt,
                workload=r.get("workload", ""),
                mode=r.get("mode", ""),
                batch_size=int(r.get("batch_size", 0)),
                throughput=float(r.get("throughput", 0)),
                model_name=r.get("model_name", ""),
                param_count=int(r.get("param_count", 0) or 0),
                throughput_unit=r.get("throughput_unit", ""),
                latency_p50_ms=float(r.get("latency_p50_ms", 0) or 0),
                latency_p95_ms=float(r.get("latency_p95_ms", 0) or 0),
                latency_p99_ms=float(r.get("latency_p99_ms", 0) or 0),
                avg_gpu_util=float(r.get("avg_gpu_utilization_pct", 0) or 0),
                avg_gpu_mem_mb=float(r.get("avg_gpu_memory_used_mb", 0) or 0),
                cost_per_hour=cost,
                benchmark_iters=int(r.get("benchmark_iters", 0) or 0),
                seed=int(r.get("seed", 42) or 42),
                run_source="imported",
            )
            count += 1
        logger.info("Imported %d runs from %s", count, results_dir)
        return count


# ======================================================================
# Pretty-print recommendation to console
# ======================================================================

def format_recommendation(result: dict) -> str:
    """Format a recommendation dict as a human-readable report."""
    lines = []
    lines.append("")
    lines.append("=" * 66)
    lines.append("  GPU RECOMMENDATION REPORT")
    lines.append("=" * 66)

    if result.get("status") == "error":
        lines.append(f"  ERROR: {result.get('message', 'Unknown error')}")
        lines.append("=" * 66)
        return "\n".join(lines)

    src = result.get("source", "benchmark")
    lines.append(f"  Data source: {src}")
    constr = result.get("constraints_applied", "none")
    lines.append(f"  Constraints: {constr}")
    lines.append("-" * 66)

    rankings = result.get("rankings", [])
    if rankings:
        top = rankings[0]
        lines.append("")
        lines.append(f"  >>> RECOMMENDED: {top['gpu_type']}")
        lines.append(f"      Composite Score : {top['composite_score']:.4f} / 1.0000")
        lines.append(f"      Throughput      : {top['throughput']:,.2f} {top.get('throughput_unit', 'samples/sec')}")
        lines.append(f"      Latency P95     : {top['latency_p95_ms']:.2f} ms")
        if top.get("cost_per_hour", 0) > 0:
            lines.append(f"      Cost            : ${top['cost_per_hour']:.2f}/hr")
            lines.append(f"      Throughput/$    : {top['throughput_per_dollar']:,.0f} samples/$")
        lines.append(f"      Data quality    : {top.get('confidence_note', 'n/a')}")
        lines.append("")

        for detail in top.get("detail_lines", []):
            lines.append(f"      -> {detail}")

    if len(rankings) > 1:
        lines.append("")
        lines.append("-" * 66)
        lines.append("  Alternative Rankings:")
        lines.append("")
        for r in rankings[1:]:
            tp = f"{r['throughput']:,.0f}"
            cost = f"${r['cost_per_hour']:.2f}/hr" if r.get("cost_per_hour", 0) > 0 else "n/a"
            lat = f"{r['latency_p95_ms']:.1f}ms"
            lines.append(
                f"  #{r['rank']:>2}  {r['gpu_type']:<20s}  "
                f"Score: {r['composite_score']:.4f}  |  "
                f"{tp} {r.get('throughput_unit', '')}  |  {cost}  |  {lat} p95"
            )

    excluded = result.get("excluded", [])
    if excluded:
        lines.append("")
        lines.append("-" * 66)
        lines.append("  Excluded by constraints:")
        lines.append("")
        for e in excluded:
            lines.append(f"   X  {e['gpu_type']:<20s}  {e['reason']}")

    hist = result.get("history_stats", {})
    if hist:
        lines.append("")
        lines.append("-" * 66)
        lines.append(
            f"  History: {hist.get('total_runs', 0)} runs | "
            f"{hist.get('gpus_benchmarked', 0)} GPUs | "
            f"{hist.get('workloads_benchmarked', 0)} workloads"
        )

    lines.append("=" * 66)
    lines.append("")
    return "\n".join(lines)


def save_recommendation_json(result: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Recommendation saved to %s", output_path)
    return output_path
