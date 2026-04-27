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
from .scorer import score_gpus, GpuScore, DEFAULT_WEIGHTS
from .constraints import UserConstraints, apply_constraints, ExcludedGpu
from .history import HistoryStore
from .partial import PartialProfiler, PartialResult
from .predictor import WorkloadPredictor

logger = logging.getLogger(__name__)


def _throughput_quantity_unit(unit: str | None) -> str:
    if not unit:
        return "samples"
    return unit[:-4] if unit.endswith("/sec") else unit


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

        result = self._recommend_from_aggregate_df(
            agg_df=agg_df,
            constraints=constraints or UserConstraints(),
            source="benchmark",
        )
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

    def _effective_weights(self) -> dict[str, float]:
        weights = {**DEFAULT_WEIGHTS, **(self.weights or {})}
        total = weights["throughput"] + weights["cost_efficiency"] + weights["latency"]
        return {key: value / total for key, value in weights.items()}

    def _composite_score_explanation(self, multi_scenario: bool) -> str:
        weights = self._effective_weights()
        base = (
            "Composite score is normalized within each comparison set as "
            f"{weights['throughput'] * 100:.0f}% throughput + "
            f"{weights['cost_efficiency'] * 100:.0f}% throughput-per-dollar + "
            f"{weights['latency'] * 100:.0f}% inverse P95 latency."
        )
        if multi_scenario:
            return (
                f"{base} For workload recommendations across multiple batch sizes, "
                "the per-scenario scores are averaged across the tested batch sizes."
            )
        return base

    def _comparison_context(self, df: pd.DataFrame) -> dict:
        units = sorted({unit for unit in df.get("throughput_unit", pd.Series(dtype=str)).dropna().tolist() if unit})
        return {
            "workloads": sorted(df["workload"].dropna().unique().tolist()),
            "modes": sorted(df["mode"].dropna().unique().tolist()),
            "batch_sizes": sorted(int(v) for v in df["batch_size"].dropna().unique().tolist()),
            "throughput_units": units,
        }

    def _recommend_from_aggregate_df(
        self,
        agg_df: pd.DataFrame,
        constraints: UserConstraints,
        source: str,
    ) -> dict:
        context = self._comparison_context(agg_df)
        workload_count = len(context["workloads"])
        mode_count = len(context["modes"])
        batch_size_count = len(context["batch_sizes"])
        unit_count = len(context["throughput_units"])

        # A single workload/mode/batch-size scenario is directly comparable.
        if workload_count == 1 and mode_count == 1 and batch_size_count == 1:
            scores = score_gpus(agg_df, gpu_rates=self.gpu_rates, weights=self.weights)
            feasible, excluded = apply_constraints(scores, constraints)
            result = self._build_output(feasible, excluded, constraints, source=source)
            result["recommendation_scope"] = "scenario"
            result["comparison_context"] = context
            result["score_weights"] = self._effective_weights()
            result["composite_score_explanation"] = self._composite_score_explanation(multi_scenario=False)
            return result

        workload_recommendations, scenario_exclusions = self._build_workload_recommendations(
            agg_df,
            constraints,
        )
        if not workload_recommendations:
            return {
                "status": "no_feasible_gpus",
                "source": source,
                "recommended_gpu": None,
                "composite_score": None,
                "reasoning": "",
                "detail_lines": [],
                "constraints_applied": constraints.describe(),
                "rankings": [],
                "excluded": scenario_exclusions,
                "history_stats": self.history.summary_stats(),
                "recommendation_scope": "mixed_suite" if workload_count > 1 or mode_count > 1 or unit_count > 1 else "workload_mode",
                "comparison_context": context,
                "score_weights": self._effective_weights(),
                "composite_score_explanation": self._composite_score_explanation(multi_scenario=True),
                "summary_note": "No GPU satisfied the constraints across the evaluated scenarios.",
                "workload_recommendations": [],
            }

        single_workload_mode = workload_count == 1 and mode_count == 1
        if single_workload_mode:
            top = workload_recommendations[0]
            return {
                "status": "ok",
                "source": source,
                "recommended_gpu": top["recommended_gpu"],
                "composite_score": top["avg_composite_score"],
                "reasoning": top["summary"],
                "detail_lines": top["detail_lines"],
                "constraints_applied": constraints.describe(),
                "rankings": top["rankings"],
                "excluded": scenario_exclusions,
                "history_stats": self.history.summary_stats(),
                "recommendation_scope": "workload_mode",
                "comparison_context": context,
                "score_weights": self._effective_weights(),
                "composite_score_explanation": self._composite_score_explanation(multi_scenario=True),
                "summary_note": (
                    f"Recommendation aggregated across {top['scenarios_total']} tested batch sizes "
                    f"for {top['workload']} / {top['mode']}."
                ),
                "workload_recommendations": workload_recommendations,
            }

        return {
            "status": "ok",
            "source": source,
            "recommended_gpu": None,
            "composite_score": None,
            "reasoning": "",
            "detail_lines": [],
            "constraints_applied": constraints.describe(),
            "rankings": [],
            "excluded": scenario_exclusions,
            "history_stats": self.history.summary_stats(),
            "recommendation_scope": "mixed_suite",
            "comparison_context": context,
            "score_weights": self._effective_weights(),
            "composite_score_explanation": self._composite_score_explanation(multi_scenario=True),
            "summary_note": (
                "This benchmark suite spans multiple workloads, modes, or throughput units. "
                "Use the workload-level recommendations instead of a single global GPU choice."
            ),
            "workload_recommendations": workload_recommendations,
        }

    def _build_workload_recommendations(
        self,
        agg_df: pd.DataFrame,
        constraints: UserConstraints,
    ) -> tuple[list[dict], list[dict]]:
        recommendations: list[dict] = []
        scenario_exclusions: list[dict] = []

        for (workload, mode), workload_df in agg_df.groupby(["workload", "mode"], sort=True):
            batch_sizes = sorted(int(v) for v in workload_df["batch_size"].dropna().unique().tolist())
            gpu_types = sorted(workload_df["gpu_type"].dropna().unique().tolist())

            per_gpu = {
                gpu: {
                    "gpu_type": gpu,
                    "scenario_score_sum": 0.0,
                    "scenario_wins": 0,
                    "throughput_wins": 0,
                    "latency_wins": 0,
                    "value_wins": 0,
                    "feasible_scenarios": 0,
                    "throughput_values": [],
                    "throughput_per_dollar_values": [],
                    "latency_values": [],
                    "cost_per_hour": 0.0,
                    "throughput_unit": "",
                }
                for gpu in gpu_types
            }

            for batch_size, scenario_df in workload_df.groupby("batch_size", sort=True):
                scores = score_gpus(scenario_df, gpu_rates=self.gpu_rates, weights=self.weights)
                feasible, excluded = apply_constraints(scores, constraints)

                for excluded_gpu in excluded:
                    scenario_exclusions.append({
                        "gpu_type": excluded_gpu.gpu_type,
                        "reason": f"{workload}/{mode}/bs={int(batch_size)}: {excluded_gpu.reason}",
                    })

                if not feasible:
                    continue

                best_throughput = max(score.throughput for score in feasible)
                best_latency = min(score.latency_p95_ms for score in feasible)
                best_value = max(score.throughput_per_dollar for score in feasible)

                for score in feasible:
                    gpu_entry = per_gpu[score.gpu_type]
                    gpu_entry["scenario_score_sum"] += score.composite_score
                    gpu_entry["feasible_scenarios"] += 1
                    gpu_entry["throughput_values"].append(score.throughput)
                    gpu_entry["throughput_per_dollar_values"].append(score.throughput_per_dollar)
                    gpu_entry["latency_values"].append(score.latency_p95_ms)
                    gpu_entry["cost_per_hour"] = score.cost_per_hour
                    gpu_entry["throughput_unit"] = score.throughput_unit

                    if score.rank == 1:
                        gpu_entry["scenario_wins"] += 1
                    if score.throughput == best_throughput:
                        gpu_entry["throughput_wins"] += 1
                    if score.latency_p95_ms == best_latency:
                        gpu_entry["latency_wins"] += 1
                    if score.throughput_per_dollar == best_value:
                        gpu_entry["value_wins"] += 1

            scenarios_total = len(batch_sizes)
            rankings = []
            for gpu, gpu_entry in per_gpu.items():
                feasible_scenarios = gpu_entry["feasible_scenarios"]
                if feasible_scenarios == 0:
                    avg_throughput = 0.0
                    mean_tpd = 0.0
                    median_latency = None
                else:
                    avg_throughput = sum(gpu_entry["throughput_values"]) / feasible_scenarios
                    mean_tpd = sum(gpu_entry["throughput_per_dollar_values"]) / feasible_scenarios
                    median_latency = float(pd.Series(gpu_entry["latency_values"]).median())

                rankings.append({
                    "gpu_type": gpu,
                    "avg_composite_score": round(gpu_entry["scenario_score_sum"] / scenarios_total, 4) if scenarios_total else 0.0,
                    "scenario_wins": gpu_entry["scenario_wins"],
                    "throughput_wins": gpu_entry["throughput_wins"],
                    "latency_wins": gpu_entry["latency_wins"],
                    "value_wins": gpu_entry["value_wins"],
                    "feasible_scenarios": feasible_scenarios,
                    "scenarios_total": scenarios_total,
                    "mean_throughput_across_batch_sizes": round(avg_throughput, 2),
                    "mean_throughput_per_dollar_across_batch_sizes": round(mean_tpd, 2),
                    "median_latency_p95_ms": round(median_latency, 4) if median_latency is not None else None,
                    "throughput_unit": gpu_entry["throughput_unit"],
                    "cost_per_hour": round(gpu_entry["cost_per_hour"], 4),
                })

            rankings = sorted(
                rankings,
                key=lambda row: (
                    row["avg_composite_score"],
                    row["scenario_wins"],
                    row["value_wins"],
                    row["mean_throughput_per_dollar_across_batch_sizes"],
                ),
                reverse=True,
            )
            for idx, ranking in enumerate(rankings, start=1):
                ranking["rank"] = idx

            if not rankings:
                continue

            top = rankings[0]
            if top["feasible_scenarios"] == 0:
                continue

            detail_lines = [
                f"Scenario wins: {top['scenario_wins']}/{scenarios_total}",
                f"Throughput wins: {top['throughput_wins']}/{scenarios_total}",
                f"Value wins: {top['value_wins']}/{scenarios_total}",
                f"Latency wins: {top['latency_wins']}/{scenarios_total}",
            ]
            if top["median_latency_p95_ms"] is not None:
                detail_lines.append(f"Median P95 latency across tested batch sizes: {top['median_latency_p95_ms']:.1f} ms")
            if top["mean_throughput_per_dollar_across_batch_sizes"] > 0:
                detail_lines.append(
                    "Mean throughput-per-dollar across tested batch sizes: "
                    f"{top['mean_throughput_per_dollar_across_batch_sizes']:,.0f} "
                    f"{_throughput_quantity_unit(top.get('throughput_unit'))}/$"
                )

            recommendations.append({
                "workload": workload,
                "mode": mode,
                "batch_sizes_evaluated": batch_sizes,
                "throughput_unit": top.get("throughput_unit") or "samples/sec",
                "recommended_gpu": top["gpu_type"],
                "avg_composite_score": top["avg_composite_score"],
                "scenario_wins": top["scenario_wins"],
                "throughput_wins": top["throughput_wins"],
                "value_wins": top["value_wins"],
                "latency_wins": top["latency_wins"],
                "scenarios_total": scenarios_total,
                "mean_throughput_across_batch_sizes": top["mean_throughput_across_batch_sizes"],
                "mean_throughput_per_dollar_across_batch_sizes": top["mean_throughput_per_dollar_across_batch_sizes"],
                "median_latency_p95_ms": top["median_latency_p95_ms"],
                "cost_per_hour": top["cost_per_hour"],
                "summary": f"Best average scenario score across {scenarios_total} tested batch sizes",
                "detail_lines": detail_lines,
                "rankings": rankings,
            })

        recommendations.sort(key=lambda item: (item["workload"], item["mode"]))
        return recommendations, scenario_exclusions

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
    if result.get("summary_note"):
        lines.append(f"  Note: {result['summary_note']}")
    if result.get("composite_score_explanation"):
        lines.append(f"  Score: {result['composite_score_explanation']}")
    lines.append("-" * 66)

    scope = result.get("recommendation_scope", "scenario")
    rankings = result.get("rankings", [])
    workload_recommendations = result.get("workload_recommendations", [])

    if scope == "mixed_suite" and workload_recommendations:
        lines.append("")
        lines.append("  Workload-Level Recommendations:")
        for recommendation in workload_recommendations:
            lines.append("")
            lines.append(f"  {recommendation['workload']} / {recommendation['mode']}")
            lines.append(f"      Recommended GPU : {recommendation['recommended_gpu']}")
            lines.append(f"      Avg Score       : {recommendation['avg_composite_score']:.4f} / 1.0000")
            lines.append(
                "      Batch Sizes     : "
                + ", ".join(str(bs) for bs in recommendation.get("batch_sizes_evaluated", []))
            )
            lines.append(
                f"      Scenario Wins   : {recommendation['scenario_wins']}/{recommendation['scenarios_total']}"
            )
            lines.append(
                f"      Throughput Wins : {recommendation['throughput_wins']}/{recommendation['scenarios_total']}"
            )
            lines.append(
                f"      Value Wins      : {recommendation['value_wins']}/{recommendation['scenarios_total']}"
            )
            lines.append(
                f"      Latency Wins    : {recommendation['latency_wins']}/{recommendation['scenarios_total']}"
            )
            if recommendation.get("mean_throughput_per_dollar_across_batch_sizes", 0) > 0:
                lines.append(
                    "      Mean Throughput/$ : "
                    f"{recommendation['mean_throughput_per_dollar_across_batch_sizes']:,.0f} "
                    f"{_throughput_quantity_unit(recommendation.get('throughput_unit'))}/$"
                )
            if recommendation.get("median_latency_p95_ms") is not None:
                lines.append(
                    f"      Median P95        : {recommendation['median_latency_p95_ms']:.2f} ms"
                )

    elif rankings:
        top = rankings[0]
        lines.append("")
        lines.append(f"  >>> RECOMMENDED: {top['gpu_type']}")
        score_label = "Composite Score"
        throughput_label = "Throughput"
        latency_label = "Latency P95"
        cost_label = "Cost"
        value_label = "Throughput/$"
        data_quality_label = "Data quality"

        if scope == "workload_mode":
            score_label = "Avg Scenario Score"
            throughput_label = "Mean Throughput"
            latency_label = "Median P95"
            value_label = "Mean Throughput/$"

        score_value = top.get("composite_score", top.get("avg_composite_score", 0))
        lines.append(f"      {score_label:<16}: {score_value:.4f} / 1.0000")
        if "throughput" in top:
            lines.append(f"      {throughput_label:<16}: {top['throughput']:,.2f} {top.get('throughput_unit', 'samples/sec')}")
            lines.append(f"      {latency_label:<16}: {top['latency_p95_ms']:.2f} ms")
            if top.get("cost_per_hour", 0) > 0:
                lines.append(f"      {cost_label:<16}: ${top['cost_per_hour']:.2f}/hr")
                lines.append(
                    f"      {value_label:<16}: {top['throughput_per_dollar']:,.0f} "
                    f"{_throughput_quantity_unit(top.get('throughput_unit'))}/$"
                )
            lines.append(f"      {data_quality_label:<16}: {top.get('confidence_note', 'n/a')}")
        else:
            lines.append(
                f"      {throughput_label:<16}: {top.get('mean_throughput_across_batch_sizes', 0):,.2f} "
                f"{top.get('throughput_unit', 'samples/sec')}"
            )
            if top.get("median_latency_p95_ms") is not None:
                lines.append(f"      {latency_label:<16}: {top['median_latency_p95_ms']:.2f} ms")
            if top.get("cost_per_hour", 0) > 0:
                lines.append(f"      {cost_label:<16}: ${top['cost_per_hour']:.2f}/hr")
                lines.append(
                    f"      {value_label:<16}: "
                    f"{top.get('mean_throughput_per_dollar_across_batch_sizes', 0):,.0f} "
                    f"{_throughput_quantity_unit(top.get('throughput_unit'))}/$"
                )
            lines.append(
                "      Scenario Wins   : "
                f"{top.get('scenario_wins', 0)}/{top.get('scenarios_total', 0)}"
            )
        lines.append("")

        detail_lines = top.get("detail_lines") or result.get("detail_lines", [])
        for detail in detail_lines:
            lines.append(f"      -> {detail}")

    if scope != "mixed_suite" and len(rankings) > 1:
        lines.append("")
        lines.append("-" * 66)
        lines.append("  Alternative Rankings:")
        lines.append("")
        for r in rankings[1:]:
            if "throughput" in r:
                tp = f"{r['throughput']:,.0f}"
                cost = f"${r['cost_per_hour']:.2f}/hr" if r.get("cost_per_hour", 0) > 0 else "n/a"
                lat = f"{r['latency_p95_ms']:.1f}ms"
                score_value = r["composite_score"]
                lines.append(
                    f"  #{r['rank']:>2}  {r['gpu_type']:<20s}  "
                    f"Score: {score_value:.4f}  |  "
                    f"{tp} {r.get('throughput_unit', '')}  |  {cost}  |  {lat} p95"
                )
            else:
                cost = f"${r['cost_per_hour']:.2f}/hr" if r.get("cost_per_hour", 0) > 0 else "n/a"
                lat = (
                    f"{r['median_latency_p95_ms']:.1f}ms"
                    if r.get("median_latency_p95_ms") is not None
                    else "n/a"
                )
                lines.append(
                    f"  #{r['rank']:>2}  {r['gpu_type']:<20s}  "
                    f"Score: {r['avg_composite_score']:.4f}  |  "
                    f"{r.get('scenario_wins', 0)}/{r.get('scenarios_total', 0)} wins  |  "
                    f"{cost}  |  {lat} median p95"
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
