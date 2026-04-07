"""CLI entry point: python -m src.recommender <mode> [options]

Modes:
    recommend   Analyse existing benchmark results and produce GPU recommendation
    partial     Run short benchmarks with early stopping, then recommend
    predict     Estimate GPU performance from history (no benchmarking)
    import      Import existing results into the history database
    history     Show history database summary
"""

import argparse
import json
import logging
import sys

from .engine import RecommendationEngine, format_recommendation, save_recommendation_json
from .constraints import UserConstraints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("recommender")


def _build_constraints(args) -> UserConstraints:
    return UserConstraints(
        max_cost_per_hour=getattr(args, "max_cost", None),
        max_latency_p95_ms=getattr(args, "max_latency", None),
        min_throughput=getattr(args, "min_throughput", None),
    )


def cmd_recommend(args):
    engine = RecommendationEngine(
        config_path=args.rec_config,
        cost_rates_path=args.cost_rates,
    )
    result = engine.recommend(
        results_dir=args.results_dir,
        constraints=_build_constraints(args),
        workload=args.workload,
        mode=args.mode,
        batch_size=args.batch_size,
    )
    print(format_recommendation(result))
    if args.output:
        save_recommendation_json(result, args.output)


def cmd_partial(args):
    engine = RecommendationEngine(
        config_path=args.rec_config,
        cost_rates_path=args.cost_rates,
    )
    result = engine.partial_and_recommend(
        benchmark_config_path=args.benchmark_config,
        constraints=_build_constraints(args),
        device=args.device,
    )
    print(format_recommendation(result))
    if args.output:
        save_recommendation_json(result, args.output)


def cmd_predict(args):
    engine = RecommendationEngine(
        config_path=args.rec_config,
        cost_rates_path=args.cost_rates,
    )
    result = engine.predict_and_recommend(
        param_count=args.param_count,
        batch_size=args.batch_size,
        mode=args.mode or "inference",
        family=args.family or "vision",
        constraints=_build_constraints(args),
    )
    print(format_recommendation(result))
    if args.output:
        save_recommendation_json(result, args.output)


def cmd_import(args):
    engine = RecommendationEngine(
        config_path=args.rec_config,
        cost_rates_path=args.cost_rates,
    )
    count = engine.import_results_to_history(args.results_dir, gpu_type=args.gpu_type or "")
    print(f"Imported {count} benchmark runs into history.")


def cmd_history(args):
    engine = RecommendationEngine(
        config_path=args.rec_config,
        cost_rates_path=args.cost_rates,
    )
    stats = engine.history.summary_stats()
    print()
    print("=" * 50)
    print("  Benchmark History Summary")
    print("=" * 50)
    print(f"  Total runs : {stats['total_runs']}")
    print(f"  GPUs       : {', '.join(stats['distinct_gpus']) or 'none'}")
    print(f"  Workloads  : {', '.join(stats['distinct_workloads']) or 'none'}")
    print("=" * 50)
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src.recommender",
        description="GPU Recommendation Engine — find the best GPU for your ML workload",
    )
    parser.add_argument("--rec-config", default="config/recommendation_config.yaml")
    parser.add_argument("--cost-rates", default="config/gpu_cost_rates.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- recommend ---
    p_rec = sub.add_parser("recommend", help="Recommend from existing results")
    p_rec.add_argument("--results-dir", default="results")
    p_rec.add_argument("--workload", default=None)
    p_rec.add_argument("--mode", default=None)
    p_rec.add_argument("--batch-size", type=int, default=None)
    p_rec.add_argument("--max-cost", type=float, default=None, help="Max $/hr")
    p_rec.add_argument("--max-latency", type=float, default=None, help="Max P95 latency in ms")
    p_rec.add_argument("--min-throughput", type=float, default=None, help="Min throughput")
    p_rec.add_argument("-o", "--output", default="", help="Save recommendation JSON to this path")

    # --- partial ---
    p_par = sub.add_parser("partial", help="Run partial benchmarks then recommend")
    p_par.add_argument("--benchmark-config", default="config/benchmark_config.yaml")
    p_par.add_argument("--device", default=None)
    p_par.add_argument("--max-cost", type=float, default=None)
    p_par.add_argument("--max-latency", type=float, default=None)
    p_par.add_argument("--min-throughput", type=float, default=None)
    p_par.add_argument("-o", "--output", default="", help="Save recommendation JSON to this path")

    # --- predict ---
    p_pred = sub.add_parser("predict", help="Predict from history (no GPU needed)")
    p_pred.add_argument("--param-count", type=int, required=True, help="Model parameter count")
    p_pred.add_argument("--batch-size", type=int, required=True)
    p_pred.add_argument("--mode", default="inference")
    p_pred.add_argument("--family", default="vision", choices=["vision", "nlp", "audio", "tabular", "generative"])
    p_pred.add_argument("--max-cost", type=float, default=None)
    p_pred.add_argument("--max-latency", type=float, default=None)
    p_pred.add_argument("--min-throughput", type=float, default=None)
    p_pred.add_argument("-o", "--output", default="", help="Save recommendation JSON to this path")

    # --- import ---
    p_imp = sub.add_parser("import", help="Import existing results into history DB")
    p_imp.add_argument("--results-dir", default="results")
    p_imp.add_argument("--gpu-type", default="")

    # --- history ---
    sub.add_parser("history", help="Show history database summary")

    args = parser.parse_args()

    dispatch = {
        "recommend": cmd_recommend,
        "partial": cmd_partial,
        "predict": cmd_predict,
        "import": cmd_import,
        "history": cmd_history,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
