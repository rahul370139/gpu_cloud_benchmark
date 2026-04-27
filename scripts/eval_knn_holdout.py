"""Leave-one-workload-out KNN evaluation.

For each of the 5 workloads currently in the unified history:
  1. Remove ALL rows for that workload (held-out).
  2. Use the predictor (KNN over remaining 4 workloads) to estimate
     throughput / latency for a representative scenario of the held-out workload.
  3. Compare predicted-winner vs actual-winner (computed from the held-out rows).
  4. Report MAPE on throughput and latency for each (workload, GPU) pair.

This validates the project's "no-run recommendation" claim quantitatively.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommender.engine import RecommendationEngine  # noqa: E402
from src.recommender.predictor import WorkloadPredictor  # noqa: E402

# Representative scenario + family per workload (matches the runner config)
QUERIES = [
    # (workload, param_count, batch_size, mode, family)
    ("resnet50",             25_557_032,  32, "inference", "vision"),
    ("bert_base",            109_482_240, 32, "inference", "nlp"),
    ("example_mlp",          34_052,      32, "inference", "tabular"),
    ("clip_image_embedding", 87_849_217,  8,  "inference", "vision"),
    ("llm_text_generation",  23_631_360,  4,  "inference", "generative"),
]


def actual_winner(actual_df: pd.DataFrame) -> tuple[str, float]:
    """Return (gpu_type, mean_throughput) for the highest-throughput GPU."""
    grouped = (
        actual_df.groupby("gpu_type")["throughput"]
        .mean()
        .sort_values(ascending=False)
    )
    return grouped.index[0], float(grouped.iloc[0])


def main(db_path: str, out_path: str) -> None:
    engine = RecommendationEngine(history_db_path=db_path)
    predictor = WorkloadPredictor(k_neighbors=3, min_history_entries=5)
    full_hist = engine.history.get_all_runs()

    report = {
        "history_summary": engine.history.summary_stats(),
        "evaluations": [],
        "totals": {},
    }

    matches = 0
    tp_errors = []
    lat_errors = []

    for workload, params, bs, mode, family in QUERIES:
        held_in   = full_hist[full_hist["workload"] != workload]
        held_out  = full_hist[
            (full_hist["workload"] == workload)
            & (full_hist["mode"] == mode)
            & (full_hist["batch_size"] == bs)
        ]

        if held_out.empty or held_in.empty:
            continue

        actual_gpu, actual_tp = actual_winner(held_out)

        # --- KNN prediction over held-in (4 workloads) ---
        preds = predictor.predict(held_in, params, bs, mode, family)
        pred_by_gpu = {p.gpu_type: p for p in preds}
        if not preds:
            continue

        # Predicted winner = highest predicted throughput across all GPUs in history
        pred_gpu = preds[0].gpu_type
        pred_tp = preds[0].predicted_throughput

        # Per-GPU comparisons (only for GPUs present in both held-in and held-out)
        per_gpu = []
        for gpu, p in pred_by_gpu.items():
            actual_rows = held_out[held_out["gpu_type"] == gpu]
            if actual_rows.empty:
                continue
            a_tp = float(actual_rows["throughput"].mean())
            a_lat = float(actual_rows["latency_p95_ms"].mean())
            tp_err = abs(p.predicted_throughput - a_tp) / a_tp * 100 if a_tp else None
            lat_err = abs(p.predicted_latency_p95 - a_lat) / a_lat * 100 if a_lat else None
            per_gpu.append({
                "gpu_type": gpu,
                "predicted_throughput": p.predicted_throughput,
                "actual_throughput": round(a_tp, 2),
                "throughput_pct_err": round(tp_err, 2) if tp_err is not None else None,
                "predicted_latency_p95_ms": p.predicted_latency_p95,
                "actual_latency_p95_ms": round(a_lat, 4),
                "latency_pct_err": round(lat_err, 2) if lat_err is not None else None,
                "confidence": p.confidence,
                "similar_workloads": p.similar_workloads,
            })
            if tp_err is not None:
                tp_errors.append(tp_err)
            if lat_err is not None:
                lat_errors.append(lat_err)

        winner_match = pred_gpu == actual_gpu
        if winner_match:
            matches += 1

        report["evaluations"].append({
            "workload": workload,
            "query": {"param_count": params, "batch_size": bs, "mode": mode, "family": family},
            "actual_winner": actual_gpu,
            "predicted_winner": pred_gpu,
            "winner_match": winner_match,
            "predicted_top_throughput": pred_tp,
            "actual_top_throughput": round(actual_tp, 2),
            "per_gpu": per_gpu,
        })

    n = len(report["evaluations"])
    report["totals"] = {
        "evaluated_workloads": n,
        "winner_match_count": matches,
        "winner_match_rate": round(matches / n, 4) if n else 0,
        "median_throughput_pct_err": round(pd.Series(tp_errors).median(), 2) if tp_errors else None,
        "mean_throughput_pct_err":   round(pd.Series(tp_errors).mean(), 2)   if tp_errors else None,
        "median_latency_pct_err":    round(pd.Series(lat_errors).median(), 2) if lat_errors else None,
        "mean_latency_pct_err":      round(pd.Series(lat_errors).mean(), 2)   if lat_errors else None,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps(report["totals"], indent=2))
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark_history_unified.db"
    out = sys.argv[2] if len(sys.argv) > 2 else "results_eval/knn_loo_eval.json"
    main(db, out)
