"""Leave-one-batch-size-out KNN evaluation.

Realistic "partial-run" scenario: the user has benchmarked some batch
sizes and asks the predictor to estimate the remaining one.

For each (workload, mode, batch_size) tuple in the unified history we:
  1. Remove all rows for that exact tuple.
  2. Use the predictor over the remaining rows.
  3. Compare predicted winner / throughput / latency vs actual.

This is the metric most relevant to the project motto: recommend a GPU
for a *new* batch size *without re-running it*.
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

FAMILY_MAP = {
    "resnet50":             "vision",
    "bert_base":            "nlp",
    "example_mlp":          "tabular",
    "clip_image_embedding": "vision",
    "llm_text_generation":  "generative",
}


def main(db_path: str, out_path: str) -> None:
    engine = RecommendationEngine(history_db_path=db_path)
    predictor = WorkloadPredictor(k_neighbors=3, min_history_entries=5)
    full = engine.history.get_all_runs()

    keys = (
        full[["workload", "mode", "batch_size"]]
        .drop_duplicates()
        .to_dict("records")
    )

    rows = []
    for k in keys:
        held_out = full[
            (full["workload"] == k["workload"])
            & (full["mode"] == k["mode"])
            & (full["batch_size"] == k["batch_size"])
        ]
        held_in = full.drop(held_out.index)
        if held_in.empty or held_out.empty:
            continue

        params = int(held_out["param_count"].iloc[0])
        family = FAMILY_MAP.get(k["workload"], "vision")
        preds = predictor.predict(held_in, params, k["batch_size"], k["mode"], family)
        if not preds:
            continue

        # Actual GPU winner among GPUs that have data for this scenario
        actual_grp = held_out.groupby("gpu_type")["throughput"].mean()
        actual_gpu = actual_grp.idxmax()

        pred_gpu = preds[0].gpu_type

        # Per-GPU error
        for p in preds:
            actual_rows = held_out[held_out["gpu_type"] == p.gpu_type]
            if actual_rows.empty:
                continue
            a_tp = float(actual_rows["throughput"].mean())
            a_lat = float(actual_rows["latency_p95_ms"].mean())
            tp_err = abs(p.predicted_throughput - a_tp) / a_tp * 100 if a_tp else None
            lat_err = abs(p.predicted_latency_p95 - a_lat) / a_lat * 100 if a_lat else None
            rows.append({
                "workload": k["workload"],
                "mode": k["mode"],
                "batch_size": k["batch_size"],
                "gpu": p.gpu_type,
                "actual_winner": actual_gpu,
                "predicted_winner": pred_gpu,
                "winner_match": pred_gpu == actual_gpu,
                "predicted_throughput": p.predicted_throughput,
                "actual_throughput": round(a_tp, 2),
                "throughput_pct_err": round(tp_err, 2) if tp_err is not None else None,
                "latency_pct_err": round(lat_err, 2) if lat_err is not None else None,
                "confidence": p.confidence,
            })

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Aggregate stats per workload
    overall = {
        "scenarios_evaluated": int(df.drop_duplicates(["workload", "mode", "batch_size"]).shape[0]),
        "total_per_gpu_comparisons": int(len(df)),
        "winner_match_rate": float(
            df.drop_duplicates(["workload", "mode", "batch_size"])["winner_match"].mean()
        ),
        "median_throughput_pct_err": float(df["throughput_pct_err"].median()),
        "mean_throughput_pct_err": float(df["throughput_pct_err"].mean()),
        "median_latency_pct_err": float(df["latency_pct_err"].median()),
        "mean_latency_pct_err": float(df["latency_pct_err"].mean()),
    }

    per_workload = (
        df.groupby("workload")
        .agg(
            scenarios=("winner_match", lambda s: int(s.drop_duplicates().shape[0])),
            winner_match_rate=("winner_match", lambda s: float(s.mean())),
            median_throughput_pct_err=("throughput_pct_err", "median"),
            median_latency_pct_err=("latency_pct_err", "median"),
        )
        .round(2)
        .reset_index()
        .to_dict("records")
    )

    Path(out_path).write_text(
        json.dumps(
            {"overall": overall, "per_workload": per_workload, "rows": rows},
            indent=2,
            default=str,
        )
    )

    print(json.dumps({"overall": overall, "per_workload": per_workload}, indent=2))
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark_history_unified.db"
    out = sys.argv[2] if len(sys.argv) > 2 else "results_eval/knn_batch_loo_eval.json"
    main(db, out)
