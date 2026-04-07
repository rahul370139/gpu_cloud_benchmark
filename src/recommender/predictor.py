"""Lightweight workload-similarity predictor.

Given features of a *new* workload (param count, batch size, mode, family),
finds the K most similar workloads in the history database and predicts
throughput / latency on each GPU type via weighted-average interpolation.

No heavy ML library needed — just numpy euclidean distance on normalised
feature vectors.  This is deliberately simple; the value comes from reuse
of historical benchmark data, not from model complexity.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FAMILY_ENCODING = {
    "vision": 0,
    "nlp": 1,
    "audio": 2,
    "tabular": 3,
    "generative": 4,
}

DEFAULT_FEATURE_WEIGHTS = {
    "param_count": 0.30,
    "batch_size": 0.25,
    "memory_footprint": 0.25,
    "is_training": 0.20,
}


@dataclass
class PredictionResult:
    gpu_type: str
    predicted_throughput: float
    predicted_latency_p95: float
    predicted_gpu_mem_mb: float
    confidence: float
    similar_workloads: list[str] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)


class WorkloadPredictor:
    """K-nearest-neighbour predictor over historical benchmark runs."""

    def __init__(
        self,
        k_neighbors: int = 3,
        min_history_entries: int = 5,
        feature_weights: dict[str, float] | None = None,
    ):
        self.k = k_neighbors
        self.min_entries = min_history_entries
        self.fw = {**DEFAULT_FEATURE_WEIGHTS, **(feature_weights or {})}

    @classmethod
    def from_config(cls, cfg: dict) -> "WorkloadPredictor":
        pcfg = cfg.get("predictor", {})
        return cls(
            k_neighbors=pcfg.get("k_neighbors", 3),
            min_history_entries=pcfg.get("min_history_entries", 5),
            feature_weights=pcfg.get("feature_weights"),
        )

    def _build_feature_vector(
        self,
        param_count: int,
        batch_size: int,
        is_training: bool,
        family: str = "vision",
    ) -> np.ndarray:
        return np.array([
            np.log1p(param_count) * self.fw["param_count"],
            np.log1p(batch_size) * self.fw["batch_size"],
            # rough memory proxy: params * (1 + 2*is_training for grads+optim)
            np.log1p(param_count * (3 if is_training else 1) * batch_size / 1e6) * self.fw["memory_footprint"],
            float(is_training) * self.fw["is_training"],
        ])

    def _features_from_row(self, row) -> np.ndarray:
        return self._build_feature_vector(
            param_count=int(row.get("param_count", 0)),
            batch_size=int(row.get("batch_size", 1)),
            is_training=str(row.get("mode", "")).lower() == "training",
        )

    def predict(
        self,
        history_df: pd.DataFrame,
        param_count: int,
        batch_size: int,
        mode: str = "inference",
        family: str = "vision",
    ) -> list[PredictionResult]:
        """Predict performance on each GPU from historical data.

        Args:
            history_df: DataFrame from HistoryStore.get_all_runs().
            param_count: parameter count of the new model.
            batch_size: desired batch size.
            mode: 'inference' or 'training'.
            family: workload family for encoding.

        Returns:
            List of PredictionResult, one per GPU type, sorted by predicted
            throughput descending.
        """
        if len(history_df) < self.min_entries:
            logger.warning(
                "Insufficient history (%d entries, need %d). Cannot predict.",
                len(history_df), self.min_entries,
            )
            return []

        query_vec = self._build_feature_vector(
            param_count, batch_size, mode == "training", family,
        )

        hist = history_df.copy()
        hist = hist[hist["throughput"].notna() & (hist["throughput"] > 0)]
        if hist.empty:
            return []

        feature_matrix = np.vstack([self._features_from_row(r) for _, r in hist.iterrows()])
        distances = np.linalg.norm(feature_matrix - query_vec, axis=1)
        hist = hist.assign(_dist=distances)

        gpu_types = hist["gpu_type"].unique()
        results: list[PredictionResult] = []

        for gpu in gpu_types:
            gpu_hist = hist[hist["gpu_type"] == gpu].copy()
            if gpu_hist.empty:
                continue

            k = min(self.k, len(gpu_hist))
            nearest = gpu_hist.nsmallest(k, "_dist")

            dists = nearest["_dist"].values
            inv_dists = 1.0 / (dists + 1e-8)
            weights = inv_dists / inv_dists.sum()

            pred_tp = float(np.dot(weights, nearest["throughput"].values))
            pred_lat = float(np.dot(weights, nearest["latency_p95_ms"].fillna(0).values))
            pred_mem = float(np.dot(weights, nearest["avg_gpu_mem_mb"].fillna(0).values))

            max_dist = distances.max() if distances.max() > 0 else 1.0
            mean_dist = float(np.mean(dists))
            confidence = max(0.0, 1.0 - mean_dist / max_dist)

            results.append(PredictionResult(
                gpu_type=gpu,
                predicted_throughput=round(pred_tp, 2),
                predicted_latency_p95=round(pred_lat, 4),
                predicted_gpu_mem_mb=round(pred_mem, 2),
                confidence=round(confidence, 4),
                similar_workloads=nearest["workload"].tolist(),
                distances=[round(float(d), 4) for d in dists],
            ))

        results.sort(key=lambda r: r.predicted_throughput, reverse=True)
        return results

    def predict_with_cost(
        self,
        history_df: pd.DataFrame,
        gpu_rates: dict[str, dict],
        param_count: int,
        batch_size: int,
        mode: str = "inference",
        family: str = "vision",
    ) -> pd.DataFrame:
        """Predict and augment with cost-efficiency metrics.

        Returns a DataFrame ready for score_gpus().
        """
        preds = self.predict(history_df, param_count, batch_size, mode, family)
        if not preds:
            return pd.DataFrame()

        rows = []
        for p in preds:
            rate = gpu_rates.get(p.gpu_type, {})
            cost = rate.get("cost_per_hour", rate.get("cost_per_gpu_hour", 0))
            tpd = p.predicted_throughput * 3600 / cost if cost > 0 else 0
            rows.append({
                "gpu_type": p.gpu_type,
                "throughput": p.predicted_throughput,
                "latency_p95_ms": p.predicted_latency_p95,
                "avg_gpu_mem_mb": p.predicted_gpu_mem_mb,
                "cost_per_hour": cost,
                "throughput_per_dollar": round(tpd, 2),
                "confidence": p.confidence,
                "similar_workloads": ", ".join(p.similar_workloads),
                "is_partial": 0,
                "source": "predicted",
            })

        return pd.DataFrame(rows)
