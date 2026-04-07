"""Tests for the GPU recommendation engine and all its components."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.recommender.history import HistoryStore
from src.recommender.scorer import score_gpus, GpuScore, _min_max_normalise
from src.recommender.constraints import UserConstraints, apply_constraints, ExcludedGpu
from src.recommender.predictor import WorkloadPredictor, PredictionResult
from src.recommender.engine import RecommendationEngine, format_recommendation


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_history.db"
    store = HistoryStore(db_path)
    yield store
    store.close()


@pytest.fixture
def populated_db(tmp_db):
    """DB with 10 runs: 2 GPUs x 2 workloads x inference + some training."""
    data = [
        ("T4", "resnet50", "inference", 32, 450.0, "images/sec", 25_600_000, 12.5, 13.0, 14.0, 85.0, 4200.0),
        ("T4", "resnet50", "inference", 1, 55.0, "images/sec", 25_600_000, 18.0, 19.0, 20.0, 30.0, 1200.0),
        ("T4", "bert_base", "inference", 32, 12000.0, "tokens/sec", 109_500_000, 1.4, 1.6, 1.8, 90.0, 6800.0),
        ("T4", "resnet50", "training", 32, 180.0, "images/sec", 25_600_000, 28.0, 30.0, 33.0, 95.0, 8500.0),
        ("T4", "bert_base", "training", 32, 4500.0, "tokens/sec", 109_500_000, 3.5, 4.0, 4.5, 92.0, 9200.0),
        ("A100", "resnet50", "inference", 32, 2800.0, "images/sec", 25_600_000, 3.5, 4.0, 4.5, 60.0, 12000.0),
        ("A100", "resnet50", "inference", 1, 350.0, "images/sec", 25_600_000, 5.0, 5.5, 6.0, 15.0, 3500.0),
        ("A100", "bert_base", "inference", 32, 65000.0, "tokens/sec", 109_500_000, 0.3, 0.4, 0.5, 55.0, 18000.0),
        ("A100", "resnet50", "training", 32, 1200.0, "images/sec", 25_600_000, 8.0, 9.0, 10.0, 75.0, 22000.0),
        ("A100", "bert_base", "training", 32, 25000.0, "tokens/sec", 109_500_000, 0.8, 1.0, 1.2, 70.0, 25000.0),
    ]
    for gpu, wl, mode, bs, tp, tu, pc, p50, p95, p99, util, mem in data:
        tmp_db.log_run(
            gpu_type=gpu, workload=wl, mode=mode, batch_size=bs,
            throughput=tp, throughput_unit=tu, param_count=pc,
            latency_p50_ms=p50, latency_p95_ms=p95, latency_p99_ms=p99,
            avg_gpu_util=util, avg_gpu_mem_mb=mem,
            cost_per_hour=0.526 if gpu == "T4" else 4.10,
        )
    return tmp_db


@pytest.fixture
def sample_benchmark_df():
    """DataFrame mimicking compute_aggregate_stats output."""
    return pd.DataFrame([
        {"gpu_type": "T4", "workload": "resnet50", "mode": "inference", "batch_size": 32,
         "mean_throughput": 450.0, "mean_latency_p95": 13.0, "mean_gpu_util": 85.0,
         "mean_gpu_mem": 4200.0, "throughput_unit": "images/sec"},
        {"gpu_type": "A100", "workload": "resnet50", "mode": "inference", "batch_size": 32,
         "mean_throughput": 2800.0, "mean_latency_p95": 4.0, "mean_gpu_util": 60.0,
         "mean_gpu_mem": 12000.0, "throughput_unit": "images/sec"},
        {"gpu_type": "V100", "workload": "resnet50", "mode": "inference", "batch_size": 32,
         "mean_throughput": 950.0, "mean_latency_p95": 9.0, "mean_gpu_util": 70.0,
         "mean_gpu_mem": 7200.0, "throughput_unit": "images/sec"},
    ])


@pytest.fixture
def gpu_rates():
    return {
        "T4": {"instance_type": "g4dn.xlarge", "cost_per_hour": 0.526, "gpu_memory_gb": 16},
        "V100": {"instance_type": "p3.2xlarge", "cost_per_hour": 3.06, "gpu_memory_gb": 16},
        "A100": {"instance_type": "p4d.24xlarge", "cost_per_hour": 4.10, "gpu_memory_gb": 40},
    }


# ======================================================================
# History Store Tests
# ======================================================================

class TestHistoryStore:

    def test_creates_database(self, tmp_db):
        assert tmp_db.db_path.exists()
        assert tmp_db.get_run_count() == 0

    def test_log_and_retrieve_run(self, tmp_db):
        row_id = tmp_db.log_run(
            gpu_type="T4", workload="resnet50", mode="inference",
            batch_size=32, throughput=500.0,
            latency_p95_ms=12.0, avg_gpu_util=85.0,
        )
        assert row_id == 1
        assert tmp_db.get_run_count() == 1
        df = tmp_db.get_all_runs()
        assert len(df) == 1
        assert df.iloc[0]["gpu_type"] == "T4"
        assert df.iloc[0]["throughput"] == 500.0

    def test_bulk_log(self, tmp_db):
        results = [
            {"workload": "resnet50", "mode": "inference", "batch_size": 32,
             "throughput": 500.0, "latency_p50_ms": 10.0, "latency_p95_ms": 12.0,
             "latency_p99_ms": 15.0, "avg_gpu_utilization_pct": 80.0,
             "avg_gpu_memory_used_mb": 4000.0, "benchmark_iters": 100, "seed": 42},
            {"workload": "bert_base", "mode": "inference", "batch_size": 8,
             "throughput": 8000.0, "error": "OOM"},
        ]
        count = tmp_db.log_benchmark_results(results, gpu_type="T4")
        assert count == 1  # second one has "error" key

    def test_distinct_gpus(self, populated_db):
        gpus = populated_db.get_distinct_gpus()
        assert set(gpus) == {"T4", "A100"}

    def test_distinct_workloads(self, populated_db):
        wls = populated_db.get_distinct_workloads()
        assert set(wls) == {"resnet50", "bert_base"}

    def test_query_by_workload(self, populated_db):
        df = populated_db.get_runs_for_workload("resnet50")
        assert len(df) >= 4

    def test_summary_stats(self, populated_db):
        stats = populated_db.summary_stats()
        assert stats["total_runs"] == 10
        assert stats["gpus_benchmarked"] == 2
        assert stats["workloads_benchmarked"] == 2

    def test_log_recommendation(self, tmp_db):
        row_id = tmp_db.log_recommendation(
            workload="resnet50", mode="inference", batch_size=32,
            constraints_json='{"max_cost": 5.0}', recommended_gpu="T4",
            composite_score=0.87, reasoning="Best cost-efficiency",
            all_scores_json="[]",
        )
        assert row_id >= 1


# ======================================================================
# Scorer Tests
# ======================================================================

class TestScorer:

    def test_min_max_normalise_higher_better(self):
        s = pd.Series([100, 200, 300])
        normed = _min_max_normalise(s, higher_is_better=True)
        assert normed.iloc[0] == pytest.approx(0.0)
        assert normed.iloc[2] == pytest.approx(1.0)

    def test_min_max_normalise_lower_better(self):
        s = pd.Series([10, 20, 30])
        normed = _min_max_normalise(s, higher_is_better=False)
        assert normed.iloc[0] == pytest.approx(1.0)
        assert normed.iloc[2] == pytest.approx(0.0)

    def test_min_max_normalise_all_equal(self):
        s = pd.Series([5.0, 5.0, 5.0])
        normed = _min_max_normalise(s, higher_is_better=True)
        assert all(normed == 1.0)

    def test_score_gpus_basic(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        assert len(scores) == 3
        assert all(isinstance(s, GpuScore) for s in scores)
        assert scores[0].rank == 1
        assert scores[0].composite_score >= scores[1].composite_score

    def test_score_gpus_adds_cost(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        for s in scores:
            assert s.cost_per_hour > 0
            assert s.throughput_per_dollar > 0

    def test_highest_throughput_per_dollar_is_t4(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        t4 = [s for s in scores if s.gpu_type == "T4"][0]
        assert t4.throughput_per_dollar > 0
        a100 = [s for s in scores if s.gpu_type == "A100"][0]
        assert t4.throughput_per_dollar > a100.throughput_per_dollar

    def test_score_with_custom_weights(self, sample_benchmark_df, gpu_rates):
        heavy_tp = {"throughput": 0.9, "cost_efficiency": 0.05, "latency": 0.05}
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates, weights=heavy_tp)
        assert scores[0].gpu_type == "A100"  # A100 has highest raw throughput

    def test_score_single_gpu(self):
        df = pd.DataFrame([{
            "gpu_type": "T4", "throughput": 500.0, "latency_p95_ms": 12.0,
        }])
        scores = score_gpus(df)
        assert len(scores) == 1
        assert scores[0].composite_score == pytest.approx(1.0)

    def test_reasoning_not_empty(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        assert scores[0].reasoning != ""
        assert len(scores[0].detail_lines) > 0


# ======================================================================
# Constraints Tests
# ======================================================================

class TestConstraints:

    def test_no_constraints_passes_all(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        feasible, excluded = apply_constraints(scores, UserConstraints())
        assert len(feasible) == 3
        assert len(excluded) == 0

    def test_cost_constraint_excludes_expensive(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        constraints = UserConstraints(max_cost_per_hour=1.0)
        feasible, excluded = apply_constraints(scores, constraints)
        feasible_names = {s.gpu_type for s in feasible}
        excluded_names = {e.gpu_type for e in excluded}
        assert "T4" in feasible_names
        assert "A100" in excluded_names
        assert "V100" in excluded_names

    def test_latency_constraint(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        constraints = UserConstraints(max_latency_p95_ms=10.0)
        feasible, excluded = apply_constraints(scores, constraints)
        for s in feasible:
            assert s.latency_p95_ms <= 10.0
        for e in excluded:
            assert "exceeds" in e.reason

    def test_throughput_constraint(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        constraints = UserConstraints(min_throughput=1000.0)
        feasible, excluded = apply_constraints(scores, constraints)
        for s in feasible:
            assert s.throughput >= 1000.0

    def test_all_excluded_returns_empty(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        constraints = UserConstraints(max_cost_per_hour=0.01)
        feasible, excluded = apply_constraints(scores, constraints)
        assert len(feasible) == 0
        assert len(excluded) == 3

    def test_describe(self):
        c = UserConstraints(max_cost_per_hour=2.0, max_latency_p95_ms=50.0)
        desc = c.describe()
        assert "$2.00" in desc
        assert "50.0 ms" in desc

    def test_re_ranking_after_filter(self, sample_benchmark_df, gpu_rates):
        scores = score_gpus(sample_benchmark_df, gpu_rates=gpu_rates)
        constraints = UserConstraints(max_cost_per_hour=3.5)
        feasible, _ = apply_constraints(scores, constraints)
        ranks = [s.rank for s in feasible]
        assert ranks == list(range(1, len(feasible) + 1))


# ======================================================================
# Predictor Tests
# ======================================================================

class TestPredictor:

    def test_insufficient_history_returns_empty(self):
        predictor = WorkloadPredictor(k_neighbors=3, min_history_entries=5)
        df = pd.DataFrame([
            {"gpu_type": "T4", "workload": "resnet50", "mode": "inference",
             "batch_size": 32, "throughput": 500.0, "param_count": 25_600_000,
             "latency_p95_ms": 12.0, "avg_gpu_mem_mb": 4000.0},
        ])
        results = predictor.predict(df, param_count=25_600_000, batch_size=32)
        assert results == []

    def test_prediction_with_enough_history(self, populated_db):
        predictor = WorkloadPredictor(k_neighbors=3, min_history_entries=5)
        history_df = populated_db.get_all_runs()
        results = predictor.predict(
            history_df, param_count=25_600_000, batch_size=32, mode="inference",
        )
        assert len(results) > 0
        for r in results:
            assert isinstance(r, PredictionResult)
            assert r.predicted_throughput > 0
            assert 0.0 <= r.confidence <= 1.0

    def test_prediction_returns_both_gpus(self, populated_db):
        predictor = WorkloadPredictor(k_neighbors=3, min_history_entries=5)
        history_df = populated_db.get_all_runs()
        results = predictor.predict(
            history_df, param_count=25_600_000, batch_size=32, mode="inference",
        )
        gpu_types = {r.gpu_type for r in results}
        assert "T4" in gpu_types
        assert "A100" in gpu_types

    def test_predict_with_cost(self, populated_db, gpu_rates):
        predictor = WorkloadPredictor(k_neighbors=3, min_history_entries=5)
        history_df = populated_db.get_all_runs()
        df = predictor.predict_with_cost(
            history_df, gpu_rates,
            param_count=25_600_000, batch_size=32, mode="inference",
        )
        assert not df.empty
        assert "throughput" in df.columns
        assert "cost_per_hour" in df.columns
        assert "throughput_per_dollar" in df.columns

    def test_feature_vector_shape(self):
        predictor = WorkloadPredictor()
        vec = predictor._build_feature_vector(
            param_count=25_600_000, batch_size=32, is_training=False,
        )
        assert vec.shape == (4,)
        assert all(np.isfinite(vec))

    def test_training_mode_changes_features(self):
        predictor = WorkloadPredictor()
        vec_inf = predictor._build_feature_vector(25_600_000, 32, is_training=False)
        vec_train = predictor._build_feature_vector(25_600_000, 32, is_training=True)
        assert not np.allclose(vec_inf, vec_train)


# ======================================================================
# Engine Tests
# ======================================================================

class TestEngine:

    def test_recommend_from_results(self, tmp_path, gpu_rates):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        df = pd.DataFrame([
            {"gpu_type": "T4", "workload": "resnet50", "mode": "inference",
             "batch_size": 32, "throughput": 500.0, "latency_p50_ms": 10.0,
             "latency_p95_ms": 12.0, "latency_p99_ms": 15.0, "latency_mean_ms": 11.0,
             "latency_std_ms": 1.5, "avg_gpu_utilization_pct": 80.0,
             "avg_gpu_memory_used_mb": 4000.0, "throughput_unit": "images/sec",
             "model_name": "ResNet-50", "param_count": 25_600_000,
             "repeat": 1, "seed": 42},
            {"gpu_type": "T4", "workload": "resnet50", "mode": "inference",
             "batch_size": 32, "throughput": 510.0, "latency_p50_ms": 9.8,
             "latency_p95_ms": 11.8, "latency_p99_ms": 14.5, "latency_mean_ms": 10.8,
             "latency_std_ms": 1.4, "avg_gpu_utilization_pct": 82.0,
             "avg_gpu_memory_used_mb": 4100.0, "throughput_unit": "images/sec",
             "model_name": "ResNet-50", "param_count": 25_600_000,
             "repeat": 2, "seed": 43},
        ])
        df.to_csv(results_dir / "benchmark_summary_T4.csv", index=False)

        config_path = tmp_path / "rec_config.yaml"
        config_path.write_text("scoring:\n  weights:\n    throughput: 0.4\n    cost_efficiency: 0.35\n    latency: 0.25\n")

        cost_path = tmp_path / "cost_rates.yaml"
        cost_path.write_text("gpu_rates:\n  T4:\n    cost_per_hour: 0.526\n    gpu_memory_gb: 16\n")

        db_path = tmp_path / "history.db"
        engine = RecommendationEngine(
            config_path=str(config_path),
            cost_rates_path=str(cost_path),
            history_db_path=str(db_path),
        )

        result = engine.recommend(results_dir=str(results_dir))
        assert result["status"] == "ok"
        assert result["recommended_gpu"] == "T4"
        assert len(result["rankings"]) == 1
        assert result["composite_score"] > 0

    def test_recommend_with_constraints(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        df = pd.DataFrame([
            {"gpu_type": "T4", "workload": "resnet50", "mode": "inference",
             "batch_size": 32, "throughput": 500.0, "latency_p50_ms": 10.0,
             "latency_p95_ms": 12.0, "latency_p99_ms": 15.0, "latency_mean_ms": 11.0,
             "latency_std_ms": 1.5, "avg_gpu_utilization_pct": 80.0,
             "avg_gpu_memory_used_mb": 4000.0, "repeat": 1, "seed": 42},
        ])
        df.to_csv(results_dir / "benchmark_summary_T4.csv", index=False)

        config_path = tmp_path / "rec_config.yaml"
        config_path.write_text("{}")
        cost_path = tmp_path / "cost_rates.yaml"
        cost_path.write_text("gpu_rates:\n  T4:\n    cost_per_hour: 5.0\n    gpu_memory_gb: 16\n")

        engine = RecommendationEngine(
            config_path=str(config_path),
            cost_rates_path=str(cost_path),
            history_db_path=str(tmp_path / "h.db"),
        )

        constraints = UserConstraints(max_cost_per_hour=1.0)
        result = engine.recommend(results_dir=str(results_dir), constraints=constraints)
        assert result["status"] == "no_feasible_gpus"
        assert len(result["excluded"]) == 1

    def test_predict_empty_history(self, tmp_path):
        config_path = tmp_path / "rec_config.yaml"
        config_path.write_text("{}")
        cost_path = tmp_path / "cost_rates.yaml"
        cost_path.write_text("gpu_rates: {}")

        engine = RecommendationEngine(
            config_path=str(config_path),
            cost_rates_path=str(cost_path),
            history_db_path=str(tmp_path / "empty.db"),
        )
        result = engine.predict_and_recommend(
            param_count=25_600_000, batch_size=32, mode="inference",
        )
        assert result["status"] == "error"

    def test_import_results(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        df = pd.DataFrame([
            {"gpu_type": "T4", "workload": "resnet50", "mode": "inference",
             "batch_size": 32, "throughput": 500.0, "latency_p50_ms": 10.0,
             "latency_p95_ms": 12.0, "latency_p99_ms": 15.0, "latency_mean_ms": 11.0,
             "latency_std_ms": 1.5, "avg_gpu_utilization_pct": 80.0,
             "avg_gpu_memory_used_mb": 4000.0, "repeat": 1, "seed": 42},
        ])
        df.to_csv(results_dir / "benchmark_summary_T4.csv", index=False)

        config_path = tmp_path / "rec_config.yaml"
        config_path.write_text("{}")
        cost_path = tmp_path / "cost_rates.yaml"
        cost_path.write_text("gpu_rates:\n  T4:\n    cost_per_hour: 0.526\n    gpu_memory_gb: 16\n")

        engine = RecommendationEngine(
            config_path=str(config_path),
            cost_rates_path=str(cost_path),
            history_db_path=str(tmp_path / "import.db"),
        )
        count = engine.import_results_to_history(str(results_dir))
        assert count == 1
        assert engine.history.get_run_count() == 1


# ======================================================================
# Format / Output Tests
# ======================================================================

class TestFormatRecommendation:

    def test_format_ok_result(self):
        result = {
            "status": "ok",
            "source": "benchmark",
            "recommended_gpu": "T4",
            "composite_score": 0.87,
            "constraints_applied": "none",
            "rankings": [{
                "gpu_type": "T4", "rank": 1, "composite_score": 0.87,
                "throughput": 500.0, "throughput_unit": "images/sec",
                "latency_p95_ms": 12.0, "cost_per_hour": 0.526,
                "throughput_per_dollar": 3420000.0,
                "confidence_note": "full benchmark run",
                "detail_lines": ["Highest composite score"],
            }],
            "excluded": [],
            "history_stats": {"total_runs": 5, "gpus_benchmarked": 1, "workloads_benchmarked": 1},
        }
        text = format_recommendation(result)
        assert "RECOMMENDED: T4" in text
        assert "0.8700" in text
        assert "500.00" in text

    def test_format_error_result(self):
        result = {"status": "error", "message": "No data found"}
        text = format_recommendation(result)
        assert "ERROR" in text
        assert "No data found" in text

    def test_format_with_excluded(self):
        result = {
            "status": "ok", "source": "benchmark",
            "recommended_gpu": "T4", "composite_score": 0.87,
            "constraints_applied": "cost <= $1.00/hr",
            "rankings": [{
                "gpu_type": "T4", "rank": 1, "composite_score": 0.87,
                "throughput": 500.0, "throughput_unit": "images/sec",
                "latency_p95_ms": 12.0, "cost_per_hour": 0.526,
                "throughput_per_dollar": 3420000.0,
                "confidence_note": "full benchmark run",
                "detail_lines": [],
            }],
            "excluded": [{"gpu_type": "A100", "reason": "cost too high"}],
            "history_stats": {},
        }
        text = format_recommendation(result)
        assert "Excluded by constraints" in text
        assert "A100" in text
