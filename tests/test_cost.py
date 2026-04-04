"""Unit tests for cost calculator."""

import pytest
import pandas as pd

from src.cost.calculator import load_gpu_rates, compute_cost_metrics


@pytest.fixture
def sample_rates():
    return {
        "T4": {"instance_type": "g4dn.xlarge", "cost_per_hour": 0.526, "gpu_memory_gb": 16},
        "A100": {"instance_type": "p4d.24xlarge", "cost_per_hour": 4.10, "gpu_memory_gb": 40},
    }


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"gpu_type": "T4", "workload": "resnet50", "batch_size": 32, "mean_throughput": 500.0},
        {"gpu_type": "A100", "workload": "resnet50", "batch_size": 32, "mean_throughput": 3000.0},
        {"gpu_type": "T4", "workload": "bert_base", "batch_size": 8, "mean_throughput": 200.0},
        {"gpu_type": "A100", "workload": "bert_base", "batch_size": 8, "mean_throughput": 1500.0},
    ])


class TestComputeCostMetrics:
    def test_adds_cost_columns(self, sample_df, sample_rates):
        result = compute_cost_metrics(sample_df, sample_rates)
        assert "throughput_per_dollar" in result.columns
        assert "cost_per_1k_samples" in result.columns
        assert "cost_efficiency_rank" in result.columns

    def test_throughput_per_dollar_formula(self, sample_df, sample_rates):
        result = compute_cost_metrics(sample_df, sample_rates)
        t4_row = result[(result["gpu_type"] == "T4") & (result["workload"] == "resnet50")].iloc[0]
        expected = 500.0 * 3600 / 0.526
        assert abs(t4_row["throughput_per_dollar"] - expected) < 1.0

    def test_cost_per_1k_formula(self, sample_df, sample_rates):
        result = compute_cost_metrics(sample_df, sample_rates)
        a100_row = result[(result["gpu_type"] == "A100") & (result["workload"] == "resnet50")].iloc[0]
        expected = (1000.0 / 3000.0) * (4.10 / 3600.0)
        assert abs(a100_row["cost_per_1k_samples"] - expected) < 1e-6

    def test_rank_ordering(self, sample_df, sample_rates):
        result = compute_cost_metrics(sample_df, sample_rates)
        resnet = result[result["workload"] == "resnet50"].sort_values("cost_efficiency_rank")
        assert resnet.iloc[0]["cost_efficiency_rank"] == 1

    def test_zero_cost_handled(self, sample_rates):
        sample_rates["FREE"] = {"instance_type": "test", "cost_per_hour": 0.0, "gpu_memory_gb": 0}
        df = pd.DataFrame([
            {"gpu_type": "FREE", "workload": "resnet50", "batch_size": 1, "mean_throughput": 100.0},
        ])
        result = compute_cost_metrics(df, sample_rates)
        assert result["throughput_per_dollar"].iloc[0] == 0.0
