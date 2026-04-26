"""Unit tests for benchmark workloads — all run on CPU.

These tests are designed to run inside the Docker container where all
dependencies (torch, torchvision, transformers, pynvml) are available.
On a host machine without the full stack, run only the container-independent
test files: test_cost.py, test_metrics.py, test_reproducibility.py.
"""

import pytest
import torch
import yaml

from src.workloads import get_workload, WORKLOAD_REGISTRY, register_workload, CUSTOM_WORKLOADS
from src.workloads.base import WorkloadMetadata
from src.runner import run_full_benchmark


class TestWorkloadRegistry:
    def test_known_workloads(self):
        assert "resnet50" in WORKLOAD_REGISTRY
        assert "bert_base" in WORKLOAD_REGISTRY

    def test_unknown_workload_raises(self):
        with pytest.raises(ValueError, match="Unknown workload"):
            get_workload("nonexistent")

    def test_can_register_custom_workload(self):
        register_workload("toy_mlp", "user_workloads.example_mlp:ExampleMLPWorkload")
        assert "toy_mlp" in CUSTOM_WORKLOADS
        workload = get_workload("toy_mlp", batch_size=2, device="cpu", mode="inference")
        workload.setup()
        batch = workload.generate_batch()
        output = workload.run_iteration(batch)
        assert output.shape == (2, 4)
        workload.cleanup()

    def test_runner_accepts_workload_target_flag(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "workloads": ["resnet50"],
            "batch_sizes": [1],
            "num_repeats": 1,
            "warmup_iters": 1,
            "benchmark_iters": 1,
            "modes": ["inference"],
            "output_dir": str(tmp_path / "results"),
            "prometheus_pushgateway": "",
        }))
        output_dir = run_full_benchmark(
            str(config_path),
            device="cpu",
            workload_target="user_workloads.example_mlp:ExampleMLPWorkload",
            workload_name="cli_custom",
        )
        assert (output_dir / "benchmark_summary_CPU.csv").exists()


class TestResNet50:
    @pytest.fixture
    def workload_inference(self):
        w = get_workload("resnet50", batch_size=2, device="cpu", mode="inference")
        w.setup()
        yield w
        w.cleanup()

    @pytest.fixture
    def workload_training(self):
        w = get_workload("resnet50", batch_size=2, device="cpu", mode="training")
        w.setup()
        yield w
        w.cleanup()

    def test_generate_batch_shape(self, workload_inference):
        batch = workload_inference.generate_batch()
        assert batch["images"].shape == (2, 3, 224, 224)

    def test_inference_output_shape(self, workload_inference):
        batch = workload_inference.generate_batch()
        output = workload_inference.run_iteration(batch)
        assert output.shape == (2, 1000)

    def test_training_returns_scalar_loss(self, workload_training):
        batch = workload_training.generate_batch()
        loss = workload_training.run_iteration(batch)
        assert loss.dim() == 0

    def test_metadata(self, workload_inference):
        meta = workload_inference.get_metadata()
        assert isinstance(meta, WorkloadMetadata)
        assert meta.name == "resnet50"
        assert meta.param_count > 0
        assert meta.throughput_unit == "images/sec"

    def test_warmup_runs(self, workload_inference):
        workload_inference.warmup(n_iters=2)

    def test_samples_per_batch(self, workload_inference):
        assert workload_inference.samples_per_batch() == 2


class TestBertBase:
    @pytest.fixture
    def workload_inference(self):
        w = get_workload("bert_base", batch_size=2, device="cpu", mode="inference")
        w.setup()
        yield w
        w.cleanup()

    @pytest.fixture
    def workload_training(self):
        w = get_workload("bert_base", batch_size=2, device="cpu", mode="training")
        w.setup()
        yield w
        w.cleanup()

    def test_generate_batch_shape(self, workload_inference):
        batch = workload_inference.generate_batch()
        assert batch["input_ids"].shape == (2, 512)
        assert batch["attention_mask"].shape == (2, 512)

    def test_inference_output_shape(self, workload_inference):
        batch = workload_inference.generate_batch()
        output = workload_inference.run_iteration(batch)
        assert output.shape[0] == 2
        assert output.shape[1] == 512

    def test_training_returns_scalar_loss(self, workload_training):
        batch = workload_training.generate_batch()
        loss = workload_training.run_iteration(batch)
        assert loss.dim() == 0

    def test_metadata(self, workload_inference):
        meta = workload_inference.get_metadata()
        assert meta.name == "bert_base"
        assert meta.param_count > 0
        assert meta.throughput_unit == "tokens/sec"

    def test_samples_per_batch_is_tokens(self, workload_inference):
        assert workload_inference.samples_per_batch() == 2 * 512
