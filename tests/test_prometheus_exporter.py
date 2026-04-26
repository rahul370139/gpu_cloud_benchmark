"""Tests for Prometheus Pushgateway export behavior."""

import sys
import types

from src.metrics import prometheus_exporter as exporter


class _FakeRegistry:
    def __init__(self):
        self.metric_names = set()
        self.values = {}


class _FakeGauge:
    def __init__(self, name, _desc, _labels, registry):
        if name in registry.metric_names:
            raise ValueError(f"duplicate metric: {name}")
        registry.metric_names.add(name)
        self.name = name
        self.registry = registry

    def labels(self, *vals):
        self.label_vals = vals
        return self

    def set(self, value):
        self.registry.values[(self.name, self.label_vals)] = value


def test_push_benchmark_metrics_uses_fresh_registry(monkeypatch):
    pushes = []

    fake_prom = types.SimpleNamespace(
        CollectorRegistry=_FakeRegistry,
        Gauge=_FakeGauge,
        push_to_gateway=lambda gateway, job, registry: pushes.append((gateway, job, registry)),
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", fake_prom)

    assert exporter.init_prometheus("http://pushgateway:9091")

    kwargs = dict(
        job_name="benchmark_T4",
        gpu_type="T4",
        workload="resnet50",
        batch_size=8,
        throughput=123.4,
        latency_p50=1.2,
        latency_p95=2.3,
        latency_p99=3.4,
        gpu_utilization=91.0,
        gpu_memory_used_mb=1024.0,
    )

    exporter.push_benchmark_metrics(**kwargs)
    exporter.push_benchmark_metrics(**kwargs)

    assert len(pushes) == 2
    assert pushes[0][2] is not pushes[1][2]
