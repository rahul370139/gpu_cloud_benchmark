"""Push benchmark metrics to a Prometheus Pushgateway."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_GATEWAY: Optional[str] = None
_REGISTRY = None


def init_prometheus(pushgateway_url: str) -> bool:
    """Initialize the Prometheus client and set the pushgateway URL.

    Returns True if the client is available, False otherwise.
    """
    global _GATEWAY, _REGISTRY
    if not pushgateway_url:
        logger.info("Prometheus pushgateway URL empty — metrics push disabled")
        return False
    try:
        from prometheus_client import CollectorRegistry
        _GATEWAY = pushgateway_url.rstrip("/")
        _REGISTRY = CollectorRegistry()
        logger.info("Prometheus exporter initialized, gateway=%s", _GATEWAY)
        return True
    except ImportError:
        logger.warning("prometheus_client not installed — metrics push disabled")
        return False


def push_benchmark_metrics(
    job_name: str,
    gpu_type: str,
    workload: str,
    batch_size: int,
    throughput: float,
    latency_p50: float,
    latency_p95: float,
    latency_p99: float,
    gpu_utilization: float,
    gpu_memory_used_mb: float,
) -> None:
    """Push a single benchmark result to the Prometheus Pushgateway."""
    if _GATEWAY is None or _REGISTRY is None:
        return
    try:
        from prometheus_client import Gauge, push_to_gateway

        labels = ["gpu_type", "workload", "batch_size"]
        label_vals = [gpu_type, workload, str(batch_size)]

        gauges = {
            "benchmark_throughput": ("Throughput in samples/sec", throughput),
            "benchmark_latency_p50_ms": ("Latency p50 in ms", latency_p50),
            "benchmark_latency_p95_ms": ("Latency p95 in ms", latency_p95),
            "benchmark_latency_p99_ms": ("Latency p99 in ms", latency_p99),
            "benchmark_gpu_utilization_pct": ("GPU utilization %", gpu_utilization),
            "benchmark_gpu_memory_used_mb": ("GPU memory used in MB", gpu_memory_used_mb),
        }

        for metric_name, (desc, value) in gauges.items():
            g = Gauge(metric_name, desc, labels, registry=_REGISTRY)
            g.labels(*label_vals).set(value)

        push_to_gateway(_GATEWAY, job=job_name, registry=_REGISTRY)
        logger.info("Pushed metrics for %s/%s/bs%d to %s", gpu_type, workload, batch_size, _GATEWAY)
    except Exception as e:
        logger.warning("Failed to push to Prometheus: %s", e)
