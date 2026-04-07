"""User-defined constraint filtering for GPU recommendations.

Users specify hard limits (budget, latency, throughput) and the filter
removes any GPU that violates them *before* scoring, so the recommendation
only considers feasible options.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UserConstraints:
    """Hard constraints a GPU must satisfy to be recommended."""
    max_cost_per_hour: float | None = None
    max_latency_p95_ms: float | None = None
    min_throughput: float | None = None
    max_gpu_memory_gb: float | None = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (
                self.max_cost_per_hour,
                self.max_latency_p95_ms,
                self.min_throughput,
                self.max_gpu_memory_gb,
            )
        )

    def describe(self) -> str:
        parts = []
        if self.max_cost_per_hour is not None:
            parts.append(f"cost <= ${self.max_cost_per_hour:.2f}/hr")
        if self.max_latency_p95_ms is not None:
            parts.append(f"P95 latency <= {self.max_latency_p95_ms:.1f} ms")
        if self.min_throughput is not None:
            parts.append(f"throughput >= {self.min_throughput:.1f}")
        if self.max_gpu_memory_gb is not None:
            parts.append(f"GPU memory <= {self.max_gpu_memory_gb:.0f} GB")
        return ", ".join(parts) if parts else "none"

    def to_dict(self) -> dict:
        return {
            "max_cost_per_hour": self.max_cost_per_hour,
            "max_latency_p95_ms": self.max_latency_p95_ms,
            "min_throughput": self.min_throughput,
            "max_gpu_memory_gb": self.max_gpu_memory_gb,
        }


@dataclass
class ExcludedGpu:
    gpu_type: str
    reason: str


def apply_constraints(
    scores: list,  # list[GpuScore]
    constraints: UserConstraints,
) -> tuple[list, list[ExcludedGpu]]:
    """Partition scored GPUs into (feasible, excluded).

    Returns:
        (feasible_scores, excluded) where excluded carries the rejection reason.
    """
    if constraints.is_empty():
        return scores, []

    feasible = []
    excluded: list[ExcludedGpu] = []

    for s in scores:
        reasons = []

        if constraints.max_cost_per_hour is not None and s.cost_per_hour > constraints.max_cost_per_hour:
            reasons.append(
                f"cost ${s.cost_per_hour:.2f}/hr exceeds ${constraints.max_cost_per_hour:.2f}/hr budget"
            )

        if constraints.max_latency_p95_ms is not None and s.latency_p95_ms > constraints.max_latency_p95_ms:
            reasons.append(
                f"P95 latency {s.latency_p95_ms:.1f} ms exceeds {constraints.max_latency_p95_ms:.1f} ms limit"
            )

        if constraints.min_throughput is not None and s.throughput < constraints.min_throughput:
            reasons.append(
                f"throughput {s.throughput:.1f} below required {constraints.min_throughput:.1f}"
            )

        if reasons:
            excluded.append(ExcludedGpu(gpu_type=s.gpu_type, reason="; ".join(reasons)))
            logger.info("Excluded %s: %s", s.gpu_type, "; ".join(reasons))
        else:
            feasible.append(s)

    if not feasible:
        logger.warning(
            "All %d GPUs excluded by constraints (%s). "
            "Consider relaxing constraints.",
            len(scores), constraints.describe(),
        )

    for i, s in enumerate(feasible):
        s.rank = i + 1

    return feasible, excluded
