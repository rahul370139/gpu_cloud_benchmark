"""Helpers for parsing benchmark workload matrix configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .workloads import register_workload


@dataclass(frozen=True)
class WorkloadSpec:
    """Fully-resolved benchmark plan for one workload."""

    name: str
    batch_sizes: list[int]
    modes: list[str]


def _normalize_int_list(values, field_name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"'{field_name}' must be a non-empty list")
    normalized: list[int] = []
    for value in values:
        if not isinstance(value, int):
            raise TypeError(f"'{field_name}' entries must be integers, got {value!r}")
        normalized.append(value)
    return normalized


def _normalize_mode_list(values) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("'modes' must be a non-empty list")
    normalized: list[str] = []
    for mode in values:
        if mode not in ("inference", "training"):
            raise ValueError(f"mode must be 'inference' or 'training', got '{mode}'")
        normalized.append(mode)
    return normalized


def resolve_workload_specs(
    raw_workloads,
    *,
    default_batch_sizes,
    default_modes,
) -> list[WorkloadSpec]:
    """Resolve config workloads into a workload execution plan.

    Supports both legacy string entries and richer per-workload overrides:

    - ``"resnet50"``
    - ``{"name": "clip", "target": "user_workloads.clip:ClipWorkload"}``
    - ``{"name": "clip", "modes": ["inference"], "batch_sizes": [1, 8, 32]}``
    """

    default_batch_sizes = _normalize_int_list(default_batch_sizes, "batch_sizes")
    default_modes = _normalize_mode_list(default_modes)

    specs: list[WorkloadSpec] = []
    for entry in raw_workloads:
        if isinstance(entry, str):
            specs.append(
                WorkloadSpec(
                    name=entry,
                    batch_sizes=list(default_batch_sizes),
                    modes=list(default_modes),
                )
            )
            continue

        if isinstance(entry, dict):
            name = entry.get("name")
            target = entry.get("target")
            if not name:
                raise ValueError("Workload entries must include 'name'")
            if target:
                register_workload(name, target)

            batch_sizes = entry.get("batch_sizes", default_batch_sizes)
            modes = entry.get("modes", default_modes)
            specs.append(
                WorkloadSpec(
                    name=name,
                    batch_sizes=_normalize_int_list(batch_sizes, f"{name}.batch_sizes"),
                    modes=_normalize_mode_list(modes),
                )
            )
            continue

        raise TypeError(f"Unsupported workload config entry: {entry!r}")

    return specs
