"""Workload registry with lazy imports.

Built-in workloads are resolved at first use so importing the package does not
pull in heavy transitive dependencies until the specific workload is requested.
Custom workloads can also be registered dynamically via import paths of the
form ``module.submodule:ClassName``.
"""

import importlib

from .base import BaseWorkload

WORKLOAD_REGISTRY: dict[str, tuple[str, str]] = {
    "resnet50": (".vision", "ResNet50Workload"),
    "bert_base": (".nlp", "BertBaseWorkload"),
}

CUSTOM_WORKLOADS: dict[str, str] = {}
_resolved_cache: dict[str, type[BaseWorkload]] = {}


def register_workload(name: str, target: str) -> None:
    """Register a custom workload alias.

    Args:
        name: Friendly workload name used in config.
        target: Import path in ``module.path:ClassName`` format.
    """
    if ":" not in target:
        raise ValueError(
            f"Custom workload target '{target}' must be in 'module.path:ClassName' format"
        )
    CUSTOM_WORKLOADS[name] = target
    _resolved_cache.pop(name, None)


def register_custom_workloads(workloads: dict[str, str] | None) -> None:
    for name, target in (workloads or {}).items():
        register_workload(name, target)


def available_workloads() -> list[str]:
    return sorted(set(WORKLOAD_REGISTRY) | set(CUSTOM_WORKLOADS))


def _resolve_target(target: str) -> type[BaseWorkload]:
    module_path, class_name = target.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseWorkload):
        raise TypeError(f"{target} is not a BaseWorkload subclass")
    return cls


def _resolve(name: str) -> type[BaseWorkload]:
    if name in _resolved_cache:
        return _resolved_cache[name]
    if name in CUSTOM_WORKLOADS:
        cls = _resolve_target(CUSTOM_WORKLOADS[name])
        _resolved_cache[name] = cls
        return cls
    module_path, class_name = WORKLOAD_REGISTRY[name]
    module = importlib.import_module(module_path, package=__package__)
    cls = getattr(module, class_name)
    _resolved_cache[name] = cls
    return cls


def get_workload(name: str, **kwargs) -> BaseWorkload:
    if name not in WORKLOAD_REGISTRY and name not in CUSTOM_WORKLOADS:
        raise ValueError(
            f"Unknown workload '{name}'. Available: {available_workloads()}"
        )
    cls = _resolve(name)
    return cls(**kwargs)
