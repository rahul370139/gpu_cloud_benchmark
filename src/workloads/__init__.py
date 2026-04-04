"""Workload registry with lazy imports.

Workload classes are resolved at first use so that importing the package
does not pull in heavy transitive dependencies (transformers, etc.) until
the specific workload is actually requested.  This is the same pattern
used by torchvision.models and HuggingFace model registries.
"""

import importlib
from .base import BaseWorkload

WORKLOAD_REGISTRY: dict[str, tuple[str, str]] = {
    "resnet50":  (".vision", "ResNet50Workload"),
    "bert_base": (".nlp",    "BertBaseWorkload"),
}

_resolved_cache: dict[str, type[BaseWorkload]] = {}


def _resolve(name: str) -> type[BaseWorkload]:
    if name in _resolved_cache:
        return _resolved_cache[name]
    module_path, class_name = WORKLOAD_REGISTRY[name]
    module = importlib.import_module(module_path, package=__package__)
    cls = getattr(module, class_name)
    _resolved_cache[name] = cls
    return cls


def get_workload(name: str, **kwargs) -> BaseWorkload:
    if name not in WORKLOAD_REGISTRY:
        raise ValueError(
            f"Unknown workload '{name}'. Available: {list(WORKLOAD_REGISTRY.keys())}"
        )
    cls = _resolve(name)
    return cls(**kwargs)
