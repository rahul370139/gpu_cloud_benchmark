"""GPU Recommendation Engine — intelligent infrastructure advisor for ML workloads.

Three operating modes:
    recommend  — score & rank GPUs from existing benchmark results
    partial    — run short convergence-checked benchmarks, then recommend
    predict    — estimate performance from historical data without running anything
"""

from .scorer import score_gpus, GpuScore
from .constraints import UserConstraints, apply_constraints
from .history import HistoryStore
from .partial import PartialProfiler, PartialResult
from .predictor import WorkloadPredictor, PredictionResult
from .engine import RecommendationEngine
