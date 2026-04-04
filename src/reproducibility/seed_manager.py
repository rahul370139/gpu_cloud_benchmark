"""Deterministic seeding for reproducible benchmark runs."""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_deterministic(seed: int = 42) -> dict:
    """Pin all random number generators and enable deterministic CUDA ops.

    Returns a dict summarizing what was set, for inclusion in the run manifest.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.8+ deterministic algorithms flag
    torch.use_deterministic_algorithms(True, warn_only=True)

    summary = {
        "seed": seed,
        "python_random": True,
        "numpy_random": True,
        "torch_manual_seed": True,
        "torch_cuda_seed": cuda_seeded,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": True,
    }
    logger.info("Deterministic mode set with seed=%d (CUDA=%s)", seed, cuda_seeded)
    return summary
