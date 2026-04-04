"""Capture runtime environment details for reproducibility."""

import os
import platform
import subprocess
import logging
from datetime import datetime, timezone

import torch

from .checksum import compute_pip_freeze_hash, compute_docker_image_id

logger = logging.getLogger(__name__)


def _safe_cmd(cmd: list[str], timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def _nvidia_driver_version() -> str:
    return _safe_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "NO_GPU"


def _cuda_version() -> str:
    if torch.cuda.is_available():
        return torch.version.cuda or "UNAVAILABLE"
    return "NO_CUDA"


def _cudnn_version() -> str:
    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        return str(torch.backends.cudnn.version())
    return "UNAVAILABLE"


def capture_environment() -> dict:
    """Return a dict describing the full runtime environment."""
    pip_freeze, pip_hash = compute_pip_freeze_hash()

    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": _cuda_version(),
        "cudnn_version": _cudnn_version(),
        "nvidia_driver_version": _nvidia_driver_version(),
        "gpu_name": _gpu_name(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "docker_container_id": compute_docker_image_id(),
        "pip_freeze_sha256": pip_hash,
        "pip_freeze": pip_freeze,
    }

    # Detect NVIDIA Container Runtime
    env["nvidia_container_runtime"] = os.environ.get("NVIDIA_VISIBLE_DEVICES", "NOT_SET")

    logger.info(
        "Environment: %s, PyTorch %s, CUDA %s, GPU %s",
        env["os"], env["torch_version"], env["cuda_version"], env["gpu_name"],
    )
    return env
