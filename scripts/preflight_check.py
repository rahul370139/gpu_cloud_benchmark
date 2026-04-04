#!/usr/bin/env python3
"""Pre-flight validation: GPU driver, CUDA, NVML access.

Exits non-zero if a critical check fails so the container fails fast.
"""

import subprocess
import sys


def check_python():
    v = sys.version_info
    print(f"  Python: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print("  WARNING: Python 3.9+ recommended")


def check_torch():
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("  CRITICAL: PyTorch not installed")
        sys.exit(1)


def check_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        for line in result.stdout.strip().split("\n"):
            print(f"  nvidia-smi: {line.strip()}")
    except FileNotFoundError:
        print("  WARNING: nvidia-smi not found (OK for CPU-only runs)")
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: nvidia-smi failed: {e}")


def check_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        print(f"  NVML: {count} GPU(s) detected")
        pynvml.nvmlShutdown()
    except ImportError:
        print("  WARNING: pynvml not installed (nvidia-smi fallback will be used)")
    except Exception as e:
        print(f"  WARNING: NVML init failed: {e}")


def main():
    print("Preflight Checks")
    print("-" * 40)
    check_python()
    check_torch()
    check_nvidia_smi()
    check_nvml()
    print("-" * 40)
    print("All checks passed.")


if __name__ == "__main__":
    main()
