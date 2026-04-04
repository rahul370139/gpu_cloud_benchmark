#!/usr/bin/env bash
set -euo pipefail

CONFIG="${BENCHMARK_CONFIG:-/app/config/benchmark_config.yaml}"
DEVICE="${BENCHMARK_DEVICE:-}"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-/app/results}"

echo "=========================================="
echo " GPU Cloud Benchmark"
echo "=========================================="
echo " Config : ${CONFIG}"
echo " Device : ${DEVICE:-auto}"
echo " Results: ${RESULTS_DIR}"
echo "=========================================="

echo "[1/4] Running preflight checks..."
python /app/scripts/preflight_check.py
echo "       Preflight OK."

echo "[2/4] Capturing environment..."
python -c "
from src.reproducibility.env_capture import capture_environment
import json, pathlib
env = capture_environment()
out = pathlib.Path('${RESULTS_DIR}') / 'environment.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(env, indent=2))
print(f'       Environment snapshot -> {out}')
"

echo "[3/4] Running benchmarks..."
DEVICE_ARG=""
if [ -n "${DEVICE}" ]; then
    DEVICE_ARG="--device ${DEVICE}"
fi
python -m src.runner --config "${CONFIG}" ${DEVICE_ARG}

echo "[4/4] Generating report..."
python /app/scripts/generate_report.py --results-dir "${RESULTS_DIR}" --output "${RESULTS_DIR}/report.html"

echo "=========================================="
echo " Benchmark complete."
echo " Results: ${RESULTS_DIR}/"
echo "=========================================="
