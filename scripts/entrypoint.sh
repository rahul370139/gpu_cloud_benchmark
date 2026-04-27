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

echo "[1/6] Running preflight checks..."
python /app/scripts/preflight_check.py
echo "       Preflight OK."

echo "[2/6] Capturing environment..."
python -c "
from src.reproducibility.env_capture import capture_environment
import json, pathlib
env = capture_environment()
out = pathlib.Path('${RESULTS_DIR}') / 'environment.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(env, indent=2))
print(f'       Environment snapshot -> {out}')
"

echo "[3/6] Running benchmarks..."
DEVICE_ARG=""
if [ -n "${DEVICE}" ]; then
    DEVICE_ARG="--device ${DEVICE}"
fi
BENCHMARK_RESULTS_DIR="${RESULTS_DIR}" python -m src.runner --config "${CONFIG}" ${DEVICE_ARG}

echo "[4/6] Generating report..."
python /app/scripts/generate_report.py --results-dir "${RESULTS_DIR}" --output "${RESULTS_DIR}/report.html"

echo "[5/6] Running GPU recommendation engine..."
python -m src.recommender recommend \
    --results-dir "${RESULTS_DIR}" \
    -o "${RESULTS_DIR}/recommendation.json" \
    2>&1 || echo "       (Recommendation skipped — single-GPU run)"

echo "[6/6] Uploading artifacts to S3..."
python -m src.artifacts.s3_uploader --results-dir "${RESULTS_DIR}"

echo "=========================================="
echo " Benchmark complete."
echo " Results: ${RESULTS_DIR}/"
echo "=========================================="
