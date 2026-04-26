#!/usr/bin/env bash
#
# Dispatch one benchmark Kubernetes Job per GPU class declared in the
# Terraform inventory and wait for them to complete in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

require_cmd kubectl
require_cmd jq
require_cmd envsubst
ensure_k3s_tunnel
export KUBECONFIG="${KUBECONFIG_LOCAL}"

load_inventory
BENCHMARK_RUN_ID="$(jq -r '.benchmark_run_id' "${INVENTORY_FILE}")"
BENCHMARK_IMAGE="${BENCHMARK_IMAGE:-ghcr.io/example/ml-benchmark:latest}"
GPU_CLASSES=$(jq -r '.gpu_classes[]' "${INVENTORY_FILE}")
ARTIFACT_BUCKET_NAME="$(jq -r '.artifact_bucket_name' "${INVENTORY_FILE}")"
AWS_REGION="$(jq -r '.region' "${INVENTORY_FILE}")"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts/${BENCHMARK_RUN_ID}"

if [[ -z "${GPU_CLASSES}" ]]; then
  echo "No GPU classes found in inventory; is terraform applied?" >&2
  exit 1
fi

echo "Applying shared resources (namespace, PVC, config)..."
kubectl apply -f "${ROOT_DIR}/kubernetes/base/namespace.yaml"
kubectl apply -f "${ROOT_DIR}/kubernetes/base/benchmark-shared.yaml"

if [[ "${BENCHMARK_IMAGE}" == *.amazonaws.com/* ]]; then
  require_cmd aws
  REGISTRY_HOST="${BENCHMARK_IMAGE%%/*}"
  REGISTRY_REGION="$(echo "${REGISTRY_HOST}" | awk -F'.' '{print $4}')"
  echo "Refreshing ECR pull secret for ${REGISTRY_HOST}..."
  kubectl create secret docker-registry benchmark-registry-credentials \
    -n ml-benchmark \
    --docker-server="${REGISTRY_HOST}" \
    --docker-username=AWS \
    --docker-password="$(aws ecr get-login-password --region "${REGISTRY_REGION}")" \
    --dry-run=client -o yaml | kubectl apply -f -
fi

JOB_NAMES=()
RENDERED_DIR="$(mktemp -d)"
trap 'rm -rf "${RENDERED_DIR}"' EXIT
mkdir -p "${ARTIFACTS_DIR}/manifests" "${ARTIFACTS_DIR}/logs"

# Render and apply one Job per GPU class.
for GPU_CLASS in ${GPU_CLASSES}; do
  GPU_CLASS_LOWER="$(echo "${GPU_CLASS}" | tr '[:upper:]' '[:lower:]')"
  JOB_NAME="benchmark-run-${GPU_CLASS_LOWER}"
  RENDERED="${RENDERED_DIR}/job-${GPU_CLASS_LOWER}.yaml"

  GPU_CLASS="${GPU_CLASS}" \
  GPU_CLASS_LOWER="${GPU_CLASS_LOWER}" \
  BENCHMARK_RUN_ID="${BENCHMARK_RUN_ID}" \
  BENCHMARK_IMAGE="${BENCHMARK_IMAGE}" \
  ARTIFACT_BUCKET_NAME="${ARTIFACT_BUCKET_NAME}" \
  AWS_REGION="${AWS_REGION}" \
  envsubst < "${ROOT_DIR}/kubernetes/base/benchmark-job.yaml" > "${RENDERED}"
  cp "${RENDERED}" "${ARTIFACTS_DIR}/manifests/${JOB_NAME}.yaml"

  kubectl delete job "${JOB_NAME}" -n ml-benchmark --ignore-not-found
  kubectl apply -f "${RENDERED}"
  JOB_NAMES+=("${JOB_NAME}")
  echo "  dispatched ${JOB_NAME} (gpu_class=${GPU_CLASS})"
done

echo "Waiting for ${#JOB_NAMES[@]} job(s) to complete in parallel..."
EXIT_CODE=0
for JOB in "${JOB_NAMES[@]}"; do
  if ! kubectl wait --for=condition=complete "job/${JOB}" \
       -n ml-benchmark --timeout=2h; then
    echo "Job ${JOB} did not complete successfully" >&2
    EXIT_CODE=1
  fi
done

for JOB in "${JOB_NAMES[@]}"; do
  echo "===== logs: ${JOB} ====="
  kubectl logs "job/${JOB}" -n ml-benchmark --tail=200 | tee "${ARTIFACTS_DIR}/logs/${JOB}.log" || true
done

if [[ "${EXIT_CODE}" -eq 0 && -n "${ARTIFACT_BUCKET_NAME}" && "${ARTIFACT_BUCKET_NAME}" != "null" ]]; then
  require_cmd aws
  require_cmd python3

  RESULTS_SYNC_DIR="${ARTIFACTS_DIR}/results"
  COMPARISON_DIR="${ARTIFACTS_DIR}/comparison"
  mkdir -p "${RESULTS_SYNC_DIR}" "${COMPARISON_DIR}"

  echo "Syncing run artifacts from s3://${ARTIFACT_BUCKET_NAME}/benchmark-runs/${BENCHMARK_RUN_ID}/ ..."
  aws s3 sync \
    "s3://${ARTIFACT_BUCKET_NAME}/benchmark-runs/${BENCHMARK_RUN_ID}/" \
    "${RESULTS_SYNC_DIR}"

  echo "Generating consolidated comparison report..."
  python3 "${PROJECT_DIR}/scripts/generate_report.py" \
    --results-dir "${RESULTS_SYNC_DIR}" \
    --output "${COMPARISON_DIR}/report.html"

  echo "Generating consolidated recommendation..."
  (
    cd "${PROJECT_DIR}"
    python3 -m src.recommender recommend \
      --results-dir "${RESULTS_SYNC_DIR}" \
      -o "${COMPARISON_DIR}/recommendation.json"
  ) | tee "${COMPARISON_DIR}/recommendation.txt"

  echo "Uploading consolidated comparison bundle to S3..."
  aws s3 sync \
    "${COMPARISON_DIR}" \
    "s3://${ARTIFACT_BUCKET_NAME}/benchmark-runs/${BENCHMARK_RUN_ID}/comparison/"

  echo "Comparison bundle ready in ${COMPARISON_DIR}"
fi

exit "${EXIT_CODE}"
