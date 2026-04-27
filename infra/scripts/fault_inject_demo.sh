#!/usr/bin/env bash
#
# End-to-end fault-injection smoke run on a fresh, throw-away AWS cluster.
#
#   provision (1×T4)
#       └─ trap teardown so we ALWAYS shut the cluster down on exit
#   bootstrap → deploy
#   dispatch a tiny benchmark Job (a few iters, takes ~2 min)
#   wait for benchmark pod to be Running
#   fault-inject (cordon + delete pod + drain + 30 s + uncordon)
#   wait for Job to recover and complete (Job controller respawns the pod)
#   log costs + capture timing + teardown
#
# Cost target: ~$0.15-0.25 (T4 g4dn.xlarge $0.526/h + t3.large controller
# $0.083/h, ~15 min wall-clock).
#
# Usage:
#   cd infra
#   cp terraform/envs/aws-gpu/terraform.tfvars.smoke.example \
#      terraform/envs/aws-gpu/terraform.tfvars
#   # edit the placeholders (AMI, key_name, admin_cidrs, bucket_name,
#   # cluster_token) — see comments inside the example file.
#   ./scripts/fault_inject_demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
TFVARS="${ROOT_DIR}/terraform/envs/aws-gpu/terraform.tfvars"
ARTIFACTS_ROOT="${ROOT_DIR}/artifacts/fault_inject_demo"
TIMING_LOG="${ARTIFACTS_ROOT}/timing.log"

mkdir -p "${ARTIFACTS_ROOT}"

log()  { echo -e "\033[1;36m[fault-inject-demo]\033[0m $*"; }
warn() { echo -e "\033[1;33m[fault-inject-demo]\033[0m $*" >&2; }
fail() { echo -e "\033[1;31m[fault-inject-demo]\033[0m $*" >&2; exit 1; }

t0=$(date +%s)
record_step() {
  local name="$1" started="$2"
  local now elapsed
  now=$(date +%s)
  elapsed=$(( now - started ))
  printf '%-22s  %4d s\n' "${name}" "${elapsed}" >> "${TIMING_LOG}"
  log "${name} took ${elapsed}s"
}

##############################################################################
# 0. Pre-flight                                                              #
##############################################################################

[[ -f "${TFVARS}" ]] || fail "Missing ${TFVARS}. Copy
   terraform/envs/aws-gpu/terraform.tfvars.smoke.example
to that path and fill in the placeholders before re-running."

if grep -q 'YOUR\.PUBLIC\.IP\|replace-me\|ami-xxxxxxxxxxxxxxxxx' "${TFVARS}"; then
  fail "${TFVARS} still has placeholder values. Fill them in first."
fi

for cmd in terraform kubectl jq aws ssh envsubst; do
  command -v "${cmd}" >/dev/null 2>&1 || fail "Missing required command: ${cmd}"
done

##############################################################################
# 1. Set up the safety net                                                   #
##############################################################################

cleanup() {
  local rc=$?
  warn "TRAP fired (rc=${rc}) — running teardown to prevent runaway cost."
  set +e
  "${SCRIPT_DIR}/teardown.sh"
  record_step "teardown" "${teardown_t0:-${t0}}"
  log "Total wall-clock: $(( $(date +%s) - t0 ))s"
  log "Timing summary written to ${TIMING_LOG}"
  exit "${rc}"
}
trap cleanup EXIT INT TERM

##############################################################################
# 2. Provision the smoke cluster                                              #
##############################################################################

step_t0=$(date +%s)
log "1/6 provision (terraform apply on 1×T4 smoke profile)..."
"${SCRIPT_DIR}/provision.sh"
record_step "provision" "${step_t0}"

##############################################################################
# 3. Bootstrap k3s + deploy monitoring stack                                  #
##############################################################################

step_t0=$(date +%s)
log "2/6 bootstrap (open SSH tunnel + fetch kubeconfig)..."
"${SCRIPT_DIR}/bootstrap_cluster.sh"
record_step "bootstrap" "${step_t0}"

step_t0=$(date +%s)
log "3/6 deploy (namespace + ConfigMap + Prometheus + Pushgateway + Grafana)..."
"${SCRIPT_DIR}/deploy_benchmark_stack.sh"
record_step "deploy" "${step_t0}"

# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
ensure_k3s_tunnel
export KUBECONFIG="${KUBECONFIG_LOCAL}"

##############################################################################
# 4. Use a TINY benchmark config so we can fault-inject mid-run               #
##############################################################################

TMP_CFG="$(mktemp)"
cat > "${TMP_CFG}" <<'YAML'
# Smoke config for fault-injection demo. ResNet-50 only, 4 iters × 1 batch ×
# 1 repeat × inference. Long enough (~90 s) to inject mid-run, short enough
# that the recovery still completes in a few minutes.
workloads:
  - resnet50
batch_sizes:
  - 1
num_repeats: 1
warmup_iters: 2
benchmark_iters: 4
seed: 42
modes:
  - inference
output_dir: results/
prometheus_pushgateway: "http://pushgateway.ml-benchmark.svc:9091"
YAML

kubectl create configmap benchmark-config -n ml-benchmark \
  --from-file=benchmark_config.yaml="${TMP_CFG}" \
  --dry-run=client -o yaml | kubectl apply -f -
rm -f "${TMP_CFG}"

##############################################################################
# 5. Dispatch benchmark Job in the background                                 #
##############################################################################

step_t0=$(date +%s)
log "4/6 dispatch benchmark Job (T4) and wait for pod Running..."

# Render and apply the Job (mirrors run_benchmark_job.sh setup, no kubectl wait)
load_inventory
BENCHMARK_RUN_ID="$(jq -r '.benchmark_run_id' "${INVENTORY_FILE}")"
BENCHMARK_EXECUTION_ID="fault-demo-$(date -u '+%Y%m%dT%H%M%SZ')"
BENCHMARK_IMAGE="${BENCHMARK_IMAGE:-ghcr.io/example/ml-benchmark:latest}"
ARTIFACT_BUCKET_NAME="$(jq -r '.artifact_bucket_name' "${INVENTORY_FILE}")"
AWS_REGION="$(jq -r '.region' "${INVENTORY_FILE}")"
GPU_CLASS="T4"
GPU_CLASS_LOWER="t4"
JOB_NAME="benchmark-run-${GPU_CLASS_LOWER}"

GPU_CLASS="${GPU_CLASS}" GPU_CLASS_LOWER="${GPU_CLASS_LOWER}" \
BENCHMARK_RUN_ID="${BENCHMARK_RUN_ID}" \
BENCHMARK_EXECUTION_ID="${BENCHMARK_EXECUTION_ID}" \
BENCHMARK_IMAGE="${BENCHMARK_IMAGE}" \
ARTIFACT_BUCKET_NAME="${ARTIFACT_BUCKET_NAME}" \
AWS_REGION="${AWS_REGION}" \
envsubst < "${ROOT_DIR}/kubernetes/base/benchmark-job.yaml" \
  > "${ARTIFACTS_ROOT}/job.yaml"

kubectl delete job "${JOB_NAME}" -n ml-benchmark --ignore-not-found
kubectl apply -f "${ARTIFACTS_ROOT}/job.yaml"

log "Waiting for benchmark pod to be Running (max 5 min)..."
kubectl wait --for=condition=Ready pod \
  -n ml-benchmark \
  -l "job-name=${JOB_NAME}" \
  --timeout=300s
record_step "benchmark-running" "${step_t0}"

##############################################################################
# 6. Inject the fault                                                         #
##############################################################################

step_t0=$(date +%s)
log "5/6 inject fault (cordon + delete pod + drain + 30s wait + uncordon)..."
"${SCRIPT_DIR}/fault_injection.sh" | tee "${ARTIFACTS_ROOT}/fault_injection.log"
record_step "fault-inject" "${step_t0}"

step_t0=$(date +%s)
log "Waiting for Job to recover and complete (Job controller will spawn a new pod)..."
if kubectl wait --for=condition=complete "job/${JOB_NAME}" -n ml-benchmark --timeout=15m; then
  log "Job completed after recovery"
  record_step "recovery+complete" "${step_t0}"
else
  warn "Job did NOT complete within 15 min after fault — recovery failed"
  kubectl describe "job/${JOB_NAME}" -n ml-benchmark > "${ARTIFACTS_ROOT}/job_describe.txt" || true
  kubectl logs "job/${JOB_NAME}" -n ml-benchmark --tail=200 > "${ARTIFACTS_ROOT}/job_logs.txt" || true
  record_step "recovery+complete" "${step_t0}"
fi

##############################################################################
# 7. Capture costs + logs, then teardown via trap                             #
##############################################################################

step_t0=$(date +%s)
log "6/6 log-costs..."
"${SCRIPT_DIR}/log_costs.sh"
record_step "log-costs" "${step_t0}"

log "Capturing pod logs and Job state..."
kubectl logs "job/${JOB_NAME}" -n ml-benchmark --tail=500 > "${ARTIFACTS_ROOT}/job_logs.txt" 2>&1 || true
kubectl describe "job/${JOB_NAME}" -n ml-benchmark > "${ARTIFACTS_ROOT}/job_describe.txt" 2>&1 || true
kubectl get events -n ml-benchmark --sort-by='.lastTimestamp' \
  > "${ARTIFACTS_ROOT}/events.txt" 2>&1 || true

teardown_t0=$(date +%s)
log "All steps complete — exiting cleanly will trigger the teardown trap."
