#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_pipeline.sh <command>

Commands:
  provision     terraform apply for the AWS GPU stack
  bootstrap     fetch kubeconfig and verify cluster readiness
  deploy        install namespaces, benchmark job, and monitoring manifests
  benchmark     run the benchmark Kubernetes Job
  log-costs     collect EC2 metadata and upload a cost snapshot
  fault-inject  run the worker disruption experiment
  teardown      terraform destroy the stack
EOF
}

COMMAND="${1:-}"

case "${COMMAND}" in
  provision)
    "${SCRIPT_DIR}/provision.sh"
    ;;
  bootstrap)
    "${SCRIPT_DIR}/bootstrap_cluster.sh"
    ;;
  deploy)
    "${SCRIPT_DIR}/deploy_benchmark_stack.sh"
    ;;
  benchmark)
    "${SCRIPT_DIR}/run_benchmark_job.sh"
    ;;
  log-costs)
    "${SCRIPT_DIR}/log_costs.sh"
    ;;
  fault-inject)
    "${SCRIPT_DIR}/fault_injection.sh"
    ;;
  teardown)
    "${SCRIPT_DIR}/teardown.sh"
    ;;
  *)
    usage
    exit 1
    ;;
esac
