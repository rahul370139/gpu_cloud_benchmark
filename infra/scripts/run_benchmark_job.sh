#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd kubectl
export KUBECONFIG="${KUBECONFIG_LOCAL}"

kubectl delete job benchmark-run -n ml-benchmark --ignore-not-found
kubectl apply -f "${ROOT_DIR}/kubernetes/base/benchmark-job.yaml"
kubectl wait --for=condition=complete job/benchmark-run -n ml-benchmark --timeout=2h
kubectl logs job/benchmark-run -n ml-benchmark
