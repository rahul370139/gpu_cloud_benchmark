#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd kubectl
export KUBECONFIG="${KUBECONFIG_LOCAL}"

kubectl apply -f "${ROOT_DIR}/kubernetes/base/namespace.yaml"
kubectl apply -f "${ROOT_DIR}/kubernetes/base/benchmark-job.yaml"
kubectl apply -f "${ROOT_DIR}/kubernetes/monitoring/prometheus-configmap.yaml"
kubectl apply -f "${ROOT_DIR}/kubernetes/monitoring/prometheus.yaml"
