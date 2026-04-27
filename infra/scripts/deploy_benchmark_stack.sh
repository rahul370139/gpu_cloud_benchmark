#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

require_cmd kubectl
ensure_k3s_tunnel
export KUBECONFIG="${KUBECONFIG_LOCAL}"

kubectl apply -f "${ROOT_DIR}/kubernetes/base/namespace.yaml"
kubectl create configmap benchmark-config \
  -n ml-benchmark \
  --from-file=benchmark_config.yaml="${PROJECT_DIR}/config/benchmark_config.yaml" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "${ROOT_DIR}/kubernetes/monitoring/prometheus-configmap.yaml"
kubectl apply -f "${ROOT_DIR}/kubernetes/monitoring/prometheus.yaml"
kubectl apply -f "${ROOT_DIR}/../k8s/prometheus/pushgateway-deploy.yaml"
