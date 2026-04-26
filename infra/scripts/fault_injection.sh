#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd kubectl
ensure_k3s_tunnel
export KUBECONFIG="${KUBECONFIG_LOCAL}"

TARGET_NODE="${1:-}"

if [[ -z "${TARGET_NODE}" ]]; then
  TARGET_NODE="$(kubectl get nodes -l role=worker -o jsonpath='{.items[0].metadata.name}')"
fi

if [[ -z "${TARGET_NODE}" ]]; then
  echo "No worker node available for fault injection" >&2
  exit 1
fi

echo "Cordoning ${TARGET_NODE}"
kubectl cordon "${TARGET_NODE}"

echo "Deleting benchmark pods on ${TARGET_NODE} to simulate disruption"
kubectl delete pod -n ml-benchmark -l app=benchmark --field-selector "spec.nodeName=${TARGET_NODE}" --ignore-not-found

echo "Draining ${TARGET_NODE}"
kubectl drain "${TARGET_NODE}" --ignore-daemonsets --delete-emptydir-data --force

echo "Waiting 30 seconds to observe recovery behavior"
sleep 30

echo "Uncordoning ${TARGET_NODE}"
kubectl uncordon "${TARGET_NODE}"
