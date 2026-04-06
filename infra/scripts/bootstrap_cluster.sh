#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd kubectl

echo "Waiting for k3s control plane to expose kubeconfig..."
until run_ssh "$(controller_ip)" "sudo test -f /etc/rancher/k3s/k3s.yaml"; do
  sleep 10
done

copy_from_controller "/etc/rancher/k3s/k3s.yaml" "${KUBECONFIG_LOCAL}"
perl -0pi -e "s/127.0.0.1/$(controller_ip)/g" "${KUBECONFIG_LOCAL}"

export KUBECONFIG="${KUBECONFIG_LOCAL}"

kubectl wait --for=condition=Ready nodes --all --timeout=15m
kubectl get nodes -o wide
