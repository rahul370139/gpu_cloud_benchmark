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

# Grafana: render the dashboard JSON into a ConfigMap so the file is mounted
# straight into /var/lib/grafana/dashboards inside the Grafana pod (zero-touch
# dashboard provisioning). Then apply the Deployment + Service + datasources.
kubectl create configmap grafana-dashboard-json \
  -n ml-benchmark \
  --from-file=gpu-benchmark.json="${ROOT_DIR}/kubernetes/monitoring/grafana_dashboard.json" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "${ROOT_DIR}/kubernetes/monitoring/grafana.yaml"

echo
echo "Grafana will be reachable in ~30 s via either of:"
echo "  • port-forward (recommended): "
echo "      kubectl port-forward -n ml-benchmark svc/grafana 3000:3000"
echo "      open http://localhost:3000  (admin / admin)"
echo "  • NodePort (if your IP is in admin_cidrs):"
echo "      open http://\$(jq -r '.controller_public_ip' \\"
echo "        '${ROOT_DIR}/terraform/envs/aws-gpu/inventory.json'):30030"
echo "Prometheus datasource is pre-provisioned; the GPU Benchmark dashboard auto-loads."
