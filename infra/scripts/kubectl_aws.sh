#!/usr/bin/env bash
# Run kubectl against the k3s API with the correct kubeconfig and an active
# SSH tunnel. Use this instead of plain `kubectl` when your shell is in any
# directory — it always resolves paths relative to the infra/ root.
#
#   cd infra
#   ./scripts/kubectl_aws.sh get pods -n ml-benchmark
#   ./scripts/kubectl_aws.sh port-forward -n ml-benchmark svc/grafana 3000:3000
#
# If you see "connection refused localhost:8080", you were using a bad
# KUBECONFIG path or no kubeconfig file — run bootstrap first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_cmd kubectl ssh

if [[ ! -f "${INVENTORY_FILE}" ]]; then
  echo "Missing ${INVENTORY_FILE}. Run ./scripts/run_pipeline.sh provision first." >&2
  exit 1
fi

if [[ ! -f "${KUBECONFIG_LOCAL}" ]]; then
  echo "Missing ${KUBECONFIG_LOCAL}." >&2
  echo "Run:  cd \"${ROOT_DIR}\" && ./scripts/run_pipeline.sh bootstrap" >&2
  exit 1
fi

ensure_k3s_tunnel
export KUBECONFIG="${KUBECONFIG_LOCAL}"
exec kubectl "$@"
