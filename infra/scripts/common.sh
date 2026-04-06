#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${ROOT_DIR}/terraform/envs/aws-gpu"
INVENTORY_FILE="${TF_DIR}/inventory.json"
KUBECONFIG_LOCAL="${ROOT_DIR}/kubeconfig.yaml"
SSH_USER="${SSH_USER:-ubuntu}"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

load_inventory() {
  require_cmd jq
  if [[ ! -f "${INVENTORY_FILE}" ]]; then
    echo "Inventory file not found: ${INVENTORY_FILE}" >&2
    exit 1
  fi
}

controller_ip() {
  load_inventory
  jq -r '.controller_public_ip' "${INVENTORY_FILE}"
}

artifact_bucket() {
  load_inventory
  jq -r '.artifact_bucket_name' "${INVENTORY_FILE}"
}

run_ssh() {
  local host="$1"
  shift
  require_cmd ssh
  ssh -o StrictHostKeyChecking=no "${SSH_USER}@${host}" "$@"
}

copy_from_controller() {
  local remote_path="$1"
  local local_path="$2"
  require_cmd scp
  scp -o StrictHostKeyChecking=no "${SSH_USER}@$(controller_ip):${remote_path}" "${local_path}"
}
