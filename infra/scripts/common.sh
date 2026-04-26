#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${ROOT_DIR}/terraform/envs/aws-gpu"
INVENTORY_FILE="${TF_DIR}/inventory.json"
KUBECONFIG_LOCAL="${ROOT_DIR}/kubeconfig.yaml"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_KEY_PATH="${SSH_KEY_PATH:-}"
K3S_LOCAL_PORT="${K3S_LOCAL_PORT:-6443}"
K3S_TUNNEL_PID_FILE="${ROOT_DIR}/.k3s-tunnel.pid"
SSH_ARGS=(-o StrictHostKeyChecking=no)

if [[ -n "${SSH_KEY_PATH}" ]]; then
  SSH_ARGS+=(-i "${SSH_KEY_PATH}")
fi

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

ensure_k3s_tunnel() {
  require_cmd ssh

  if [[ -f "${K3S_TUNNEL_PID_FILE}" ]]; then
    local existing_pid
    existing_pid="$(cat "${K3S_TUNNEL_PID_FILE}")"
    if kill -0 "${existing_pid}" >/dev/null 2>&1; then
      return 0
    fi
    rm -f "${K3S_TUNNEL_PID_FILE}"
  fi

  ssh "${SSH_ARGS[@]}" -f -N \
    -L "${K3S_LOCAL_PORT}:127.0.0.1:6443" \
    "${SSH_USER}@$(controller_ip)"
  local tunnel_pid
  tunnel_pid="$(pgrep -f "127.0.0.1:6443 ${SSH_USER}@$(controller_ip)" | head -n1 || true)"
  if [[ -n "${tunnel_pid}" ]]; then
    echo "${tunnel_pid}" > "${K3S_TUNNEL_PID_FILE}"
  fi
}

run_ssh() {
  local host="$1"
  shift
  require_cmd ssh
  ssh "${SSH_ARGS[@]}" "${SSH_USER}@${host}" "$@"
}

copy_from_controller() {
  local remote_path="$1"
  local local_path="$2"
  require_cmd scp
  scp "${SSH_ARGS[@]}" "${SSH_USER}@$(controller_ip):${remote_path}" "${local_path}"
}
