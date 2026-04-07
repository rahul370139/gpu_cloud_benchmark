#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${ROOT_DIR}/terraform/envs/aws-gpu"

command -v terraform >/dev/null 2>&1 || {
  echo "terraform is required" >&2
  exit 1
}

cd "${TF_DIR}"
terraform init
terraform apply -auto-approve
