#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <ecr-image-uri> [platform]" >&2
  echo "Example: $0 999052221400.dkr.ecr.us-east-1.amazonaws.com/gpu-benchmark:latest linux/amd64" >&2
  exit 1
fi

IMAGE_URI="$1"
PLATFORM="${2:-linux/amd64}"
REGISTRY_HOST="${IMAGE_URI%%/*}"
REGISTRY_REGION="$(echo "${REGISTRY_HOST}" | awk -F'.' '{print $4}')"
BUILDER_NAME="gpu-benchmark-builder"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required" >&2
  exit 1
fi

if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER_NAME}" --use
else
  docker buildx use "${BUILDER_NAME}"
fi

echo "Logging into ECR registry ${REGISTRY_HOST}..."
aws ecr get-login-password --region "${REGISTRY_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY_HOST}"

echo "Building and pushing ${IMAGE_URI} for platform ${PLATFORM}..."
docker buildx build \
  --platform "${PLATFORM}" \
  -t "${IMAGE_URI}" \
  --push \
  .

echo "Pushed ${IMAGE_URI}"
