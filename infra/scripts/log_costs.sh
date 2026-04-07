#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd aws
require_cmd jq
require_cmd terraform

mkdir -p "${ROOT_DIR}/artifacts"

START_TIME="${1:-$(date -u -v-1H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '1 hour ago' '+%Y-%m-%dT%H:%M:%SZ')}"
END_TIME="${2:-$(date -u '+%Y-%m-%dT%H:%M:%SZ')}"

aws ec2 describe-instances \
  --instance-ids $(terraform -chdir="${TF_DIR}" output -json | jq -r '.instance_ids.value[]') \
  > "${ROOT_DIR}/artifacts/ec2_instances.json"

jq --arg start "${START_TIME}" --arg end "${END_TIME}" '
  {
    captured_at: now | todate,
    benchmark_window: { start: $start, end: $end },
    instances: [
      .Reservations[].Instances[] |
      {
        instance_id: .InstanceId,
        instance_type: .InstanceType,
        launch_time: .LaunchTime,
        state: .State.Name,
        tags: (
          .Tags // []
          | map({key: .Key, value: .Value})
          | from_entries
        )
      }
    ]
  }
' "${ROOT_DIR}/artifacts/ec2_instances.json" > "${ROOT_DIR}/artifacts/cost_snapshot.json"

aws s3 cp "${ROOT_DIR}/artifacts/cost_snapshot.json" "s3://$(artifact_bucket)/costs/cost_snapshot.json"
echo "Uploaded cost snapshot to s3://$(artifact_bucket)/costs/cost_snapshot.json"
