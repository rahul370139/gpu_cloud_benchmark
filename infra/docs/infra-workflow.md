# Infrastructure Workflow

## What this covers

Sahil's infrastructure slice is responsible for:

- declarative cloud provisioning with Terraform
- GPU-capable Kubernetes setup
- repeatable benchmark orchestration
- teardown and spend control
- simple failure injection to measure recovery

## Benchmark lifecycle

1. `provision`: create VPC, subnet, security group, S3 bucket, controller node, and worker nodes.
2. `bootstrap`: wait for `k3s`, fetch kubeconfig, and verify node readiness.
3. `deploy`: install the benchmark namespace, the GPU job definition, and Prometheus.
4. `benchmark`: run Rahul's containerized workload as a Kubernetes Job.
5. `log-costs`: snapshot instance metadata and upload spend-related tags to S3.
6. `fault-inject`: drain a worker node and observe job recovery behavior.
7. `teardown`: destroy the stack to prevent additional charges.

## Fault-injection experiment

The provided disruption helper performs a controlled node drain:

- cordon the target worker
- delete benchmark pods on that node
- drain the node
- wait briefly for re-scheduling and recovery
- uncordon the node

This is intentionally simple, but it is enough to measure:

- job completion rate
- extra elapsed time after disruption
- cost overhead caused by recovery
