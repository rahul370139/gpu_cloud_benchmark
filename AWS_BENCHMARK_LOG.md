# AWS Multi-GPU Benchmark Execution Log

**Date:** April 25, 2026 — 05:15 UTC  
**Platform:** AWS EC2 (us-east-1) → k3s → Kubernetes Jobs  
**Operator:** Sahil Mariwala  
**Source Report:** `report.html` at repo root (850 KB, self-contained)  
**Pipeline:** `infra/scripts/run_pipeline.sh provision → bootstrap → deploy → benchmark → log-costs → teardown`

---

## 1. System Specifications

### 1.1 Cluster Topology

| Role | Instance Type | vCPU / RAM | GPU | Hourly Rate | Count |
|------|---------------|-----------|-----|-------------|-------|
| Controller (k3s server) | `t3.large` | 2 vCPU / 8 GB | — | $0.0832 | 1 |
| Worker — A10G class | `g5.xlarge` | 4 vCPU / 16 GB | NVIDIA A10G (24 GB) | $1.006 | 1 |
| Worker — T4 class | `g4dn.xlarge` | 4 vCPU / 16 GB | NVIDIA T4 (16 GB) | $0.526 | 1 |

### 1.2 Software Stack

| Component | Version |
|-----------|---------|
| AMI | Ubuntu 22.04 GPU DLAMI (latest, pulled by SSM parameter) |
| Container runtime | NVIDIA CTK + `runtimeClassName: nvidia` |
| Kubernetes | k3s (lightweight distribution) |
| Container image | `nvcr.io/nvidia/pytorch:24.08-py3` (pulled from ECR `linux/amd64`) |
| PyTorch | 2.4.0 (from base image) |
| Workloads tested | ResNet-50, BERT-base |

### 1.3 IaC

| Layer | File | Status |
|-------|------|--------|
| Networking | `infra/terraform/modules/network/` | Applied |
| Security | `infra/terraform/modules/security/` | Applied |
| Compute | `infra/terraform/modules/compute/` | Applied |
| Composition | `infra/terraform/envs/aws-gpu/` | Applied |
| K8s namespace + ConfigMap | `infra/kubernetes/base/` | Applied |
| K8s monitoring (Prometheus + Pushgateway) | `infra/kubernetes/monitoring/` | Applied |

---

## 2. Benchmark Configuration

### 2.1 Per-Job ConfigMap (`benchmark-shared.yaml`)

```yaml
workloads: [resnet50, bert_base]
batch_sizes: [1, 8, 32, 64]
num_repeats: 3
warmup_iters: 10
benchmark_iters: 100
seed: 42
modes: [inference]
output_dir: /artifacts
prometheus_pushgateway: "http://pushgateway.ml-benchmark.svc:9091"
```

### 2.2 Run Matrix

| Variable | Values | Count |
|----------|--------|------:|
| GPU class | A10G, T4 | 2 |
| Workload | resnet50, bert_base | 2 |
| Batch size | 1, 8, 32, 64 | 4 |
| Mode | inference | 1 |
| Repeats | 1, 2, 3 | 3 |
| **Total runs** | | **48 per GPU class × 2 = 96** |

### 2.3 Kubernetes Dispatch

One Job per GPU class was rendered with `envsubst` from
`infra/kubernetes/base/benchmark-job.yaml` and applied in parallel:

```bash
kubectl apply -f benchmark-run-a10g.yaml
kubectl apply -f benchmark-run-t4.yaml
kubectl wait --for=condition=complete job/benchmark-run-a10g job/benchmark-run-t4 \
  -n ml-benchmark --timeout=2h
```

Each Job pinned itself to its own node pool via:

```yaml
nodeSelector:
  gpu-benchmark/gpu-class: ${GPU_CLASS}
```

---

## 3. Execution Outcome

### 3.1 Headline

| Metric | Value |
|--------|------:|
| Total scheduled runs | 96 |
| **Runs failed** | **0** |
| Total Kubernetes Jobs | 2 (one per GPU class) |
| Job retries | 0 |
| Wall-clock time | ~1 hour (parallel across two GPU classes) |
| Image pull errors | 0 |
| Per-pod artifact upload to S3 | succeeded (every pod) |
| Consolidated comparison report | succeeded |

### 3.2 Per-GPU-Class Aggregates

| GPU | Avg Throughput | Median P95 | Avg GPU Util | Memory Used |
|-----|---------------:|-----------:|-------------:|-------------|
| **A10G** | 45,587 samples/s | 39.4 ms | **78.7%** | partial cap (g5.xlarge has 24 GB) |
| **T4** | 9,906 samples/s | 130.7 ms | **90.9%** | partial cap (g4dn.xlarge has 16 GB) |

### 3.3 ResNet-50 Inference (images/sec, 3-repeat means)

| Batch | A10G | T4 | A10G / T4 | A10G P95 (ms) | T4 P95 (ms) |
|------:|-----:|---:|----------:|--------------:|------------:|
| 1 | 157.9 | 157.9 | 1.00× (tied) | 6.46 | 6.45 |
| 8 | 858.6 | 318.4 | 2.70× | 9.33 | 25.44 |
| 32 | 946.7 | 371.4 | 2.55× | 33.87 | 87.50 |
| 64 | 970.5 | 371.2 | 2.62× | 66.03 | 173.80 |

### 3.4 BERT-base Inference (tokens/sec, 3-repeat means)

| Batch | A10G | T4 | A10G / T4 | A10G P95 (ms) | T4 P95 (ms) |
|------:|-----:|---:|----------:|--------------:|------------:|
| 1 | 65,266 | 18,012 | 3.62× | 7.88 | 28.82 |
| 8 | 91,194 | 20,897 | 4.36× | 45.00 | 197.21 |
| 32 | 101,711 | 19,547 | 5.20× | 161.24 | 842.54 |
| 64 | 103,596 | 19,574 | 5.30× | 316.50 | 1,676.98 |

### 3.5 Reproducibility

Coefficient of variation across the three repeats per (GPU × workload × batch) cell:

| GPU | Min CV | Max CV | Median CV |
|-----|-------:|-------:|----------:|
| A10G | 0.0% | 0.8% | **0.0%** |
| T4 | 0.1% | **1.7%** (resnet50 bs=1) | 0.3% |

**All 16 cells had CV ≤ 1.7%**, and 14 of 16 had CV ≤ 0.5%. The deterministic-seed
pipeline holds up on real cloud GPUs.

---

## 4. Engineering Insights

### 4.1 A10G dominates outside the launch-bound regime

At `bs=1` ResNet-50 the workload is GPU-launch-bound: most time is spent in the
PyTorch dispatcher, kernel launches, and synchronization, not in the actual
matmul. Both GPUs converge to ~158 img/s and the cheaper T4 wins on cost.

Once `bs ≥ 8`, the SM count and memory bandwidth of the A10G (Ampere) pull
ahead by 2.5–5× over the T4 (Turing).

### 4.2 BERT-base scales further than ResNet-50

BERT throughput on A10G grows monotonically: 65K → 91K → 102K → 104K tok/s.
ResNet-50 saturates by `bs=8` (859 img/s) and barely improves at higher batch
sizes (970 at bs=64). The implication for our recommender: **the scoring weight
between throughput and latency should be workload-aware**; a BERT user benefits
from a bigger GPU much more than a ResNet user does.

### 4.3 Why "Value Wins: 0 / 0"

The consolidated report at root level has:

```
Throughput Wins: A10G=7, T4=1
Latency Wins:    A10G=7, T4=1
Value Wins:      A10G=0, T4=0
```

The "Value Wins" column is empty because the cross-GPU regeneration step did not
re-run the cost calculator with `--cost-rates config/gpu_cost_rates.yaml`. This
is **a known gap** documented in `PROJECT_PROGRESS.md` §10.1; rerunning
`scripts/generate_report.py --results-dir <consolidated> --cost-rates ...` will
populate it.

Quick hand-check of cost-efficiency at `bs=32`, BERT inference:

| GPU | Throughput | $/hr | tokens / $ |
|-----|-----------:|-----:|----------:|
| A10G | 101,711 tok/s | $1.006 | **101,103 tok/$** |
| T4 | 19,547 tok/s | $0.526 | 37,162 tok/$ |

A10G is **2.7× more cost-efficient** at this scenario despite being 2× more
expensive per hour. This pattern dominates everywhere except `resnet50 bs=1`
where T4 wins on cost-efficiency too.

### 4.4 A10G utilisation 78.7% vs T4 utilisation 90.9%

Counter-intuitive: the slower GPU shows **higher** utilisation. The reason is
the same scaling story — T4 saturates its compute units at lower throughput,
while A10G has spare capacity. This is a useful signal for the recommender's
"headroom" reasoning ("A10G can handle bigger batch sizes without saturating").

### 4.5 Single failure mode that did not occur

The `nodeSelector: gpu-benchmark/gpu-class=<class>` plus per-pool node labels
applied via Terraform cloud-init cleanly avoided the "wrong-class scheduling"
trap that would have run a Pod meant for A10G on T4. No Pods were ever
scheduled to the wrong node.

---

## 5. Artifact Layout (in S3)

```
s3://<bucket>/benchmark-runs/<run-id>/
├── A10G/
│   └── benchmark-run-a10g-<podname>/
│       ├── benchmark_summary_A10G.csv
│       ├── A10G_resnet50_inference_bs{1,8,32,64}_r{1,2,3}_latencies.csv
│       ├── A10G_resnet50_inference_bs{1,8,32,64}_r{1,2,3}_gpu_metrics.csv
│       ├── A10G_bert_base_inference_bs{1,8,32,64}_r{1,2,3}_latencies.csv
│       ├── A10G_bert_base_inference_bs{1,8,32,64}_r{1,2,3}_gpu_metrics.csv
│       ├── recommendation.json   (single-GPU; not the cross-GPU final)
│       ├── report.html           (single-GPU; not the cross-GPU final)
│       └── run_manifest.json
├── T4/
│   └── benchmark-run-t4-<podname>/
│       └── (same layout for T4)
├── comparison/
│   ├── report.html               (cross-GPU — copied to repo root)
│   ├── recommendation.json       (cross-GPU)
│   └── recommendation.txt
└── costs/
    └── cost_snapshot.json        (EC2 instance metadata)
```

---

## 6. Reproducibility Hand-off

Anyone who clones the repo, configures AWS credentials, and edits
`infra/terraform/envs/aws-gpu/terraform.tfvars` to point at their own VPC/admin
CIDR can reproduce this exact run with:

```bash
cd infra
./scripts/run_pipeline.sh provision
SSH_KEY_PATH=/path/to/key.pem ./scripts/run_pipeline.sh bootstrap
./scripts/run_pipeline.sh deploy
BENCHMARK_IMAGE=<your-ecr-uri> ./scripts/run_pipeline.sh benchmark
./scripts/run_pipeline.sh log-costs
./scripts/run_pipeline.sh teardown
```

The `terraform.tfvars.smoke.example` provides a 1× T4 minimal config (cheapest
sanity check, ~$0.20).

---

## 7. Outstanding Items (See PROJECT_PROGRESS.md §10)

- Re-render `report.html` with `--cost-rates` to populate the **Value Wins** column.
- Run `infra/scripts/run_pipeline.sh fault-inject` to gather recovery data.
- Combine DGX Spark (GB10) + AWS A10G + AWS T4 results into a single
  cross-platform comparison table for the final write-up.
