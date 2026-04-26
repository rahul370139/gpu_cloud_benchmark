# GPU Cloud Benchmark

Containerized, reproducible benchmarking of ML workloads across cloud GPUs.

## Overview

This framework packages ML workloads into Docker images, runs identical experiments
across multiple cloud GPU instance types, captures standardized metrics (throughput,
latency, utilization, cost), and produces reproducible, comparable reports.

## Quick Start

### Python Version

Use Python `3.11` or `3.12` for local development. The pinned `torch==2.4.0`
dependency does not install on Python `3.13`.

### Local (CPU, for testing)

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.runner --config config/benchmark_config_local.yaml --device cpu
```

### Docker (GPU)

```bash
docker build -t gpu-benchmark .
docker run --gpus all -v $(pwd)/results:/app/results gpu-benchmark
```

If you are building on Apple Silicon and pushing to AWS for `amd64` EC2 nodes,
publish the image with `buildx` for `linux/amd64` instead of a default local build:

```bash
chmod +x scripts/build_push_ecr.sh
./scripts/build_push_ecr.sh 999052221400.dkr.ecr.us-east-1.amazonaws.com/gpu-benchmark:latest
```

### Kubernetes (multi-GPU fleet)

End-to-end pipeline against AWS — provisions one pool per GPU class, runs
a benchmark Job per class concurrently, aggregates results, and tears down:

```bash
cd infra
./scripts/run_pipeline.sh provision     # terraform apply
./scripts/run_pipeline.sh bootstrap     # fetch kubeconfig
./scripts/run_pipeline.sh deploy        # install namespace/config/monitoring
./scripts/run_pipeline.sh benchmark     # dispatch Jobs in parallel
./scripts/run_pipeline.sh log-costs     # save EC2 metadata snapshot
./scripts/run_pipeline.sh teardown      # terraform destroy
```

Configure the GPU fleet in `infra/terraform/envs/aws-gpu/terraform.tfvars`
via the `worker_pools` list (one entry per GPU class). See
`terraform.tfvars.example` and `terraform.tfvars.smoke.example`.

When running on AWS, make sure the benchmark image pushed to ECR matches the
cluster architecture. The current EC2 worker setup uses `linux/amd64`.

## Project Structure

```
gpu_cloud_benchmark/
├── config/                  # Benchmark and cost configuration
├── src/
│   ├── workloads/           # ML workload definitions (ResNet-50, BERT)
│   ├── metrics/             # GPU metrics collection, timing, Prometheus export
│   ├── cost/                # Cost-per-throughput calculations
│   ├── analysis/            # Post-run analysis, visualization, report generation
│   ├── reproducibility/     # Seed management, checksums, environment capture
│   └── runner.py            # Main benchmark orchestrator
├── scripts/                 # Docker entrypoint, preflight checks, CLI tools
├── infra/                   # Terraform (AWS) + Kubernetes manifests + pipeline scripts
│   ├── terraform/           # VPC, security, compute (multi-GPU worker pools)
│   ├── kubernetes/          # Namespace, PVC/ConfigMap, benchmark Job template
│   └── scripts/             # run_pipeline.sh (provision/deploy/benchmark/teardown)
├── k8s/prometheus/          # Prometheus/Grafana configs
├── tests/                   # Unit tests (CPU-compatible)
└── notebooks/               # Interactive analysis notebooks
```

## Workloads

| Workload | Model | Input Shape | Metric |
|----------|-------|-------------|--------|
| Vision | ResNet-50 | (B, 3, 224, 224) | images/sec |
| NLP | BERT-base | (B, 512) token IDs | tokens/sec |

## Metrics Collected

- **Throughput**: samples/sec (images or tokens)
- **Latency**: p50, p95, p99 in milliseconds
- **GPU utilization**: utilization %, memory used/total, temperature, power
- **Cost efficiency**: throughput-per-dollar, cost-per-1K-samples
- **Reproducibility**: coefficient of variation across repeated runs

## Generating Reports

After a benchmark run completes:

```bash
python scripts/generate_report.py --results-dir results/ --output report.html
```

## Reproducibility

Every run captures:
- Random seeds (torch, numpy, python, CUDA)
- SHA-256 checksums of Docker image, model weights, and data
- Full environment snapshot (driver versions, pip freeze, OS info)
- All stored in `results/run_manifest.json`

## Configuration

Edit `config/benchmark_config.yaml` to customize:
- Workloads, batch sizes, iteration counts
- Number of repeated runs
- Output directory and Prometheus endpoint

Edit `config/gpu_cost_rates.yaml` to update GPU hourly rates.

## Custom Workloads

Users can benchmark their own workload classes instead of only the built-in
`resnet50` and `bert_base` examples.

1. Add a workload class under [user_workloads](/Users/sahilmariwala/dev/msml606/msml605/gpu_cloud_benchmark/user_workloads) that subclasses [BaseWorkload](/Users/sahilmariwala/dev/msml606/msml605/gpu_cloud_benchmark/src/workloads/base.py).
2. Register it in the config:

```yaml
custom_workloads:
  my_model: "user_workloads.my_model:MyModelWorkload"

workloads:
  - my_model
```

3. Run the same benchmark pipeline. The framework will reuse the existing
timing, GPU metrics, cost analysis, reporting, and recommendation steps.

For one-off experiments, you can skip editing YAML and pass a workload class
directly to the runner:

```bash
python -m src.runner \
  --config config/benchmark_config_local.yaml \
  --device cpu \
  --workload-target user_workloads.example_mlp:ExampleMLPWorkload \
  --workload-name my_custom_workload
```

See [user_workloads/example_mlp.py](/Users/sahilmariwala/dev/msml606/msml605/gpu_cloud_benchmark/user_workloads/example_mlp.py) and [user_workloads/template.py](/Users/sahilmariwala/dev/msml606/msml605/gpu_cloud_benchmark/user_workloads/template.py) for a starting point.

## Team

- **Rahul Sharma** — Benchmarking, containerization, metrics, analysis, reproducibility
- **Sahil Mariwala** — Infrastructure as code, Terraform, Kubernetes orchestration, fault injection
