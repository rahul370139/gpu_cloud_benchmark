# GPU Cloud Benchmark

Containerized, reproducible benchmarking of ML workloads across cloud GPUs.

## Overview

This framework packages ML workloads into Docker images, runs identical experiments
across multiple cloud GPU instance types, captures standardized metrics (throughput,
latency, utilization, cost), and produces reproducible, comparable reports.

## Quick Start

### Local (CPU, for testing)

```bash
pip install -r requirements.txt
python -m src.runner --config config/benchmark_config.yaml --device cpu
```

### Docker (GPU)

```bash
docker build -t gpu-benchmark .
docker run --gpus all -v $(pwd)/results:/app/results gpu-benchmark
```

### Kubernetes

```bash
kubectl apply -f k8s/benchmark-job.yaml
```

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
├── k8s/                     # Kubernetes Job manifests, Prometheus/Grafana configs
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

## Team

- **Rahul Sharma** — Benchmarking, containerization, metrics, analysis, reproducibility
- **Sahil Mariwala** — Infrastructure as code, Terraform, Kubernetes orchestration, fault injection
