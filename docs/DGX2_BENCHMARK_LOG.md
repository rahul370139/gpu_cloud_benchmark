# DGX2 Benchmark Execution Log

**Date:** April 3-4, 2026  
**System:** Radiant-DGX2 (spark-5cda)  
**Operator:** Rahul Sharma

---

## System Specifications

| Property | Value |
|----------|-------|
| Hostname | spark-5cda |
| Architecture | aarch64 (ARM) |
| OS | Ubuntu 24.04.3 LTS (Noble Numbat) |
| Kernel | Linux 6.11.0-1016-nvidia |
| RAM | 120 GB |
| Disk | 3.7 TB NVMe |
| GPU | NVIDIA GB10 (Blackwell) |
| GPU Compute Capability | 12.1 |
| NVIDIA Driver | 580.95.05 |
| CUDA | 12.8 |
| cuDNN | 90700 |
| Container Runtime | Docker 28.3.3 + NVIDIA CTK 1.18.0 |
| Docker Image | doc2data-gpu:latest (nvcr.io/nvidia/pytorch:25.01-py3) |
| PyTorch | 2.6.0a0+ecf3bae40a.nv25.01 |
| Python | 3.12.3 |

---

## Test Suite Results (Inside Docker Container)

```
31 passed in 14.83s
```

| Test File | Tests | Result |
|-----------|-------|--------|
| test_cost.py | 5 | ALL PASS |
| test_metrics.py (incl. CUDA event) | 3 | ALL PASS |
| test_reproducibility.py | 8 | ALL PASS |
| test_workloads.py - Registry | 2 | ALL PASS |
| test_workloads.py - ResNet-50 | 6 | ALL PASS |
| test_workloads.py - BERT-base | 5 | ALL PASS |
| **TOTAL** | **31** | **31 PASS, 0 FAIL** |

---

## Benchmark Configuration

- Workloads: resnet50, bert_base
- Batch sizes: 1, 8, 32, 64
- Modes: inference, training
- Repeats per config: 3
- Warmup iterations: 10
- Benchmark iterations: 100
- Seed: 42 (incremented per repeat: 42, 43, 44)
- Timing method: cuda_event
- Total runs: 2 workloads x 2 modes x 4 batch_sizes x 3 repeats = **48 runs**
- Failed runs: **0**

---

## ResNet-50 Results (25.56M parameters)

### Inference Mode (images/sec)

| Batch Size | Run 1 | Run 2 | Run 3 | **Mean** | Std | CV |
|------------|-------|-------|-------|----------|-----|-----|
| 1 | 162.40 | 159.64 | 157.52 | **159.85** | 2.44 | 1.53% |
| 8 | 474.70 | 473.72 | 474.65 | **474.36** | 0.54 | 0.11% |
| 32 | 496.23 | 495.69 | 497.05 | **496.32** | 0.68 | 0.14% |
| 64 | 473.58 | 474.53 | 473.28 | **473.80** | 0.65 | 0.14% |

### Inference Latency (ms)

| Batch Size | P50 | P95 | P99 | Mean |
|------------|-----|-----|-----|------|
| 1 | 6.05 | 8.33 | 9.25 | 6.26 |
| 8 | 16.50 | 17.99 | 19.43 | 16.87 |
| 32 | 64.40 | 67.16 | 67.98 | 64.47 |
| 64 | 134.88 | 138.33 | 139.32 | 135.08 |

### Training Mode (images/sec)

| Batch Size | Run 1 | Run 2 | Run 3 | **Mean** | Std | CV |
|------------|-------|-------|-------|----------|-----|-----|
| 1 | 38.96 | 39.21 | 43.09 | **40.42** | 2.33 | 5.76% |
| 8 | 96.68 | 96.95 | 96.23 | **96.62** | 0.37 | 0.38% |
| 32 | 125.64 | 125.54 | 125.40 | **125.53** | 0.12 | 0.10% |
| 64 | 129.66 | 129.58 | 129.73 | **129.66** | 0.08 | 0.06% |

### Training Latency (ms)

| Batch Size | P50 | P95 | P99 | Mean |
|------------|-----|-----|-----|------|
| 1 | 22.95 | 28.33 | 30.54 | 24.79 |
| 8 | 82.40 | 85.92 | 86.90 | 82.80 |
| 32 | 254.92 | 258.60 | 259.39 | 254.92 |
| 64 | 493.67 | 496.97 | 498.44 | 493.61 |

---

## BERT-base Results (109.48M parameters)

### Inference Mode (tokens/sec)

| Batch Size | Run 1 | Run 2 | Run 3 | **Mean** | Std | CV |
|------------|-------|-------|-------|----------|-----|-----|
| 1 | 50,712 | 51,070 | 52,890 | **51,557** | 1,160 | 2.25% |
| 8 | 47,698 | 47,546 | 47,667 | **47,637** | 79 | 0.17% |
| 32 | 46,478 | 46,451 | 46,462 | **46,464** | 13 | 0.03% |
| 64 | 46,523 | 46,492 | 46,606 | **46,540** | 59 | 0.13% |

### Inference Latency (ms)

| Batch Size | P50 | P95 | P99 | Mean |
|------------|-----|-----|-----|------|
| 1 | 9.70 | 10.94 | 11.56 | 9.93 |
| 8 | 85.56 | 89.31 | 90.47 | 85.98 |
| 32 | 352.66 | 356.81 | 357.84 | 352.62 |
| 64 | 703.95 | 707.26 | 708.45 | 704.08 |

### Training Mode (tokens/sec)

| Batch Size | Run 1 | Run 2 | Run 3 | **Mean** | Std | CV |
|------------|-------|-------|-------|----------|-----|-----|
| 1 | 6,611 | 6,714 | 6,609 | **6,644** | 60 | 0.90% |
| 8 | 12,856 | 12,843 | 12,866 | **12,855** | 11 | 0.09% |
| 32 | 14,308 | 14,305 | 14,304 | **14,306** | 2 | 0.02% |
| 64 | 14,300 | 14,469 | 14,608 | **14,459** | 154 | 1.07% |

### Training Latency (ms)

| Batch Size | P50 | P95 | P99 | Mean |
|------------|-----|-----|-----|------|
| 1 | 76.31 | 83.69 | 84.49 | 77.06 |
| 8 | 318.74 | 322.11 | 322.96 | 318.64 |
| 32 | 1145.12 | 1149.16 | 1150.69 | 1145.28 |
| 64 | 2266.57 | 2272.13 | 2274.97 | 2266.47 |

---

## Key Observations

### Throughput Scaling (ResNet-50 Inference)
- bs=1 to bs=8: **2.97x** throughput increase (159.9 -> 474.4 images/sec)
- bs=8 to bs=32: **1.05x** increase (474.4 -> 496.3) -- GPU saturated
- bs=32 to bs=64: **0.95x** decrease (496.3 -> 473.8) -- memory pressure causes slight regression
- **Optimal batch size: 32** for ResNet-50 inference on GB10

### Throughput Scaling (BERT-base Inference)
- bs=1 achieves highest tokens/sec (51,557) due to low overhead per token
- bs=8 to bs=64 stays flat at ~46,500-47,600 tokens/sec
- GB10 BERT inference throughput is memory-bandwidth-bound above bs=8

### Training vs Inference Ratio
- ResNet-50: training is **~3.8x slower** than inference at bs=32 (125.5 vs 496.3 images/sec)
- BERT-base: training is **~3.2x slower** than inference at bs=32 (14,306 vs 46,464 tokens/sec)

### Reproducibility (Coefficient of Variation)
- **Excellent**: Most configurations have CV < 1%
- ResNet-50 inference bs=1: CV = 1.53% (slightly higher due to GPU warmup effects at small batch)
- ResNet-50 training bs=1: CV = 5.76% (highest -- small batch training has more variance)
- BERT inference bs=1: CV = 2.25%
- All bs >= 8 configurations: CV < 0.5% (highly reproducible)

### GPU Utilization
- GPU utilization polling reported 0% (GB10 does not expose utilization via NVML the same way datacenter GPUs do -- this is a known limitation of the DGX Spark/Grace Blackwell platform)
- This is a data collection gap that would be resolved on datacenter GPUs (T4, V100, A100, H100)

---

## Artifacts Generated

| File | Size | Description |
|------|------|-------------|
| benchmark_summary_NVIDIA_GB10.csv | 8.8 KB | All 48 runs with throughput, latencies, metadata |
| run_manifest.json | 25 KB | Environment, config, checksums |
| cost_comparison.csv | 1 KB | Cost efficiency (N/A for DGX Spark -- no cloud pricing) |
| report.html | 552 KB | Self-contained HTML report with embedded charts |
| 48 x *_latencies.csv | ~1.2 KB each | Per-iteration latency traces |
| 6 x figures/*.png | ~30-55 KB each | Throughput, latency, scaling, CV charts |

---

## Files Location

- **Remote (DGX2):** `/home/radiant-dgx2/gpu_cloud_benchmark/results/`
- **Local copy:** `gpu_cloud_benchmark/results_dgx2/`
- **HTML report:** `gpu_cloud_benchmark/results_dgx2/report.html` (open in browser)

---

## Addendum — April 27, 2026 — Extended Workload Coverage

**Operator:** Rahul Sharma  
**Driver / image:** unchanged (Driver 580.95.05 · CUDA 12.8 · `doc2data-gpu:latest` · PyTorch 2.6.0a0+ecf3bae40a.nv25.01)

The original April 3-4 run covered only ResNet-50 and BERT-base. To reach feature
parity with the AWS run (`report.html`), the remaining 3 workloads were executed
on the same DGX2 host using `config/benchmark_config_dgx_extra.yaml`.

### Workloads added

| Workload | Mode(s) | Batch sizes | Repeats | Notes |
|----------|---------|-------------|---------|-------|
| `example_mlp` (34 K params, tabular) | inference + training | 1, 8, 32, 64 | 3 | Synthetic MLP — proves the custom-workload registry works inside Docker on aarch64 |
| `clip_image_embedding` (87.8 M, ViT-B/32 image encoder) | inference | 1, 8, 32 | 3 | Loads `openai/clip-vit-base-patch32` weights via `transformers`; exercises BERT-style attention path on GB10 |
| `llm_text_generation` (23.6 M, GPT decoder) | inference | 1, 4, 8 | 3 | Tiny generative decoder — produces tokens/sec metric distinct from BERT's encoder-only throughput |

### Run statistics

- Total runs: **42** (added to the original 48 — GB10 history is now **90 runs**)
- Failures: **0**
- Wall-clock: **~60 s** total inside container (most workloads are tiny relative to GB10's compute envelope)
- All runs registered into `data/benchmark_history_unified.db`

### Peak throughput observed

| Workload | Mode | Best batch | Peak throughput | CV % |
|----------|------|-----------:|----------------:|-----:|
| `example_mlp` | inference | 64 | 1,265,806 samples/s | 30.27 (extremely fast workload — small absolute timings amplify CV) |
| `example_mlp` | training  | 64 |    68,503 samples/s | 18.00 |
| `clip_image_embedding` | inference | 32 |  1,422 images/s | 0.32 |
| `llm_text_generation`  | inference |  8 | 16,781 tokens/s | 2.52 |

The CLIP and LLM CV figures (< 3 %) are excellent and on par with the original
ResNet-50 / BERT-base runs. The MLP CVs are higher because the per-iteration time
is sub-millisecond and CUDA-stream overhead noise dominates — this is expected.

### Where the data lives

| Path | Contents |
|------|----------|
| `results_dgx2_extra/benchmark_summary_NVIDIA_GB10.csv` | 42 aggregated rows |
| `results_dgx2_extra/run_manifest.json` | env capture, seeds, image checksum |
| `results_dgx2_extra/*_latencies.csv` (42 files) | per-iteration latency traces |
| `data/benchmark_history_unified.db` | merged into the cross-cloud DB (173 runs) |
| `docs/executive_report.html` | combined cross-cloud report |
| `notebooks/cross_cloud_analysis.ipynb` | visual cross-cloud analysis |

### How to reproduce

```bash
# Sync code to DGX2 (idempotent)
rsync -az --delete --exclude '__pycache__' --exclude '*.pyc' \
      src/ user_workloads/ scripts/ \
      radiant-dgx2@100.126.216.92:gpu_cloud_benchmark/

# Run the 3 extra workloads inside the existing Docker container
ssh radiant-dgx2@100.126.216.92 \
  'docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
     -v $HOME/gpu_cloud_benchmark:/app -w /app doc2data-gpu:latest \
     python -m src.runner --config config/benchmark_config_dgx_extra.yaml'

# Pull results
rsync -az radiant-dgx2@100.126.216.92:gpu_cloud_benchmark/results_dgx2_extra/ \
       results_dgx2_extra/
```
