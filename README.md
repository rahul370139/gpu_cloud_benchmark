# GPU Cloud Benchmark

Containerized, reproducible benchmarking **and intelligent GPU recommendation** for ML workloads across heterogeneous accelerators.

## What this project is

A turn-key system that does four things end-to-end:

1. **Benchmarks** any ML workload (built-in or custom) on any combination of GPUs — DGX Spark, AWS, or local CPU — inside a deterministic Docker container.
2. **Captures** a complete, reproducible record of every run (throughput, P50/P95/P99 latency, GPU utilisation, memory, power, cost) into an SQLite history.
3. **Recommends** the best GPU for a workload via a multi-criteria scorer (throughput · cost-efficiency · latency) with cost-aware budget / latency constraints.
4. **Predicts** the best GPU for a *new* workload via a KNN model over the history — no benchmark required for the new workload.

Everything is one CLI:

```bash
python -m src.runner          --config config/benchmark_config.yaml      # benchmark
python -m src.recommender recommend --results-dir results/                # rank GPUs
python -m src.recommender partial   --workload bert_base --device cuda    # short converged run
python -m src.recommender predict   --workload my_new_model --param-count 50e6  # no-run prediction
```

## Status (April 27, 2026)

| Capability | Status | Evidence |
|---|---|---|
| 5-workload benchmark suite | DONE | `config/benchmark_config.yaml` |
| 4-platform coverage (GB10 / A10G / T4 / CPU) | DONE | `data/benchmark_history_unified.db` — **173 runs** |
| Multi-criteria recommendation engine | DONE | `results_unified/recommendation_all.json` |
| KNN no-run predictor — quantitatively validated | DONE | **80% winner-match** (`RECOMMENDER_EVALUATION.md`) |
| AWS multi-GPU pipeline (Terraform + k3s + S3) | DONE | `report.html`, `infra/` |
| GitHub Actions CI (python · terraform · docker) | DONE | `.github/workflows/ci.yaml` |
| Optional CI extras (release / aws-smoke) | DONE | `.github/workflows/{release,aws-smoke}.yaml` |
| Grafana UI in-cluster + scripted fault-inject replay | "optional" | manifests + dashboard JSON exist under `infra/kubernetes/monitoring/` — not required to close this phase |
| Cross-cloud executive report | DONE | `docs/executive_report.html`, `notebooks/cross_cloud_analysis.ipynb` |
| AWS benchmark narrative (this wrap-up) | DONE | Primary evidence: **`report.html`** (full A10G/T4 campaign; header summarises run counts including any failed runs) |

## Workloads

The suite currently runs **5 workloads** that span model families and parameter scales:

| Workload | Model | Family | Params | Metric |
|---|---|---|---:|---|
| `resnet50` | ResNet-50 | vision | 25.6 M | images/sec |
| `bert_base` | BERT-base uncased | nlp | 109 M | tokens/sec |
| `example_mlp` | 4-layer MLP | tabular | 34 K | samples/sec |
| `clip_image_embedding` | CLIP ViT-B/32 image encoder | vision | 87.8 M | images/sec |
| `llm_text_generation` | small GPT-style decoder | generative | 23.6 M | tokens/sec |

Custom workloads can be added under `user_workloads/` by subclassing `src/workloads/base.py:BaseWorkload` (see `user_workloads/template.py` and `user_workloads/example_mlp.py`).

## Platforms covered

| Platform | Instance / SKU | Cost rate (used by recommender) | Workloads run |
|---|---|---|---|
| **NVIDIA GB10** (DGX Spark, on-prem ARM) | dev kit | **$0.30/h** (fully-loaded TCO — see `config/gpu_cost_rates.yaml`) | 5 / 5 |
| **NVIDIA A10G** (AWS) | `g5.xlarge` | $1.006/h | 5 / 5 |
| **NVIDIA T4** (AWS) | `g4dn.xlarge` | $0.526/h | 5 / 5 |
| Local CPU | macOS / Apple Silicon | $0/h | 5 / 5 (smoke) |

## Quick start

### Local (CPU smoke)

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m src.runner --config config/benchmark_config_cpu_5wl.yaml --device cpu
```

### Docker (GPU)

```bash
docker build -t gpu-benchmark .
docker run --gpus all -v $(pwd)/results:/app/results gpu-benchmark
```

When developing on Apple Silicon and pushing to amd64 EC2 nodes, build multi-arch:

```bash
./scripts/build_push_ecr.sh 999052221400.dkr.ecr.us-east-1.amazonaws.com/gpu-benchmark:latest
```

### AWS multi-GPU (Terraform + k3s + S3)

```bash
cd infra
export SSH_KEY_PATH="$HOME/.ssh/rahul-test.pem"   # path to the private key file (if not using ssh-agent)

./scripts/run_pipeline.sh provision    # terraform apply
./scripts/run_pipeline.sh bootstrap    # fetch kubeconfig + open SSH tunnel to :6443
./scripts/run_pipeline.sh deploy       # namespace / config / Prometheus + Pushgateway (+ optional Grafana when you revisit infra)
./scripts/run_pipeline.sh benchmark    # one Job per GPU class, in parallel
./scripts/run_pipeline.sh log-costs    # ec2 metadata + cost snapshot
# ./scripts/run_pipeline.sh fault-inject   # optional — defer unless you reprovision AWS
./scripts/run_pipeline.sh teardown     # terraform destroy
```

Worker pools are declared in `infra/terraform/envs/aws-gpu/terraform.tfvars`.

**`terraform.tfvars` gotchas**

- `key_name` must be the **Key pair name** in the EC2 console (e.g. `rahul-test`), not `rahul-test.pem`.
- `admin_cidrs` should be your **current** public IP `/32` (it changes when you move networks).
- `artifact_bucket_name` must be **globally unique** across all of S3.

**`kubectl`: "connection refused" to `localhost:8080`**

That means kubectl is not reading a valid kubeconfig (wrong path or file missing). From `infra/`, use `$(pwd)/kubeconfig.yaml`, not `infra/kubeconfig.yaml`. Easiest: always use the wrapper (creates the SSH tunnel and sets `KUBECONFIG` for you):

```bash
cd infra
./scripts/kubectl_aws.sh get nodes
# Grafana port-forward only when you bring the cluster back — optional for this wrap-up.
# ./scripts/kubectl_aws.sh port-forward -n ml-benchmark svc/grafana 3000:3000
```

Or after `bootstrap`, from `infra/`: `export KUBECONFIG="$PWD/kubeconfig.yaml"` and ensure the tunnel is up (re-run `bootstrap` or any pipeline step that sources `common.sh` and calls `ensure_k3s_tunnel`).

### Recommendation modes

```bash
# Mode 1 — recommend over an existing results directory
python -m src.recommender recommend --results-dir results_unified

# Mode 2 — short, converged "partial" benchmark (5–10× cheaper than full)
python -m src.recommender partial --workload bert_base --device cuda \
       --max-iters 200 --convergence-threshold 0.05

# Mode 3 — KNN no-run prediction (no GPU required)
python -m src.recommender predict \
       --workload my_new_workload \
       --param-count 50000000 \
       --batch-size 16 \
       --mode inference \
       --family vision
```

Constraints accepted by all modes:

```bash
--max-cost-per-hour 2.0 --max-latency-ms 100 --min-throughput 100
```

## Reports

```bash
# Per-platform interactive report
python scripts/generate_report.py --results-dir results_unified --output report.html

# Cross-cloud executive report (single file, all charts inline)
python scripts/generate_executive_report.py
# → docs/executive_report.html
```

The executive report is the recommended hand-off artefact: it summarises 173 runs across 5 workloads × 4 platforms, plus the recommender output and the KNN validation, on one page.

For interactive exploration:

```bash
jupyter lab notebooks/cross_cloud_analysis.ipynb
```

## Project structure

```
gpu_cloud_benchmark/
├── config/                  # Benchmark + cost configuration (YAML)
├── src/
│   ├── workloads/           # Built-in workloads + lazy custom-workload registry
│   ├── metrics/             # GPU metrics, timing, Prometheus pushgateway
│   ├── cost/                # cost-per-throughput calculator
│   ├── analysis/            # report generation, plots
│   ├── reproducibility/     # seeds, checksums, env capture
│   ├── recommender/         # multi-criteria scorer, partial profiler, KNN, history
│   ├── artifacts/           # S3 uploader (per-pod, per-GPU-class)
│   └── runner.py            # benchmark orchestrator
├── user_workloads/          # User-extensible workloads (CLIP, LLM, MLP, template)
├── scripts/                 # entrypoint, preflight, eval helpers, report generators
├── infra/
│   ├── terraform/           # VPC + security + multi-GPU compute
│   ├── kubernetes/          # namespace, ConfigMap, per-GPU Job template, monitoring
│   └── scripts/             # run_pipeline.sh + provision / bootstrap / deploy / fault-inject
├── notebooks/               # cross_cloud_analysis.ipynb (visual analysis)
├── docs/                    # executive_report.html, validation checklist
├── tests/                   # 72 unit tests across 9 files (CPU-compatible)
├── data/                    # SQLite history (benchmark_history_unified.db)
├── results_unified/         # consolidated per-platform summary CSVs + recommendations
├── results_eval/            # KNN holdout evaluation outputs
├── ARCHITECTURE.md          # system design
├── PROJECT_PROGRESS.md      # progress report
├── RECOMMENDER_EVALUATION.md # quantitative validation report
└── README.md                # you are here
```

## Custom workloads

Add a class under `user_workloads/` that subclasses `BaseWorkload`:

```yaml
# config/my_config.yaml
custom_workloads:
  my_model: "user_workloads.my_model:MyModelWorkload"

workloads:
  - my_model
```

Or pass a class directly without editing YAML:

```bash
python -m src.runner \
  --config config/benchmark_config_local.yaml \
  --device cpu \
  --workload-target user_workloads.example_mlp:ExampleMLPWorkload \
  --workload-name my_custom
```

The framework reuses the existing timing, GPU metrics, cost analysis, reporting, and recommendation code — your workload class only declares model construction and batch generation.

## Reproducibility guarantees

Every run captures and persists:

- Random seeds (torch, numpy, python, CUDA)
- SHA-256 checksums of the Docker image, model weights, and data
- Full environment snapshot (driver versions, CUDA, cuDNN, `pip freeze`, OS info)
- All written to `results/run_manifest.json` and the SQLite history

Reproducibility was measured on the unified dataset: **CV < 5% on GB10 / A10G / T4** for nearly every (workload × batch_size) scenario.

## CI

GitHub Actions runs three jobs on every PR and push to main:

- **python** — `pytest` (CPU-only torch wheel)
- **terraform** — `fmt -check` + `init -backend=false` + `validate`
- **docker** — `buildx` build (no push)

Two additional workflows are present but only fire on demand:

- **release** — tag-triggered, pushes the image to GHCR
- **aws-smoke** — `workflow_dispatch`, optional 1×T4 smoke run (requires AWS OIDC role)

We deliberately do **not** run GPU benchmarks in CI — see `PROJECT_PROGRESS.md` §9 for the rationale.

## Documentation

| Document | Purpose |
|---|---|
| `README.md` | Quick start (this file) |
| `ARCHITECTURE.md` | System design, mermaid diagrams, component reference |
| `PROJECT_PROGRESS.md` | Status, milestones, lessons learned, pending work |
| `RECOMMENDER_EVALUATION.md` | Quantitative validation of all three recommender modes |
| `UPGRADED_PROPOSAL.md` | Original proposal vs delivered system |
| `AWS_BENCHMARK_LOG.md` | AWS A10G + T4 cloud run record |
| `DGX2_BENCHMARK_LOG.md` | NVIDIA GB10 on-prem run record |
| `docs/executive_report.html` | One-page consolidated report (open in browser) |
| `docs/final-validation-checklist.md` | Local → Docker → AWS sequence |
| `notebooks/cross_cloud_analysis.ipynb` | Interactive cross-cloud analysis |

## Team

- **Rahul Sharma** — Benchmark framework, contarization, recommendation engine, history DB, KNN predictor, custom-workload extension, cross-cloud analysis, executive report
- **Sahil Mariwala** — Terraform IaC (VPC + compute), k3s pipeline, S3 artifacts, Prometheus / Pushgateway, fault-injection script, AWS cost logging
