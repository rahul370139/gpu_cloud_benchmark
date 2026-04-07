# Upgraded Project Proposal

**Project title:** Containerized, Reproducible Benchmarking of ML Workloads Across Cloud GPUs  
**Team:** Sahil Mariwala, Rahul Sharma  
**Version:** 2.0 — Upgraded from initial proposal  
**Date:** April 5, 2026

---

## How to Read This Document

This document is structured in two parts:

1. **Section I** — A concise summary of what changed and why.
2. **Section II** — The full upgraded proposal, ready for submission, with all enhancements integrated into the original structure.

Text marked as **[UPGRADED]** indicates a section that has been materially enhanced or added beyond the original proposal. Everything else remains from the original proposal or has been refined with implementation experience.

---

# SECTION I — Summary of Upgrades Over the Original Proposal

## What Changed and Why

The original proposal described a benchmarking framework that runs workloads, collects metrics, and produces reports. Through implementation and real GPU testing, we identified a fundamental gap: **the system tells you numbers, but it doesn't tell you what to do with them**. An ML engineer still had to manually interpret CSVs and charts to decide which GPU to rent.

The upgraded system closes that gap. It doesn't just benchmark — it **recommends**, **predicts**, and **learns over time**.

---

## The Five Upgrades

### Upgrade 1: GPU Recommendation Layer (NEW)

| | Original Proposal | Upgraded System |
|---|---|---|
| **Output** | CSVs + charts + HTML report | CSVs + charts + HTML report **+ ranked GPU recommendation with reasoning** |
| **Decision** | User interprets results manually | System says: *"Use A100 for this workload (score: 0.94, throughput/$: 2.4M)"* |
| **Scoring** | Not present | Multi-criteria weighted scorer: throughput (40%) + cost-efficiency (35%) + latency (25%), configurable |

**What was built:**
- `src/recommender/scorer.py` — normalises throughput, cost, latency to [0,1], applies weights, ranks GPUs
- `src/recommender/engine.py` — orchestrator with `recommend()`, `partial_and_recommend()`, `predict_and_recommend()` modes
- Full CLI: `python -m src.recommender recommend --results-dir results/ --max-cost 3.0`
- JSON output with composite scores, per-axis breakdown, and human-readable reasoning

**Why it matters:** This transforms the system from a measurement tool into a decision-support tool. ML teams don't want data — they want answers.

---

### Upgrade 2: Partial Benchmarking Strategy (NEW)

| | Original Proposal | Upgraded System |
|---|---|---|
| **Run duration** | Full suite: 100 iters × 3 repeats × all batch sizes | Adaptive: 5–30 iters with convergence detection |
| **Cost per benchmark** | Full cloud cost | 5-10× cheaper |
| **Confidence** | Mean ± std from 3 repeats | 95% confidence interval from convergence window |

**What was built:**
- `src/recommender/partial.py` — `PartialProfiler` class with:
  - Sliding window convergence detection (window=8, CV threshold=5%)
  - Time budget enforcement (default 300 seconds)
  - Early stopping when throughput stabilises
  - 95% confidence intervals (z=1.96 normal approximation)
- Configurable via `config/recommendation_config.yaml`

**Why it matters:** The original proposal acknowledged "cloud budget" as a risk and proposed "short experiments" as mitigation. This upgrade formalises that into a rigorous statistical method. Instead of hoping short runs are representative, the system mathematically proves convergence before stopping.

---

### Upgrade 3: Historical Logging & Reuse (NEW)

| | Original Proposal | Upgraded System |
|---|---|---|
| **Data persistence** | CSV files + S3 bucket (one-shot) | SQLite database (cumulative, queryable) |
| **Cross-run analysis** | Manual comparison of CSVs | Automatic: every run enriches the knowledge base |
| **Future value** | None — old results are stale files | Old results power predictions for new workloads |

**What was built:**
- `src/recommender/history.py` — `HistoryStore` class backed by SQLite with:
  - Two tables: `benchmark_runs` (all metrics per run) and `recommendations` (query + result audit trail)
  - Indexed by workload, GPU type, and mode for fast querying
  - Tracks both full and partial runs with confidence bounds
  - `import` command to ingest legacy CSV results
- Auto-logging hook in `src/runner.py` — every benchmark run is automatically written to history
- CLI: `python -m src.recommender history` and `python -m src.recommender import`

**Why it matters:** The original proposal treated each benchmark as a standalone event. The upgrade treats benchmarking as a cumulative investment — every run makes future runs cheaper (because predictions become possible) and future decisions better (because more data points exist).

---

### Upgrade 4: Workload-Similarity Predictor (NEW)

| | Original Proposal | Upgraded System |
|---|---|---|
| **New workloads** | Must benchmark from scratch on every GPU | Can predict performance without running anything |
| **Method** | N/A | K-nearest-neighbours on workload features |
| **Features used** | N/A | param_count, batch_size, memory_footprint, is_training |
| **Cost** | Full benchmark cost per new model | Zero — runs on CPU, no GPU needed |

**What was built:**
- `src/recommender/predictor.py` — `WorkloadPredictor` class:
  - Feature extraction: `log(param_count)`, `log(batch_size)`, `log(memory_footprint)`, `is_training` — each weighted
  - Euclidean distance in normalised feature space
  - K=3 nearest neighbours per GPU type
  - Inverse-distance weighted interpolation for throughput, latency, and memory
  - Confidence score: `1 - (mean_distance / max_distance)`
- CLI: `python -m src.recommender predict --param-count 25600000 --batch-size 32 --mode inference --family vision`

**Why it matters:** This is the key differentiator that moves the project from "benchmarking tool" to "intelligent infrastructure advisor." When a team is evaluating a new model (say, a 300M-param LLM), they don't need to spend $50 on cloud GPUs to get a recommendation — the system extrapolates from its history of ResNet-50 and BERT-base runs. As the history grows with more workloads and GPU types, prediction accuracy improves automatically.

---

### Upgrade 5: Cost-Aware Constraint Filtering (NEW)

| | Original Proposal | Upgraded System |
|---|---|---|
| **Budget handling** | Compute throughput-per-dollar, user interprets | User specifies: *"under $2/hr"* — system filters automatically |
| **Latency SLAs** | Report P95 latency, user checks manually | User specifies: *"P95 < 100ms"* — infeasible GPUs excluded |
| **Output** | All GPUs shown equally | Feasible GPUs ranked + excluded GPUs listed with rejection reasons |

**What was built:**
- `src/recommender/constraints.py` — `UserConstraints` dataclass + `apply_constraints()`:
  - `max_cost_per_hour` — hard budget cap (e.g., `$2.00/hr`)
  - `max_latency_p95_ms` — SLA requirement (e.g., `100ms`)
  - `min_throughput` — minimum acceptable throughput
  - `max_gpu_memory_gb` — memory cap
  - Per-GPU rejection reasons (e.g., *"cost $4.10/hr exceeds $2.00/hr budget"*)
  - Warning when all GPUs are excluded (constraints too restrictive)
- Integrated into all three engine modes (recommend, partial, predict)
- CLI flags: `--max-cost 2.0 --max-latency 100 --min-throughput 500`

**Why it matters:** The original proposal mentioned "cost-efficiency as a first-class metric" but only in the reporting layer. The upgrade makes cost a first-class *constraint* — the system won't recommend a GPU you can't afford, no matter how fast it is.

---

## What Remained from the Original Proposal (Validated and Delivered)

These items were proposed and have been fully implemented and tested:

| Original Proposal Item | Implementation Status |
|---|---|
| Docker image with model code, measurement scripts, pinned deps | **DONE** — `Dockerfile` + `requirements.txt` + `entrypoint.sh` |
| Benchmark runner (throughput, latency, GPU metrics) | **DONE** — `src/runner.py`, 48 GPU runs completed |
| ResNet-50 + BERT-base workloads | **DONE** — `src/workloads/vision.py`, `src/workloads/nlp.py` |
| Reproducibility (seeds, checksums, env snapshot) | **DONE** — `src/reproducibility/`, CV < 0.5% confirmed |
| Cost calculation (throughput-per-dollar) | **DONE** — `src/cost/calculator.py` |
| Prometheus Pushgateway integration | **DONE** — `src/metrics/prometheus_exporter.py` |
| K8s Job manifests | **DONE** — `k8s/benchmark-job.yaml` |
| Grafana dashboard | **DONE** — `k8s/prometheus/grafana-dashboard.json` |
| Python analysis scripts (matplotlib/pandas) | **DONE** — 7 chart types in `src/analysis/visualizer.py` |
| HTML report | **DONE** — self-contained with base64-embedded charts |
| Preflight check (driver validation) | **DONE** — `scripts/preflight_check.py` |
| Extensible multi-workload design | **DONE** — lazy workload registry, add any model in 3 lines |
| Unit tests | **DONE** — 68 tests (31 original + 37 recommender) |

---

## Quantitative Impact of Upgrades

| Metric | Original Scope | After Upgrades |
|--------|---------------|----------------|
| Source files | ~25 | 53 |
| Lines of code (Python) | ~1,800 | ~3,600+ |
| Unit tests | 31 | 68 (37 new for recommender) |
| Configuration files | 3 | 4 (+ recommendation_config.yaml) |
| CLI entry points | 1 (`python -m src.runner`) | 2 (`src.runner` + `src.recommender` with 5 subcommands) |
| Output artifacts | CSVs + charts + HTML | CSVs + charts + HTML + `recommendation.json` + SQLite history |
| System modes | 1 (full benchmark) | 4 (full benchmark, recommend, partial, predict) |
| GPU runs required for advice | Always (full suite) | Sometimes none (predict mode) |
| Decision support | None (user interprets) | Automated ranked recommendation with reasoning |

---

# SECTION II — Full Upgraded Proposal

*(This section is the complete, submission-ready proposal with all upgrades integrated.)*

---

## Project Title

**Containerized, Reproducible Benchmarking of ML Workloads Across Cloud GPUs — with Intelligent GPU Recommendation Engine**

## Names

- Sahil Mariwala
- Rahul Sharma

## Team Contributions

**Sahil Mariwala:** Infrastructure as code and orchestration — design and implement Terraform modules, cloud provisioning workflows, Kubernetes cluster setup, and fault-injection experiments. Sahil will also be responsible for cost logging and automating teardown to avoid runaway bills.

**Rahul Sharma:** Benchmarking, containerization, and intelligent recommendation — create the Docker image(s) that contain the models and measurement code, implement the benchmark runners, collect and pre-process metrics, build the analysis scripts and visualizations, and **[UPGRADED]** develop the GPU recommendation engine (scoring, constraint filtering, partial benchmarking, historical logging, and workload-similarity prediction). Rahul will also coordinate reproducibility checks (seeds, pinned deps, checksums).

## Problem Statement — What We Are Solving

Picking the right GPU in the cloud matters more than people think. Different GPUs behave very differently depending on model architecture, batch size, and framework; cloud teams often test manually and informally, which leads to inconsistent, hard-to-reproduce results and wasted spend.

We will build an automated, containerized benchmarking framework that runs identical ML workloads across multiple cloud GPU instance types, captures standardized metrics (throughput, latency, utilization, cost), and produces reproducible, comparable reports to help practitioners choose the most cost-effective and reliable hardware for a given workload.

**[UPGRADED]** Beyond benchmarking, the system will function as an **intelligent GPU advisor**. After benchmarking, it automatically recommends the optimal GPU based on weighted scoring of performance, cost, and latency — with support for user-defined budget and SLA constraints. For new workloads that haven't been benchmarked, the system predicts GPU performance from historical data using workload-similarity analysis, eliminating the need for expensive trial runs. A partial benchmarking mode reduces cloud cost by 5-10× through convergence detection and early stopping while still producing statistically reliable estimates with confidence intervals.

## High-Level Approach and Novelty

At a high level, we package a workload into a Docker image, spin up GPU instances declaratively, run the same experiments on each GPU type, collect metrics centrally, and tear down infrastructure when finished. The workflow is automated end-to-end, so a user can request a benchmark (model + dataset + config) and get back standardized performance-per-dollar and reliability results.

What makes this project more than "yet another benchmark":

1. **Reproducibility-first design:** The whole stack (AMI/driver, Docker image, Terraform, Kubernetes manifests, random seeds) is codified so results are repeatable by anyone with access. That moves an ad-hoc testing habit toward a proper MLOps artifact.

2. **MLOps integration:** Benchmarking is treated as an operational pipeline component (CI trigger, automated runs, artifact storage) rather than a one-off. That demonstrates how benchmarking fits into lifecycle practices.

3. **Cost-efficiency as a first-class metric:** We compute throughput-per-dollar and present trade-offs visually and numerically. This is the metric cloud teams actually care about when selecting instances.

4. **Failure and availability tests:** We will inject simple faults (kill a node, restart pods) to measure how orchestrated deployments recover and how that impacts completed work and cost.

5. **Extensible, multi-workload approach:** While we start with two representative workloads (ResNet-50 for vision, BERT-base for NLP), the framework is designed so that more models or clouds can be added later without changing the core automation.

6. **[UPGRADED] Automated GPU recommendation:** After benchmarking, the system produces a concrete recommendation ("Use A100 for this workload") with a composite score, not just raw numbers for the user to interpret. The scoring function weighs throughput (40%), cost-efficiency (35%), and latency (25%), and is fully configurable.

7. **[UPGRADED] Partial benchmarking with convergence detection:** Instead of running full 100-iteration × 3-repeat suites, the system can run short controlled bursts (5–30 iterations) and stop as soon as throughput stabilises (coefficient of variation < 5% in a sliding window). This produces 95% confidence intervals at 5-10× lower cloud cost, adding both novelty and practical value.

8. **[UPGRADED] Historical learning and prediction:** Every benchmark result is stored in a persistent database. When a user brings a new model, the system estimates GPU performance via K-nearest-neighbour similarity on workload features (parameter count, batch size, memory footprint, training vs. inference) — at zero cloud cost.

9. **[UPGRADED] Cost-aware constraint filtering:** Users can specify hard constraints (e.g., "under $2/hr," "P95 latency < 100ms") and the system recommends only from feasible GPUs, listing excluded options with specific rejection reasons.

## Implementation Plan and Tools

### Environment and Packaging

- **Docker** — build a reproducible runtime image using `nvcr.io/nvidia/pytorch:24.08-py3` as the base, including model code, measurement scripts, and utilities. All library versions are pinned in `requirements.txt`, and a **5-stage entrypoint script** orchestrates: (1) preflight check, (2) environment capture, (3) benchmark execution, (4) report generation, **[UPGRADED]** (5) GPU recommendation.

### Infrastructure Provisioning and Orchestration

- **Terraform** — provision GPU-enabled VMs on AWS (EC2 with T4/V100/A100/H100 families) with parameterized modules for easy instance swapping.
- **Kubernetes** — deploy benchmarks as K8s Jobs using NVIDIA's device plugin. Manifests, PVCs, and ConfigMaps are pre-built and validated.

### Benchmarking and Metrics

- **Benchmark runner** (`src/runner.py`) — config-driven loop over all (workload × mode × batch_size × repeat) combinations. Uses CUDA events for sub-millisecond GPU timing, pynvml for hardware telemetry, and deterministic seeding per run. Outputs per-iteration latency CSVs, a summary CSV, and a cryptographically signed run manifest.
- **[UPGRADED] Partial benchmark mode** (`src/recommender/partial.py`) — runs a reduced iteration count with real-time convergence monitoring. Stops early when throughput stabilises, produces confidence intervals, and logs results to the history database.
- **Cost calculation** — captures GPU hourly rates from a YAML rate table and computes throughput-per-dollar, cost-per-1K-samples, and cost-efficiency ranking.

### **[UPGRADED] Recommendation Engine**

A new subsystem (`src/recommender/`) with six modules:

| Module | Purpose |
|--------|---------|
| `engine.py` | Orchestrator — three modes: `recommend`, `partial`, `predict` |
| `scorer.py` | Multi-criteria weighted scorer — normalise, weight, rank |
| `constraints.py` | Hard constraint filter — budget, latency SLA, throughput floor |
| `partial.py` | Convergence-checked short benchmarks with confidence intervals |
| `history.py` | SQLite database for persistent storage of all benchmark results |
| `predictor.py` | KNN workload-similarity predictor using feature vectors |

**CLI access:**
```
python -m src.recommender recommend --results-dir results/ --max-cost 3.0
python -m src.recommender partial --benchmark-config config/... --device cuda
python -m src.recommender predict --param-count 25600000 --batch-size 32
python -m src.recommender import --results-dir old_results/
python -m src.recommender history
```

### **[UPGRADED] Historical Logging**

All benchmarking results (full and partial) are automatically stored in a SQLite database (`data/benchmark_history.db`). The schema tracks:
- Throughput, latency percentiles, GPU utilization, memory, cost
- Whether the run was full or partial (with confidence bounds)
- All recommendation queries and their outcomes (audit trail)

This enables the prediction mode and cross-run trend analysis.

### Monitoring and Visualization

- **Prometheus + Grafana** — live resource monitoring via Pushgateway integration and a pre-built 5-panel Grafana dashboard.
- **Python analysis scripts** — 7 publication-quality chart types generated by `src/analysis/visualizer.py`, embedded in a self-contained HTML report.

### Automation and CI/CD

- **GitHub Actions** — trigger end-to-end benchmark runs from a commit or manual dispatch.
- **Automatic teardown** — scripts ensure GPU resources are destroyed after completion or on failure.

## Evaluation Methodology

- **Representative models:** ResNet-50 (25.6M params, vision) and BERT-base (109.5M params, NLP), covering compute-bound and memory-bound patterns.
- **Repeats:** 3 repeats per config, reporting mean, std, and coefficient of variation.
- **Metrics:** Throughput (primary), latency percentiles (P50/P95/P99), GPU utilization/memory, throughput-per-dollar, job completion rate under faults.
- **[UPGRADED] Recommendation accuracy:** Compare the system's recommendation against the actual best GPU (by throughput-per-dollar) to validate scoring weights and predictor accuracy.
- **[UPGRADED] Partial benchmark reliability:** Compare partial-run estimates (with CI) against full-run means to validate that convergence detection produces reliable extrapolations.
- **Scaling:** Measure throughput scaling across 1, 2, and 4 GPU nodes.

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Cloud budget | Limit instance counts; use partial benchmarking (5-10× cheaper); mandatory teardown; local testing first. **[UPGRADED]** Partial mode is now a formal feature, not just a workaround. |
| Driver/config mismatch | Preflight check fails fast so faulty nodes don't contaminate results. |
| Noisy neighbours in cloud | Repeats + CV analysis surface noisy conditions; re-run on different AZs if needed. |
| **[UPGRADED]** Cold-start for predictor | Predictor requires ≥5 historical entries to produce reliable estimates. We bootstrap history by importing all benchmark results automatically. Confidence scores indicate prediction reliability. |
| **[UPGRADED]** Overfitting scoring weights | Default weights (40/35/25) are configurable. Users can override for their specific priority (e.g., 90% throughput weight for latency-insensitive batch jobs). |

## Deliverables

1. A Git repository with Terraform modules, Kubernetes manifests, Dockerfile(s), benchmark scripts, **[UPGRADED]** recommendation engine, and CI workflow examples.

2. A short web dashboard or PDF report with tables/figures covering performance, cost-efficiency, variance, and recovery under fault injection.

3. **[UPGRADED]** An interactive recommendation CLI that accepts a workload description and constraints, and produces a ranked GPU recommendation with reasoning — either from existing results, short partial runs, or zero-cost historical prediction.

4. **[UPGRADED]** A persistent benchmark history database that grows with each run, enabling progressively better predictions for new workloads over time.
