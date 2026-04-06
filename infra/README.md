# Containerized, Reproducible Benchmarking of ML Workloads Across Cloud GPUs

This repository contains the infrastructure and orchestration scaffold for Sahil's portion of the project:

- AWS infrastructure as code with Terraform
- GPU-ready `k3s` cluster bootstrapping
- Kubernetes manifests for benchmark orchestration
- Cost logging helpers
- Fault-injection experiments
- Automated teardown workflows to avoid runaway spend

## Layout

- `terraform/`: reusable AWS Terraform modules and an environment composition
- `scripts/`: end-to-end orchestration, cluster bootstrap, fault injection, teardown, and cost logging
- `kubernetes/`: benchmark job, namespace, and monitoring manifests
- `docs/`: workflow notes for experiments and recovery testing

## Quick start

1. Copy [terraform.tfvars.example](/Users/sahilmariwala/dev/msml606/msml605/terraform/envs/aws-gpu/terraform.tfvars.example) to `terraform.tfvars`.
2. Set AWS credentials in your shell or through your preferred profile.
3. Provision the environment:

```bash
./scripts/run_pipeline.sh provision
```

4. Configure `kubectl` access and deploy cluster add-ons:

```bash
./scripts/run_pipeline.sh bootstrap
./scripts/run_pipeline.sh deploy
```

5. Run a benchmark job and upload cost metadata:

```bash
./scripts/run_pipeline.sh benchmark
./scripts/run_pipeline.sh log-costs
```

6. Exercise the fault-injection experiment:

```bash
./scripts/run_pipeline.sh fault-inject
```

7. Tear everything down when finished:

```bash
./scripts/run_pipeline.sh teardown
```

## Notes

- The Terraform stack defaults to `k3s` for a lighter-weight Kubernetes control plane.
- GPU driver installation is handled in `cloud-init` during instance bootstrap.
- Cost logging uses AWS pricing metadata supplied through Terraform variables so the project can compute performance-per-dollar consistently even if live pricing APIs are unavailable.
