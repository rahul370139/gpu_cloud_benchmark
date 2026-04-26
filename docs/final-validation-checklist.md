# Final Validation Checklist

Use this sequence to finish the project with the least risk and cost.

## 1. Local Validation

- Create a Python `3.11` or `3.12` virtual environment.
- Install dependencies from `requirements.txt`.
- Run `pytest`.
- Run a CPU smoke test:

```bash
python -m src.runner --config config/benchmark_config_local.yaml --device cpu
```

- Generate or verify the local report:

```bash
python scripts/generate_report.py --results-dir results --output results/report.html
```

## 2. Docker Validation

- Build the container image:

```bash
docker build -t gpu-benchmark .
```

- If you are building from an Apple Silicon laptop and deploying to AWS `amd64`
  workers, publish an `amd64` image to ECR:

```bash
./scripts/build_push_ecr.sh 999052221400.dkr.ecr.us-east-1.amazonaws.com/gpu-benchmark:latest
```

- Run a local container smoke test:

```bash
docker run --rm -v "$(pwd)/results:/app/results" gpu-benchmark
```

- If using AWS, push the image to a registry the cluster can pull from.

## 3. AWS Smoke Test

- Copy `infra/terraform/envs/aws-gpu/terraform.tfvars.smoke.example` to `terraform.tfvars`.
- Replace the placeholders with your real AWS values.
- Set `BENCHMARK_IMAGE` to the real pushed container image.
- Run:

```bash
cd infra
./scripts/run_pipeline.sh provision
./scripts/run_pipeline.sh bootstrap
./scripts/run_pipeline.sh deploy
BENCHMARK_IMAGE=<your-image> ./scripts/run_pipeline.sh benchmark
./scripts/run_pipeline.sh log-costs
./scripts/run_pipeline.sh teardown
```

- Inspect:
  - `infra/artifacts/<benchmark_run_id>/manifests/`
  - `infra/artifacts/<benchmark_run_id>/logs/`
  - the Prometheus/Pushgateway pods
  - S3 cost snapshot upload

## 4. Final Cloud Run

- Expand `worker_pools` to the GPU classes you want to compare.
- Re-run the benchmark flow.
- Run the fault-injection experiment:

```bash
./scripts/run_pipeline.sh fault-inject
```

- Save:
  - benchmark summary CSVs
  - latency CSVs
  - recommendation JSON
  - report HTML
  - cost snapshot
  - fault-injection observations for the final write-up

## Information You Still Need To Provide

- AWS region
- AWS availability zones to use
- Your public admin CIDR or IP
- EC2 key pair name
- Unique S3 bucket name
- The first GPU instance type you want to test
- The real benchmark container image URI
