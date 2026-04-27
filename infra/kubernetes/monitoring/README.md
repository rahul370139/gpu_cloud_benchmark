# Monitoring stack

Live metrics for the benchmark pipeline. Everything in this directory is
applied as part of `./scripts/run_pipeline.sh deploy` when you bring AWS infra
back up (optional for project closure — see `PROJECT_PROGRESS.md` §10.4).

For a deep-dive into k3s networking and how the access patterns actually work,
see [`infra/docs/k3s-networking-explained.md`](../../docs/k3s-networking-explained.md).

## Components

| File                       | Resources created                                              |
|----------------------------|----------------------------------------------------------------|
| `prometheus-configmap.yaml`| `ConfigMap/prometheus-config` (scrape config)                  |
| `prometheus.yaml`          | `Deployment/prometheus` + `Service/prometheus` (NodePort 30090)|
| `grafana.yaml`             | `Deployment/grafana` + `Service/grafana` (NodePort 30030) + datasource & dashboard provider ConfigMaps |
| `grafana_dashboard.json`   | The "GPU Benchmark Live" dashboard (8 panels, 3 template vars) — mounted into Grafana via `ConfigMap/grafana-dashboard-json` |

The Pushgateway Deployment lives at `infra/k8s/prometheus/pushgateway-deploy.yaml`
and is also applied by `deploy_benchmark_stack.sh`.

## Zero-touch dashboard provisioning

You **do not** need to import the JSON manually. The deploy step does this:

```bash
kubectl create configmap grafana-dashboard-json \
  -n ml-benchmark \
  --from-file=gpu-benchmark.json=infra/kubernetes/monitoring/grafana_dashboard.json
```

That ConfigMap is mounted into the Grafana pod at
`/var/lib/grafana/dashboards/gpu-benchmark.json`. Grafana's
`grafana-dashboards-provider` ConfigMap tells it to scan that path on
startup, so the dashboard appears automatically.

The Prometheus datasource is added the same way (via the
`grafana-datasources` ConfigMap), pointing at
`http://prometheus.ml-benchmark.svc.cluster.local:9090`.

## Reaching Grafana

After `./scripts/run_pipeline.sh deploy` finishes:

```bash
cd infra
# Use the wrapper so kubeconfig path + SSH tunnel to the API are correct:
./scripts/kubectl_aws.sh port-forward -n ml-benchmark svc/grafana 3000:3000
open http://localhost:3000        # admin / admin
```

Or, if your IP is in `admin_cidrs`:

```bash
PUB_IP=$(jq -r '.controller_public_ip' \
  infra/terraform/envs/aws-gpu/inventory.json)
open "http://${PUB_IP}:30030"
```

Same patterns work for Prometheus (`:9090` / NodePort `:30090`) and
Pushgateway (`:9091`).

## Metric labels

The benchmark workloads push metrics to Pushgateway tagged with:

- `gpu_type` — `NVIDIA_GB10`, `A10G`, `T4`, `CPU`
- `workload` — `resnet50`, `bert_base`, `clip_image_embedding`, `llm_text_generation`, `example_mlp`
- `batch_size`

Series scraped:

- `benchmark_throughput` (samples/s)
- `benchmark_latency_p50_ms`, `benchmark_latency_p95_ms`, `benchmark_latency_p99_ms`
- `benchmark_gpu_utilization_pct`
- `benchmark_gpu_memory_used_mb`

The dashboard exposes three template variables (`gpu_type`, `workload`,
`batch_size`) so you can drill into a single workload-batch combo or
compare A10G vs T4 vs GB10 head-to-head.

## Updating the dashboard

Edit `grafana_dashboard.json` (or click "Save JSON" in the Grafana UI and
copy back), then re-run:

```bash
kubectl create configmap grafana-dashboard-json \
  -n ml-benchmark \
  --from-file=gpu-benchmark.json=infra/kubernetes/monitoring/grafana_dashboard.json \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment/grafana -n ml-benchmark
```

Grafana's file-provider polls every 30 s, so a `rollout restart` is only
needed if you edit the datasources or providers ConfigMaps.
