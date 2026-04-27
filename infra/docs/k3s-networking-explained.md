# How k3s/Kubernetes works in this project (and how to reach Grafana)

This is a practical walk-through of the networking model we use, written
specifically for the AWS GPU benchmark stack. If you've never set up k8s by
hand, this is the doc that should make `kubectl port-forward` and "why can't I
hit Grafana on port 3000?" make sense.

### Troubleshooting: `connection refused` to `127.0.0.1:8080`

`kubectl` talks to `localhost:8080` only when **no valid kubeconfig** is loaded
(that is the built-in default). Typical causes:

1. **Wrong `KUBECONFIG` path for your current directory.** If your shell is
   already in `infra/`, use `export KUBECONFIG="$PWD/kubeconfig.yaml"`, **not**
   `infra/kubeconfig.yaml` (that would look for `infra/infra/kubeconfig.yaml`).
2. **File never created.** `kubeconfig.yaml` is written by
   `./scripts/run_pipeline.sh bootstrap` after `provision`, and is gitignored.
3. **SSH tunnel to the API server is down.** Even with a valid file, the
   server URL is `https://127.0.0.1:6443` tunneled to the controller. Re-open the
   tunnel by re-running `bootstrap` or use the wrapper below.

**Fix (recommended):** from `infra/`, use the wrapper so path + tunnel are always correct:

```bash
./scripts/kubectl_aws.sh get nodes
./scripts/kubectl_aws.sh port-forward -n ml-benchmark svc/grafana 3000:3000
```

---

## 1. k3s vs Kubernetes (k8s)

**Kubernetes (k8s)** is a generic orchestrator — it schedules containers
("pods") onto a fleet of machines ("nodes"). The control plane (API server,
scheduler, controller-manager, etcd) usually runs on a dedicated set of
machines.

**k3s** is a single Go binary that bundles the same control plane + kubelet +
container runtime + flannel CNI + traefik ingress into ~70 MB. We pick it
because:

* `k3s` boots in ~30 s on a `t3.large`; vanilla `kubeadm` takes 3–5 min.
* No external etcd — embedded SQLite is fine for ≤5 nodes.
* `kubectl` is identical, so the rest of the project (manifests, Helm charts,
  port-forwarding) is portable to EKS without changes.

In our cluster topology:

```
┌──────────────────────────────────────────────────────────┐
│  Controller node (t3.large, EC2 public IP)               │
│  ──────────────────────────────────────────              │
│  • k3s server  (API on :6443, internal-only)             │
│  • k3s agent   (so the node itself is also a worker for  │
│                 small pods like Prometheus / Grafana)    │
│  • Pod CIDR:   10.42.0.0/24                              │
│  • Svc CIDR:   10.43.0.0/16                              │
│  • Public:     SG opens :22 + :30000-32767 from          │
│                admin_cidrs only                          │
└─────────────┬────────────────────────────────────────────┘
              │  kubectl from your laptop reaches :6443
              │  via an SSH tunnel — never exposed publicly.
              ▼
┌──────────────────────────────────────────────────────────┐
│  Worker pool (gpu_class=T4, 1× g4dn.xlarge)              │
│  ──────────────────────────────────────────              │
│  • k3s agent (joins via cluster_token)                   │
│  • NVIDIA runtimeClass + DCGM exporter                   │
│  • Pod CIDR:  10.42.1.0/24                               │
│  • No public ingress required                            │
└──────────────────────────────────────────────────────────┘
```

The controller's k3s API listens on `127.0.0.1:6443`. Your laptop reaches it
via `ssh -L 6443:127.0.0.1:6443 ubuntu@<controller_public_ip>`, which is
exactly what `infra/scripts/common.sh::ensure_k3s_tunnel` does.

---

## 2. The four objects you need to know

For our stack you only ever touch four kinds of object:

| Object       | What it is                                                      | Example in this repo                              |
|--------------|------------------------------------------------------------------|---------------------------------------------------|
| **Pod**      | One or more containers scheduled on one node, sharing a network. | `benchmark-run-t4-xyz` running ResNet-50.         |
| **Deployment** | A controller that keeps N replicas of a Pod alive.             | `prometheus`, `grafana`.                          |
| **Job**      | Run a Pod to completion (one-shot).                              | `benchmark-run-t4` after a benchmark dispatch.    |
| **Service**  | A stable cluster-internal IP + DNS name in front of pods.        | `prometheus`, `grafana`, `pushgateway`.           |
| **ConfigMap**| A blob of config data mounted as files / env vars.               | `benchmark-config`, `grafana-dashboard-json`.     |

Two networking facts that confuse most people first time:

1. **Pods are mortal — Pod IPs change every restart.** Never hardcode them.
2. **Services give you a stable IP + DNS name.** k3s' CoreDNS auto-creates the
   record `<service>.<namespace>.svc.cluster.local`. Anything inside the
   cluster can reach Grafana at
   `http://grafana.ml-benchmark.svc.cluster.local:3000` regardless of where
   the pod is currently running.

---

## 3. Service types — the "how do I reach this from outside?" question

A `Service` has a `.spec.type` that decides where the stable IP lives:

* **ClusterIP** (default): internal only. Other pods can hit it. Your laptop
  cannot. Use this for things only pods need (e.g. Pushgateway).
* **NodePort**: ClusterIP **plus** every node opens a port in the
  `30000-32767` range and forwards traffic to the Service. Reachable from
  anywhere that can hit a node's IP. We use this for Prometheus (`:30090`)
  and Grafana (`:30030`) so the smoke test works without an LB.
* **LoadBalancer**: tells the cloud provider to spin up an actual ALB/NLB
  and point it at the NodePort. We deliberately don't use this — it adds
  ~$0.025/h per LB and our smoke runs are too short to amortise that.
* **Ingress**: an HTTP(S) reverse proxy in front of multiple Services on a
  single LB / hostname. Overkill for this project.

Our deploy applies these:

```
namespace/ml-benchmark
├─ Deployment/prometheus       Service/prometheus   NodePort  30090
├─ Deployment/pushgateway      Service/pushgateway  ClusterIP   —
├─ Deployment/grafana          Service/grafana      NodePort  30030
└─ Job/benchmark-run-<gpu>     (no Service — short-lived)
```

---

## 4. Four ways to reach Grafana

Pick whichever fits your situation. **`kubectl port-forward` is what we
recommend during dev** because it doesn't need any firewall changes.

### 4a. From inside the cluster (no port-forward, no NodePort)

Any other pod (e.g. a debug shell) can curl Grafana directly:

```bash
kubectl run -n ml-benchmark dbg --rm -it --image=curlimages/curl -- \
  curl -s http://grafana.ml-benchmark.svc.cluster.local:3000/api/health
# {"database":"ok","version":"11.2.0", ...}
```

### 4b. From your laptop via `kubectl port-forward` (recommended for dev)

`port-forward` opens a local TCP socket on your laptop and tunnels it,
through the kube-apiserver, into the Pod's port. No public ports, no SG
changes, no auth headaches.

```bash
cd infra
# Wrapper sets KUBECONFIG and ensures the SSH tunnel to :6443 is running.
./scripts/kubectl_aws.sh port-forward -n ml-benchmark svc/grafana 3000:3000
# Forwarding from 127.0.0.1:3000 -> 3000

# In another terminal:
open http://localhost:3000   # admin / admin (per grafana.yaml)
```

You'll see the Prometheus datasource already wired up and the
**GPU Benchmark Live** dashboard already in the sidebar — that's because we
mount `grafana_dashboard.json` as a `ConfigMap` and Grafana auto-imports it
on startup via the dashboards-provider config (see `grafana.yaml`).

The same trick works for Prometheus (`:9090`) and Pushgateway (`:9091`).

### 4c. From your laptop via NodePort (only if your IP is allow-listed)

We declared the Grafana Service as `NodePort: 30030`. Combined with the
SG rule

```hcl
ingress {
  from_port   = 30000
  to_port     = 32767
  cidr_blocks = var.admin_cidrs    # only YOUR /32, not the world
}
```

your laptop can hit `<controller_public_ip>:30030` directly:

```bash
PUB_IP=$(jq -r '.controller_public_ip' \
  infra/terraform/envs/aws-gpu/inventory.json)
open "http://${PUB_IP}:30030"
```

This is the "show my teammate the live dashboard" path — anyone whose public
IP is in `admin_cidrs` can hit it, no SSH required. **Do not use this for
production**: NodePort + a public IP is a flat HTTP socket with no TLS.

### 4d. From your laptop via raw SSH tunnel (the cluster-agnostic way)

If `kubectl` is unavailable but SSH works:

```bash
ssh -L 3030:127.0.0.1:30030 ubuntu@$(controller_public_ip)
# in another terminal:
open http://localhost:3030
```

---

## 5. What happens during fault-injection (so the recovery story makes sense)

1. `kubectl cordon <node>` — marks the node *unschedulable*. New pods will
   not land on it; existing pods keep running.
2. `kubectl delete pod -l app=benchmark` — kills the running benchmark pod.
   The `Job` controller sees the pod failed and (because `backoffLimit: 1`)
   creates a replacement.
3. `kubectl drain <node>` — evicts all remaining pods from the node.
4. `sleep 30` — measurement window for "how long does the cluster take to
   notice a node is gone?" Prometheus scrapes every 15 s, so we see the
   gap clearly on the dashboard.
5. `kubectl uncordon <node>` — node is schedulable again. The Job's
   replacement pod now has somewhere to run, gets scheduled, and the
   benchmark resumes.

The Grafana dashboard's `up{job="benchmark"}` panel makes the gap and the
recovery extremely visible — that's exactly what we capture for the report.

---

## 6. Cost containment for the fault-inject demo

Run the dedicated orchestrator — it provisions, runs, and **always** tears
down via a `trap`, even on Ctrl-C:

```bash
cd infra
cp terraform/envs/aws-gpu/terraform.tfvars.smoke.example \
   terraform/envs/aws-gpu/terraform.tfvars
# fill in: AMI, key_name, admin_cidrs, bucket_name, cluster_token

./scripts/fault_inject_demo.sh
```

| Step                | Wall-clock (typ.) |
|---------------------|------------------:|
| `provision`         | 3–4 min           |
| `bootstrap`         | 1–2 min           |
| `deploy`            | 1 min             |
| benchmark + fault   | 4–6 min           |
| `log-costs`         | 10 s              |
| `teardown`          | 3 min             |
| **Total**           | **≈ 13–16 min**   |

At smoke rates that's `0.526 + 0.083 = $0.609 / h` × 0.25 h ≈ **$0.15**.
The `trap teardown` runs even if the benchmark fails or you Ctrl-C, so a
runaway cluster is essentially impossible.
