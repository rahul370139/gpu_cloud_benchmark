FROM nvcr.io/nvidia/pytorch:24.08-py3

LABEL maintainer="Rahul Sharma"
LABEL description="GPU Cloud Benchmark — reproducible ML workload benchmarking"

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
COPY user_workloads/ user_workloads/
RUN chmod +x scripts/entrypoint.sh

RUN mkdir -p results/figures

ENV BENCHMARK_CONFIG=/app/config/benchmark_config.yaml
ENV BENCHMARK_RESULTS_DIR=/app/results
ENV PYTHONUNBUFFERED=1

VOLUME ["/app/results"]

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
