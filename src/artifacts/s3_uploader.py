"""Upload benchmark artifacts to S3."""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize(value: str | None) -> str | None:
    if value in (None, "", "null"):
        return None
    return value


def _build_prefix(
    run_id: str | None,
    execution_id: str | None,
    gpu_class: str | None,
    pod_name: str | None,
) -> str:
    parts = ["benchmark-runs"]
    if run_id:
        parts.append(run_id)
    if execution_id:
        parts.extend(["executions", execution_id])
    if gpu_class:
        parts.append(gpu_class)
    if pod_name:
        parts.append(pod_name)
    return "/".join(parts)


def upload_directory_to_s3(
    results_dir: str | Path,
    bucket: str,
    prefix: str,
    region: str | None = None,
) -> list[str]:
    """Upload a directory tree to S3 and return uploaded URIs."""

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 artifact uploads") from exc

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_path}")

    client_kwargs = {"region_name": region} if region else {}
    s3 = boto3.client("s3", **client_kwargs)

    uploaded: list[str] = []
    for file_path in sorted(p for p in results_path.rglob("*") if p.is_file()):
        relative_key = file_path.relative_to(results_path).as_posix()
        object_key = f"{prefix.rstrip('/')}/{relative_key}" if prefix else relative_key
        extra_args = {}
        content_type, _ = mimetypes.guess_type(file_path.name)
        if content_type:
            extra_args["ExtraArgs"] = {"ContentType": content_type}
        s3.upload_file(str(file_path), bucket, object_key, **extra_args)
        uploaded.append(f"s3://{bucket}/{object_key}")
    return uploaded


def maybe_upload_results(
    results_dir: str | Path,
    bucket: str | None = None,
    region: str | None = None,
    run_id: str | None = None,
    execution_id: str | None = None,
    gpu_class: str | None = None,
    pod_name: str | None = None,
) -> list[str]:
    """Upload benchmark artifacts to S3 if a bucket is configured."""

    bucket = _normalize(bucket or os.environ.get("BENCHMARK_ARTIFACT_BUCKET"))
    if not bucket:
        logger.info("No BENCHMARK_ARTIFACT_BUCKET configured — S3 upload skipped")
        return []

    region = _normalize(region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    run_id = _normalize(run_id or os.environ.get("BENCHMARK_RUN_ID"))
    execution_id = _normalize(execution_id or os.environ.get("BENCHMARK_EXECUTION_ID"))
    gpu_class = _normalize(gpu_class or os.environ.get("BENCHMARK_GPU_CLASS"))
    pod_name = _normalize(pod_name or os.environ.get("POD_NAME"))

    prefix = _build_prefix(
        run_id=run_id,
        execution_id=execution_id,
        gpu_class=gpu_class,
        pod_name=pod_name,
    )
    uploaded = upload_directory_to_s3(
        results_dir=results_dir,
        bucket=bucket,
        prefix=prefix,
        region=region,
    )
    logger.info(
        "Uploaded %d artifact(s) to s3://%s/%s",
        len(uploaded),
        bucket,
        prefix,
    )
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload benchmark artifacts to S3")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--execution-id", default=None)
    parser.add_argument("--gpu-class", default=None)
    parser.add_argument("--pod-name", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    maybe_upload_results(
        results_dir=args.results_dir,
        bucket=args.bucket,
        region=args.region,
        run_id=args.run_id,
        execution_id=args.execution_id,
        gpu_class=args.gpu_class,
        pod_name=args.pod_name,
    )


if __name__ == "__main__":
    main()
