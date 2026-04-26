"""Tests for S3 artifact upload helpers."""

import sys
import types

from src.artifacts.s3_uploader import maybe_upload_results


def test_maybe_upload_results_uses_expected_prefix(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)
    (results_dir / "benchmark_summary_T4.csv").write_text("ok\n")
    (figures_dir / "throughput.png").write_text("png\n")

    uploads = []

    class _FakeS3Client:
        def upload_file(self, filename, bucket, key, **kwargs):
            uploads.append((filename, bucket, key, kwargs))

    fake_boto3 = types.SimpleNamespace(client=lambda *_args, **_kwargs: _FakeS3Client())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    uploaded = maybe_upload_results(
        results_dir=results_dir,
        bucket="example-bucket",
        region="us-east-1",
        run_id="run-123",
        gpu_class="T4",
        pod_name="benchmark-run-t4-abc",
    )

    assert len(uploaded) == 2
    assert uploads[0][1] == "example-bucket"
    assert uploads[0][2].startswith("benchmark-runs/run-123/T4/benchmark-run-t4-abc/")
    assert uploads[1][2].endswith("/figures/throughput.png")
