"""Unit tests for reproducibility utilities."""

import hashlib
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from src.reproducibility.seed_manager import set_deterministic
from src.reproducibility.checksum import compute_file_sha256, compute_string_sha256, checksum_directory
from src.reproducibility.env_capture import capture_environment


class TestSeedManager:
    def test_returns_summary(self):
        summary = set_deterministic(42)
        assert summary["seed"] == 42
        assert summary["python_random"] is True
        assert summary["numpy_random"] is True
        assert summary["torch_manual_seed"] is True

    def test_deterministic_torch(self):
        set_deterministic(123)
        a = torch.randn(5)
        set_deterministic(123)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_deterministic_numpy(self):
        set_deterministic(456)
        a = np.random.rand(5)
        set_deterministic(456)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_deterministic(1)
        a = torch.randn(5)
        set_deterministic(2)
        b = torch.randn(5)
        assert not torch.allclose(a, b)


class TestChecksum:
    def test_file_sha256(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        digest = compute_file_sha256(f)
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert digest == expected

    def test_string_sha256(self):
        digest = compute_string_sha256("test")
        expected = hashlib.sha256(b"test").hexdigest()
        assert digest == expected

    def test_checksum_directory(self, tmp_path):
        (tmp_path / "a.csv").write_text("data1")
        (tmp_path / "b.csv").write_text("data2")
        (tmp_path / "c.txt").write_text("ignored")
        checksums = checksum_directory(tmp_path, pattern="*.csv")
        assert len(checksums) == 2
        assert "a.csv" in checksums
        assert "b.csv" in checksums


class TestEnvCapture:
    def test_captures_basic_fields(self):
        env = capture_environment()
        assert "python_version" in env
        assert "torch_version" in env
        assert "timestamp_utc" in env
        assert "hostname" in env

    def test_pip_freeze_hash_present(self):
        env = capture_environment()
        assert "pip_freeze_sha256" in env
        assert len(env["pip_freeze_sha256"]) == 64  # SHA-256 hex length
