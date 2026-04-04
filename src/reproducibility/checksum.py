"""SHA-256 checksums for artifacts, images, and environment snapshots."""

import hashlib
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1 << 20  # 1 MiB


def compute_file_sha256(path: str | Path) -> str:
    """Return hex SHA-256 digest of a file, reading in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    digest = h.hexdigest()
    logger.debug("SHA-256(%s) = %s", path, digest)
    return digest


def compute_string_sha256(text: str) -> str:
    """Return hex SHA-256 of a UTF-8 string (e.g. pip freeze output)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_pip_freeze_hash() -> tuple[str, str]:
    """Run pip freeze and return (freeze_output, sha256_of_output)."""
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        freeze = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        freeze = "UNAVAILABLE"
    return freeze, compute_string_sha256(freeze)


def compute_docker_image_id() -> str:
    """Read the Docker image ID from /proc/self/cgroup or hostname."""
    # Inside a container the hostname is typically the short container ID
    try:
        result = subprocess.run(
            ["hostname"], capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "UNKNOWN"


def checksum_directory(directory: str | Path, pattern: str = "*.csv") -> dict[str, str]:
    """Compute SHA-256 for every file matching *pattern* under *directory*."""
    directory = Path(directory)
    checksums = {}
    for fp in sorted(directory.rglob(pattern)):
        checksums[str(fp.relative_to(directory))] = compute_file_sha256(fp)
    return checksums
