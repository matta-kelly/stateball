"""Shared I/O utilities for the xg package."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("xg.core.io")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def log_memory(label: str):
    """Log current RSS memory in MB. Linux only (reads /proc/self/status)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    logger.debug("[mem] %s: %d MB", label, int(line.split()[1]) / 1024)
                    return
    except Exception:
        pass


def get_s3fs():
    """Create s3fs filesystem from environment variables."""
    import s3fs
    return s3fs.S3FileSystem(
        key=os.environ.get("S3_ACCESS_KEY_ID", ""),
        secret=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
        endpoint_url=os.environ.get("S3_ENDPOINT", ""),
    )
