"""Shared utilities for asset modules."""
# ruff: noqa: F401 — re-exports for asset modules.
import os
from pathlib import Path

from ..secrets import load_secrets
from ..lib import (
    get_ducklake_connection, run_dbt, handle_schema_drift,
    ensure_extraction_tracking, record_extracted_game_pks,
)

_S3_BUCKET = os.environ.get("S3_BUCKET", "dazoo")
S3_ARTIFACTS = f"s3://{_S3_BUCKET}/stateball/artifacts"


def _save_json_to_s3(data: dict, path: str, log_fn) -> None:
    """Write a dict as JSON to an S3 path."""
    import json
    from xg.core.io import get_s3fs
    fs = get_s3fs()
    with fs.open(path, "w") as f:
        json.dump(data, f, indent=2)
    log_fn(f"Saved to {path}")


def _xg_config_path(name: str) -> Path:
    """Resolve a config file inside the xg package (works installed or editable)."""
    import xg
    return Path(xg.__file__).parent / "configs" / name

