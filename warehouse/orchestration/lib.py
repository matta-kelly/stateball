"""DuckLake connection and dbt run helpers."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import duckdb
from dagster import AssetExecutionContext

DBT = os.path.join(os.path.dirname(sys.executable), "dbt")
_ORCH_DIR = Path(__file__).resolve().parent


def _parse_s3_endpoint(endpoint: str) -> tuple[str, bool]:
    """Extract (hostname, use_ssl) from S3 endpoint URL."""
    if endpoint.startswith("https://"):
        return endpoint[len("https://"):], True
    elif endpoint.startswith("http://"):
        return endpoint[len("http://"):], False
    return endpoint, False


def get_ducklake_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with DuckLake attached.

    Returns an in-memory DuckDB connection with:
    - ducklake + httpfs extensions loaded
    - S3 secret configured from env vars
    - DuckLake catalog attached as 'lakehouse'
    """
    conn = duckdb.connect(":memory:")

    conn.execute("INSTALL ducklake; LOAD ducklake;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")

    # HTTP logging — stored in-memory, queryable via duckdb_logs_parsed('HTTP')
    conn.execute("CALL enable_logging('HTTP');")

    # S3 resilience — match profiles.yml settings for dbt parity
    conn.execute("SET http_retries = 10;")
    conn.execute("SET http_retry_wait_ms = 1000;")
    conn.execute("SET http_retry_backoff = 2;")
    conn.execute("SET http_timeout = 120;")

    s3_key = os.environ.get("S3_ACCESS_KEY_ID", "")
    s3_secret = os.environ.get("S3_SECRET_ACCESS_KEY", "")
    s3_endpoint = os.environ.get("S3_ENDPOINT", "")
    s3_host, s3_ssl = _parse_s3_endpoint(s3_endpoint)

    conn.execute(f"""
        CREATE SECRET s3_secret (
            TYPE S3,
            KEY_ID '{s3_key}',
            SECRET '{s3_secret}',
            ENDPOINT '{s3_host}',
            USE_SSL {str(s3_ssl).lower()},
            URL_STYLE 'path'
        )
    """)

    pg_host = os.environ.get("DUCKLAKE_DB_HOST", "localhost")
    pg_port = os.environ.get("DUCKLAKE_DB_PORT", "5432")
    pg_pass = os.environ.get("DUCKLAKE_DB_PASSWORD", "")

    s3_bucket = os.environ.get("S3_BUCKET", "dazoo")
    conn.execute(f"""
        ATTACH 'ducklake:postgres:host={pg_host} port={pg_port} dbname=ducklake user=ducklake password={pg_pass}'
        AS lakehouse (DATA_PATH 's3://{s3_bucket}/stateball/', METADATA_SCHEMA 'lakehouse')
    """)

    return conn


# ---------------------------------------------------------------------------
# Extraction tracking
# ---------------------------------------------------------------------------


def ensure_extraction_tracking(conn: duckdb.DuckDBPyConnection) -> None:
    """Create extraction tracking table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lakehouse.landing.extracted_game_pks (
            game_pk INTEGER,
            extracted_at TIMESTAMPTZ
        )
    """)


def record_extracted_game_pks(conn, game_pks: set[int], log=None) -> None:
    """Record successfully extracted game_pks in the tracking table."""
    if not game_pks:
        return
    ensure_extraction_tracking(conn)
    values = ", ".join(f"({pk}, now())" for pk in sorted(game_pks))
    conn.execute(f"INSERT INTO lakehouse.landing.extracted_game_pks VALUES {values}")
    if log:
        log.info(f"[tracking] Recorded {len(game_pks)} extracted game_pks")


# ---------------------------------------------------------------------------
# dbt run helpers
# ---------------------------------------------------------------------------


def run_dbt(
    context: AssetExecutionContext,
    model: str,
    *,
    vars: dict | None = None,
    full_refresh: bool = False,
    seed: bool = False,
) -> None:
    """Run a single dbt model (or seed) via subprocess.

    Centralizes the subprocess pattern used by all dbt-calling assets.
    Logs stdout, raises on non-zero exit.
    """
    dbt_cmd = "seed" if seed else "run"
    cmd = [
        DBT, dbt_cmd,
        "--select", model,
        "--project-dir", str(_ORCH_DIR),
        "--profiles-dir", str(_ORCH_DIR),
    ]
    if vars and not seed:
        cmd.extend(["--vars", json.dumps(vars)])
    if full_refresh and not seed:
        cmd.append("--full-refresh")

    context.log.info(f"Running dbt {dbt_cmd}: {model}" + (f" (vars: {vars})" if vars else ""))
    result = subprocess.run(cmd, capture_output=True, text=True)
    context.log.info(result.stdout)
    if result.returncode != 0:
        context.log.error(result.stderr)
        raise Exception(f"dbt {dbt_cmd} failed for {model}:\n{result.stdout}\n{result.stderr}")



# ---------------------------------------------------------------------------
# Schema drift detection
# ---------------------------------------------------------------------------


def _compile_dbt_models() -> bool:
    """Compile all dbt models with --full-refresh.

    Uses a filesystem marker to skip recompilation within the same
    pipeline run (5-min TTL matches sensor interval).
    """
    marker = _ORCH_DIR / "target" / ".drift_compiled"
    if marker.exists() and (time.time() - marker.stat().st_mtime) < 300:
        return True
    result = subprocess.run([
        DBT, "compile", "--full-refresh",
        "--project-dir", str(_ORCH_DIR),
        "--profiles-dir", str(_ORCH_DIR),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    marker.touch()
    return True


def handle_schema_drift(
    context: AssetExecutionContext,
    model: str,
) -> bool:
    """Detect schema drift and drop table if columns changed.

    Compiles the model with --full-refresh (no {{ this }} references),
    uses DESCRIBE to get expected columns, compares with existing table.
    Drops the table if mismatch detected.

    Returns True if the table was dropped (caller rebuilds from scratch).
    """
    if not _compile_dbt_models():
        context.log.warning(f"dbt compile failed, skipping drift check for {model}")
        return False

    compiled_dir = _ORCH_DIR / "target" / "compiled" / "stateball"
    matches = list(compiled_dir.glob(f"**/{model}.sql"))
    if not matches:
        context.log.warning(f"No compiled SQL found for {model}")
        return False
    compiled_sql = matches[0].read_text().strip().rstrip(";")

    conn = get_ducklake_connection()
    try:
        try:
            existing = {row[0] for row in
                        conn.execute(f"DESCRIBE lakehouse.main.{model}").fetchall()}
        except Exception:
            return False  # Table doesn't exist — will be created fresh

        try:
            expected = {row[0] for row in
                        conn.execute(f"DESCRIBE ({compiled_sql})").fetchall()}
        except Exception as e:
            context.log.warning(f"DESCRIBE failed for {model}: {e}")
            return False

        if existing == expected:
            return False

        added = expected - existing
        removed = existing - expected
        context.log.warning(
            f"Schema drift on {model}: "
            f"+{len(added)} cols, -{len(removed)} cols"
        )
        if added:
            context.log.info(f"  Added: {sorted(added)}")
        if removed:
            context.log.info(f"  Removed: {sorted(removed)}")

        conn.execute(f"DROP TABLE IF EXISTS lakehouse.main.{model}")
        context.log.info(f"Dropped {model} for schema rebuild")
        return True
    finally:
        conn.close()
