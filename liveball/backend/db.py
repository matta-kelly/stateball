import logging
import threading
import time

import duckdb

from backend.config import settings

logger = logging.getLogger(__name__)

_conn: duckdb.DuckDBPyConnection | None = None
_lock = threading.Lock()


def _create_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute("SET memory_limit = '256MB'")
    conn.execute("INSTALL ducklake; LOAD ducklake;")
    # Strip http(s):// — DuckDB S3 secret expects host:port only
    s3_host = settings.s3_endpoint.removeprefix("https://").removeprefix("http://")
    use_ssl = settings.s3_endpoint.startswith("https://") or settings.s3_use_ssl
    try:
        conn.execute(f"""
            CREATE SECRET s3_secret (
                TYPE S3,
                KEY_ID '{settings.s3_access_key_id}',
                SECRET '{settings.s3_secret_access_key}',
                ENDPOINT '{s3_host}',
                USE_SSL {str(use_ssl).lower()},
                URL_STYLE 'path'
            )
        """)
    except Exception:
        raise RuntimeError("Failed to create S3 secret") from None
    pg = (
        f"host={settings.ducklake_db_host} "
        f"port={settings.ducklake_db_port} "
        f"dbname={settings.ducklake_db_name} "
        f"user={settings.ducklake_db_user} "
        f"password={settings.ducklake_db_password}"
    )
    try:
        conn.execute(
            f"ATTACH 'ducklake:postgres:{pg}' AS lh "
            f"(DATA_PATH 's3://{settings.s3_bucket}/stateball/', "
            f"METADATA_SCHEMA 'lakehouse')"
        )
    except Exception:
        raise RuntimeError("Failed to attach DuckLake") from None
    return conn


def get_conn() -> duckdb.DuckDBPyConnection:
    """Return a cursor from the cached DuckLake connection.

    DuckDB cursors are isolated — safe for concurrent use from a
    single parent connection. The parent is created once and reused.
    Reconnects on failure with one retry. Callers must close the
    cursor when done.
    """
    global _conn
    with _lock:
        if _conn is not None:
            try:
                _conn.execute("SELECT 1")
            except Exception:
                logger.info("Cached DuckLake connection stale, reconnecting")
                _conn = None

        if _conn is None:
            for attempt in range(2):
                try:
                    _conn = _create_conn()
                    _conn.execute("SELECT 1")
                    break
                except Exception:
                    _conn = None
                    if attempt == 0:
                        logger.warning("DuckLake connection failed, retrying in 0.5s")
                        time.sleep(0.5)
                        continue
                    raise

    return _conn.cursor()
