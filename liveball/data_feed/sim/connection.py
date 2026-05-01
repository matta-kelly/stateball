"""DuckLake connection for data feed — no dagster dependency."""

import os

import duckdb


def get_ducklake_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with DuckLake attached.

    Reads S3 and DuckLake credentials from env vars.
    Equivalent to warehouse/orchestration/lib.py:get_ducklake_connection()
    but without the dagster import.
    """
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL ducklake; LOAD ducklake;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")

    s3_key = os.environ.get("S3_ACCESS_KEY_ID", "")
    s3_secret = os.environ.get("S3_SECRET_ACCESS_KEY", "")
    s3_endpoint = os.environ.get("S3_ENDPOINT", "")

    use_ssl = s3_endpoint.startswith("https://")
    host = s3_endpoint.removeprefix("https://").removeprefix("http://")

    conn.execute(f"""
        CREATE SECRET s3_secret (
            TYPE S3,
            KEY_ID '{s3_key}',
            SECRET '{s3_secret}',
            ENDPOINT '{host}',
            USE_SSL {str(use_ssl).lower()},
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
