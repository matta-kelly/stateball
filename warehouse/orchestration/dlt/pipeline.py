"""dlt pipeline runner."""
import os

import dlt
from dlt.destinations.impl.ducklake.configuration import DuckLakeCredentials


def create_pipeline(source_name: str) -> dlt.Pipeline:
    """Create dlt pipeline with DuckLake destination."""
    pg_host = os.environ.get("DUCKLAKE_DB_HOST", "localhost")
    pg_pass = os.environ.get("DUCKLAKE_DB_PASSWORD", "")

    s3_key = os.environ.get("S3_ACCESS_KEY_ID", "")
    s3_secret = os.environ.get("S3_SECRET_ACCESS_KEY", "")
    s3_endpoint = os.environ.get("S3_ENDPOINT", "")
    s3_bucket = os.environ.get("S3_BUCKET", "dazoo")

    credentials = DuckLakeCredentials(
        "lakehouse",
        catalog=f"postgresql://ducklake:{pg_pass}@{pg_host}:5432/ducklake",
        storage={
            "bucket_url": f"s3://{s3_bucket}/stateball",
            "credentials": {
                "aws_access_key_id": s3_key,
                "aws_secret_access_key": s3_secret,
                "endpoint_url": s3_endpoint,
            },
        },
    )

    return dlt.pipeline(
        pipeline_name=source_name,
        destination=dlt.destinations.ducklake(credentials=credentials),
        dataset_name="landing",
    )


def run(source: str, stream: str, resource, log=None):
    """Run dlt pipeline for a resource.

    Args:
        source: Source name (e.g., 'mlb')
        stream: Stream name (e.g., 'games')
        resource: dlt resource generator
        log: Optional logger
    """
    pipeline = create_pipeline(source)

    # Flatten nested JSON
    resource.max_table_nesting = 0

    if log:
        log.info(f"[pipeline] Running {source}/{stream}")

    info = pipeline.run(resource, table_name=stream, loader_file_format="parquet")

    if log:
        log.info(f"[pipeline] Load info: {info}")

    return info
