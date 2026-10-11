"""dlt pipeline factory — Postgres destination."""

import os

import dlt


def _credentials() -> str:
    """Postgres connection string from env. Same vars dbt and compose read."""
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "stateball")
    user = os.environ.get("POSTGRES_USER", "stateball")
    password = os.environ["POSTGRES_PASSWORD"]
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def source_pipeline(source_name: str, pipelines_dir: str | None = None) -> dlt.Pipeline:
    """One dlt pipeline per source. `dataset_name` becomes the Postgres schema.

    `@dlt_assets` runs the whole source through this pipeline and emits one
    asset per resource; dlt keeps each resource's incremental state isolated
    inside the pipeline's own tables.

    pipelines_dir: where dlt keeps its LOCAL working state (extract/normalize
        packages, a schema + state cache). Default None uses the shared
        ~/.dlt/pipelines/<name>. dagster-dlt drops and restores that local state
        from the destination on every run, so two runs of the same pipeline
        sharing it will wipe each other's in-flight packages (LoadPackageNotFound
        mid-extract). Passing a per-run dir isolates them.
    """
    return dlt.pipeline(
        pipeline_name=source_name,
        destination=dlt.destinations.postgres(credentials=_credentials()),
        dataset_name=source_name,
        pipelines_dir=pipelines_dir,
    )
