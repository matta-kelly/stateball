"""Shared Dagster wiring: the dbt project and the per-run knobs.

Ingestion assets live in `ingestion`, transform in `transform`; the code
location is `definitions`.
"""
from pathlib import Path

from dagster import Config
from dagster_dbt import DbtCliResource, DbtProject

# Every schedule in the code location is anchored here. It is UTC rather than
# a local zone: market and game clocks are natively UTC, and nothing
# here is pinned to an hour of a business day, so there is no DST drift to hold
# still. Matches the session TimeZone in profiles.yml.
TZ = "UTC"

# The dbt project sits alongside this package (dbt_project.yml + profiles.yml).
_DBT_DIR = Path(__file__).parent

# prepare_if_dev reparses the manifest at load under `dagster dev` so source keys
# reflect the real env rather than whatever last wrote target/manifest.json.
# Outside dev it is a no-op and the existing manifest is used as-is.
dbt_project = DbtProject(project_dir=_DBT_DIR)
dbt_project.prepare_if_dev()

dbt_resource = DbtCliResource(project_dir=dbt_project)


class DbtConfig(Config):
    """Per-run transform knob. full_refresh drops and rebuilds instead of
    running incrementally."""
    full_refresh: bool = False


class RefreshConfig(Config):
    """Per-run reset knob for an ingest asset — the cursor scalpel.

    reset_cursor flips the run from a normal pull to dlt's `refresh="drop_data"`:
    it truncates the selected resource's tables and wipes its incremental state,
    so the run reloads from scratch. SCOPED TO THE RUN'S SELECTION. Off by
    default, so a normal cron or manual run never resets. Flip it, materialize,
    and it is back off next run.
    """
    reset_cursor: bool = False
