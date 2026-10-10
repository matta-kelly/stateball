"""Grouped dlt sources and the `@dlt_assets` that drive them.

One `@dlt.source` per source groups its resources; one `@dlt_assets` per source
emits one Dagster asset per resource.
"""
import shutil
import tempfile
from contextlib import contextmanager

import dlt
from dagster import (
    AssetExecutionContext,
    AutomationCondition,
    MaterializeResult,
)
from dagster_dlt import DagsterDltResource, DagsterDltTranslator, dlt_assets

from orchestration.dlt.pipeline import source_pipeline
from orchestration.dlt.sources import kalshi as kalshi_streams
from orchestration.shared import TZ, RefreshConfig

# Markets open a day or more before first pitch, so hourly leaves enormous
# margin to learn about a ticker before anything needs it — and two or three
# consecutive failed pulls still leave the recorder with tonight's roster.
KALSHI_CRON = "0 * * * *"


@contextmanager
def isolated_pipeline(source_name: str, run_id: str):
    """A dlt pipeline in a private, per-run working dir, cleaned up after.

    dagster-dlt drops the pipeline's LOCAL state at the start of every run and
    restores it from the destination. When two runs of the same source share
    ~/.dlt/pipelines/<name>, that drop deletes the other run's in-flight extract
    packages and it dies with LoadPackageNotFound. A per-run dir removes the
    sharing entirely; nothing is lost because the real state lives in Postgres.
    """
    workdir = tempfile.mkdtemp(prefix=f"dlt-{source_name}-{run_id}-")
    try:
        yield source_pipeline(source_name, pipelines_dir=workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _on_cron_ingest(cron: str) -> AutomationCondition:
    """Fire on the cron, WITHOUT gating on the dlt source dep.

    AutomationCondition.on_cron expands to
        in_latest_time_window & cron_tick_passed().since_last_handled()
                              & all_deps_updated_since_cron
    but a dlt asset's only parent is its `<source>_<stream>` dlt *source* asset,
    which is external and never materializes — so all_deps_updated_since_cron is
    never true and the ingest never fires on its own. Only manual pulls would
    ever run. This is a root pull from an external API; it must fire on the clock
    regardless of that never-updating source dep. So keep on_cron's first two
    clauses and drop the deps one.
    """
    return (
        AutomationCondition.in_latest_time_window()
        & AutomationCondition.cron_tick_passed(cron, TZ).since_last_handled()
    ).with_label(f"on_cron_ingest({cron}, {TZ})")


def _run_ingest(context, dlt_resource, source, source_name, *, reset_cursor):
    """Run the source through a per-run pipeline and forward its results."""
    for resource in source.resources.values():
        # Flatten. Nothing kept from the market payload is nested today, but a
        # field Kalshi later turns into an object should land as a json column
        # for dbt to unnest, not silently fan out into a child table.
        resource.max_table_nesting = 0

    refresh = "drop_data" if reset_cursor else None
    with isolated_pipeline(source_name, context.run_id) as pipeline:
        for result in dlt_resource.run(
            context=context,
            dlt_source=source,
            dlt_pipeline=pipeline,
            # csv is the fast path for the postgres destination: dlt loads it
            # with COPY rather than row-by-row inserts.
            loader_file_format="csv",
            refresh=refresh,
        ):
            yield MaterializeResult(
                asset_key=result.asset_key,
                metadata=result.metadata or {},
            )


class IngestSchedule(DagsterDltTranslator):
    """Stamps the emitted asset with its cron condition and group.

    This has to go through the translator rather than the decorator's own
    arguments: `@dlt_assets` takes no automation_condition, and the condition
    has to reach the spec before the AssetsDefinition is built. Keys and deps
    still fall through to the default translator, so the dlt asset keys are
    untouched.
    """

    def __init__(self, cron: str, group: str):
        super().__init__()
        self._cron = cron
        self._group = group

    def get_asset_spec(self, data):
        spec = super().get_asset_spec(data)
        return spec.replace_attributes(
            automation_condition=_on_cron_ingest(self._cron),
            group_name=self._group,
        )


@dlt.source(name="kalshi")
def kalshi_source():
    return [kalshi_streams.markets]


@dlt_assets(
    dlt_source=kalshi_source(),
    dlt_pipeline=source_pipeline("kalshi"),
    name="kalshi",
    dagster_dlt_translator=IngestSchedule(KALSHI_CRON, "kalshi_landing"),
    pool="kalshi",  # one kalshi run at a time — protects the signed API
)
def kalshi_assets(
    context: AssetExecutionContext,
    dlt: DagsterDltResource,
    config: RefreshConfig,
):
    source = kalshi_source()
    yield from _run_ingest(
        context, dlt, source, "kalshi", reset_cursor=config.reset_cursor
    )
