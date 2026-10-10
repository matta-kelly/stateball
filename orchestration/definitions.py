"""stateball Dagster code location."""

from dagster import (
    AutomationConditionSensorDefinition,
    DefaultSensorStatus,
    Definitions,
)
from dagster_dlt import DagsterDltResource

from orchestration.ingestion import kalshi_assets

# Ticks every ~30s and evaluates each asset's automation condition — the on_cron
# ingest clock, and the eager transforms once there are any. Without it those
# conditions never fire. RUNNING so it is armed the moment the code location
# loads, no UI toggle.
automation_sensor = AutomationConditionSensorDefinition(
    name="automation_condition_sensor",
    target="*",
    default_status=DefaultSensorStatus.RUNNING,
)

defs = Definitions(
    assets=[kalshi_assets],
    sensors=[automation_sensor],
    resources={
        "dlt": DagsterDltResource(),
    },
)
