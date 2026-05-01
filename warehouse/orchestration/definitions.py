"""Dagster definitions for stateball."""
from dagster import (
    AssetSelection,
    DagsterRunStatus,
    DefaultSensorStatus,
    Definitions,
    RunRequest,
    RunsFilter,
    ScheduleDefinition,
    SkipReason,
    define_asset_job,
    sensor,
)
from .secrets import load_secrets

from .assets import (
    raw_games, proc_mlb__games, raw_events, raw_boxscores, proc_mlb__rosters, proc_mlb__events,
    raw_gumbo_events, proc_mlb__gumbo_events, ana_mlb__event_state,
    int_mlb__batter_counts, int_mlb__pitcher_counts,
    int_mlb__batters, int_mlb__pitchers,
    int_mlb__pitcher_arsenal_counts, int_mlb__pitcher_arsenal,
    int_mlb__batter_arsenal_counts, int_mlb__batter_arsenal,
    int_mlb__batter_discipline_counts, int_mlb__batter_discipline,
    int_mlb__batter_profile, int_mlb__pitcher_profile,
    int_mlb__game_state,
    feat_mlb__vectors,
    raw_players, ref_mlb__players,
    cleanup_dbt_orphans, compact_ducklake, gc_ducklake,
    select_features, train_xgboost, sweep_xgboost, validate_artifact, analyze_features,
    build_baserunning_table, build_win_expectancy_table, train_pitcher_exit,
    build_n_lookup_artifact, build_stopping_thresholds_artifact,
    build_gamma_schedule_artifact, build_horizon_weights_artifact,
    calibration_eval, production_eval,
)
from .resources import dbt_resource

load_secrets()

# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

raw_games_job = define_asset_job(
    "raw_games_job",
    selection=AssetSelection.assets(raw_games, proc_mlb__games),
)

raw_events_job = define_asset_job(
    "raw_events_job",
    selection=AssetSelection.assets(raw_events),
)

gumbo_events_job = define_asset_job(
    "gumbo_events_job",
    selection=AssetSelection.assets(raw_gumbo_events, proc_mlb__gumbo_events, ana_mlb__event_state),
)

raw_boxscores_job = define_asset_job(
    "raw_boxscores_job",
    selection=AssetSelection.assets(raw_boxscores, proc_mlb__rosters),
)

proc_pipeline_job = define_asset_job(
    "proc_pipeline_job",
    selection=AssetSelection.assets(
        proc_mlb__events,
        int_mlb__batter_counts, int_mlb__pitcher_counts,
        int_mlb__pitcher_arsenal_counts, int_mlb__batter_arsenal_counts,
        int_mlb__batter_discipline_counts,
        int_mlb__batters, int_mlb__pitchers,
        int_mlb__pitcher_arsenal, int_mlb__batter_arsenal,
        int_mlb__batter_discipline,
        int_mlb__batter_profile, int_mlb__pitcher_profile,
        int_mlb__game_state,
        raw_players, ref_mlb__players,
        feat_mlb__vectors,
    ),
)

# ---------------------------------------------------------------------------
# Schedule + Sensors
# ---------------------------------------------------------------------------

raw_games_schedule = ScheduleDefinition(
    job=raw_games_job,
    cron_schedule="*/30 * * * *",  # every 30 minutes (:00, :30)
)

raw_events_schedule = ScheduleDefinition(
    job=raw_events_job,
    cron_schedule="15 * * * *",  # hourly at :15 (offset from games)
)

gumbo_events_schedule = ScheduleDefinition(
    job=gumbo_events_job,
    cron_schedule="30 * * * *",  # hourly at :30 (raw + proc together, isolated from proc_pipeline_job)
)

raw_boxscores_schedule = ScheduleDefinition(
    job=raw_boxscores_job,
    cron_schedule="45 * * * *",  # hourly at :45 (offset from events)
)

@sensor(
    job=raw_events_job,
    minimum_interval_seconds=300,
    default_status=DefaultSensorStatus.STOPPED,
)
def raw_events_backfill_sensor(context):
    """Backfill sensor — churns through unextracted games every ~5 min. Enable manually."""
    runs = context.instance.get_runs(
        filters=RunsFilter(
            job_name="raw_events_job",
            statuses=[
                DagsterRunStatus.NOT_STARTED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.QUEUED,
            ],
        ),
        limit=1,
    )
    if runs:
        return SkipReason("raw_events_job is currently running or queued")
    return RunRequest()


@sensor(
    job=gumbo_events_job,
    minimum_interval_seconds=300,
    default_status=DefaultSensorStatus.STOPPED,
)
def gumbo_events_backfill_sensor(context):
    """Backfill sensor — churns through unextracted/unprocessed gumbo games every ~5 min. Enable manually."""
    runs = context.instance.get_runs(
        filters=RunsFilter(
            job_name="gumbo_events_job",
            statuses=[
                DagsterRunStatus.NOT_STARTED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.QUEUED,
            ],
        ),
        limit=1,
    )
    if runs:
        return SkipReason("gumbo_events_job is currently running or queued")
    return RunRequest()


@sensor(
    job=raw_boxscores_job,
    minimum_interval_seconds=300,
    default_status=DefaultSensorStatus.STOPPED,
)
def raw_boxscores_backfill_sensor(context):
    """Backfill sensor — churns through unfetched boxscores every ~5 min. Enable manually."""
    runs = context.instance.get_runs(
        filters=RunsFilter(
            job_name="raw_boxscores_job",
            statuses=[
                DagsterRunStatus.NOT_STARTED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.QUEUED,
            ],
        ),
        limit=1,
    )
    if runs:
        return SkipReason("raw_boxscores_job is currently running or queued")
    return RunRequest()


@sensor(
    job=proc_pipeline_job,
    minimum_interval_seconds=300,
    default_status=DefaultSensorStatus.RUNNING,
)
def proc_pipeline_sensor(context):
    """Trigger proc → intermediates → vectors pipeline every ~5 minutes, skipping if already running."""
    runs = context.instance.get_runs(
        filters=RunsFilter(
            job_name="proc_pipeline_job",
            statuses=[
                DagsterRunStatus.NOT_STARTED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.QUEUED,
            ],
        ),
        limit=1,
    )
    if runs:
        return SkipReason("proc_pipeline_job is currently running or queued")
    return RunRequest()


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

# Job-only orchestration. raw_games_job runs raw_games + proc_mlb__games
# every 30 min. Everything else is triggered by jobs/sensors.
# proc_pipeline_job runs proc_mlb__events → intermediates → profiles → feat_vectors.

defs = Definitions(
    assets=[
        raw_games, proc_mlb__games, raw_events, raw_boxscores, proc_mlb__rosters, proc_mlb__events,
        raw_gumbo_events, proc_mlb__gumbo_events, ana_mlb__event_state,
        int_mlb__batter_counts, int_mlb__pitcher_counts,
        int_mlb__batters, int_mlb__pitchers,
        int_mlb__pitcher_arsenal_counts, int_mlb__pitcher_arsenal,
        int_mlb__batter_arsenal_counts, int_mlb__batter_arsenal,
        int_mlb__batter_discipline_counts, int_mlb__batter_discipline,
        int_mlb__batter_profile, int_mlb__pitcher_profile,
        int_mlb__game_state,
        feat_mlb__vectors,
        raw_players, ref_mlb__players,
        cleanup_dbt_orphans, compact_ducklake, gc_ducklake,
        select_features, train_xgboost, sweep_xgboost, validate_artifact, analyze_features,
        build_baserunning_table, build_win_expectancy_table, train_pitcher_exit,
        build_n_lookup_artifact, build_stopping_thresholds_artifact,
        build_gamma_schedule_artifact, build_horizon_weights_artifact,
        calibration_eval, production_eval,
    ],
    schedules=[
        raw_games_schedule, raw_events_schedule, gumbo_events_schedule, raw_boxscores_schedule,
    ],
    sensors=[
        raw_events_backfill_sensor, gumbo_events_backfill_sensor,
        raw_boxscores_backfill_sensor, proc_pipeline_sensor,
    ],
    resources={
        "dbt": dbt_resource,
    },
)

