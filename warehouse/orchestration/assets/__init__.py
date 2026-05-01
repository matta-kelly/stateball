"""Dagster asset definitions for MLB pipeline."""
# ruff: noqa: F401 — all imports are intentional re-exports.

from ._configs import (
    RawEventsConfig,
    RawBoxscoresConfig,
    RawGumboEventsConfig,
    ProcGumboEventsConfig,
    ProcEventsConfig,
    DbtConfig,
    CalibrationConfig,
    CalibrationArtifactConfig,
    ProductionConfig,
    SelectFeaturesConfig,
    TrainXgboostConfig,
    PitcherExitConfig,
    AnalyzeFeaturesConfig,
)

from .mlb import (
    raw_games,
    proc_mlb__games,
    raw_events,
    raw_boxscores,
    proc_mlb__rosters,
    proc_mlb__events,
    raw_gumbo_events,
    proc_mlb__gumbo_events,
    ana_mlb__event_state,
    int_mlb__batter_counts,
    int_mlb__pitcher_counts,
    int_mlb__batters,
    int_mlb__pitchers,
    int_mlb__pitcher_arsenal_counts,
    int_mlb__pitcher_arsenal,
    int_mlb__batter_arsenal_counts,
    int_mlb__batter_arsenal,
    int_mlb__batter_discipline_counts,
    int_mlb__batter_discipline,
    int_mlb__batter_profile,
    int_mlb__pitcher_profile,
    int_mlb__game_state,
    feat_mlb__vectors,
    raw_players,
    ref_mlb__players,
)

from .maintenance import (
    cleanup_dbt_orphans,
    compact_ducklake,
    gc_ducklake,
)

from .training import (
    validate_artifact,
    analyze_features,
    select_features,
    train_xgboost,
    sweep_xgboost,
    _sweep_and_retrain,
)

from .artifacts import (
    build_baserunning_table,
    build_win_expectancy_table,
    train_pitcher_exit,
    build_n_lookup_artifact,
    build_stopping_thresholds_artifact,
    build_gamma_schedule_artifact,
    build_horizon_weights_artifact,
)

from .simulation import (
    calibration_eval,
    production_eval,
)
