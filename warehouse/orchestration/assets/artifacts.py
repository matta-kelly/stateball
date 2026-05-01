"""Simulation artifact builders: baserunning, win expectancy, pitcher exit, calibration artifacts."""
from datetime import datetime, timezone

from dagster import asset, AssetExecutionContext

from ._shared import (
    S3_ARTIFACTS, _save_json_to_s3,
    load_secrets, get_ducklake_connection,
)
from ._configs import CalibrationArtifactConfig, PitcherExitConfig
from .mlb import proc_mlb__events, proc_mlb__games, int_mlb__pitchers


@asset(
    group_name="artifacts",
    deps=[proc_mlb__events],
)
def build_baserunning_table(context: AssetExecutionContext):
    """Build empirical baserunning transition table from proc_mlb__events.

    Queries all PA-ending pitches, computes runner advancement frequencies
    by (outcome, pre_base_state, outs), writes JSON artifact to S3.
    """
    load_secrets()
    from sim.engine.lookups.baserunning import build, save, validate

    conn = get_ducklake_connection()
    try:
        context.log.info("Building baserunning transition table from proc_mlb__events")
        table = build(conn)
    finally:
        conn.close()

    validate(table)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = f"{S3_ARTIFACTS}/sim/{run_id}/baserunning.json"
    save(table, output_path)
    context.log.info(f"Artifact: {output_path}")

    # Register in artifact registry
    from sim.infra.artifact_catalog import register_artifact
    conn = get_ducklake_connection()
    try:
        register_artifact(conn, "baserunning", run_id, output_path, table["metadata"])
    finally:
        conn.close()


@asset(
    group_name="artifacts",
    deps=[proc_mlb__events, proc_mlb__games],
)
def build_win_expectancy_table(context: AssetExecutionContext):
    """Build empirical win expectancy table from proc_mlb__events + proc_mlb__games.

    Computes P(home_win) by (inning, half, outs, base_state, run_diff).
    Hierarchical fallback for sparse states. JSON artifact to S3.
    """
    load_secrets()
    from sim.engine.lookups.win_expectancy import build, save, validate

    conn = get_ducklake_connection()
    try:
        context.log.info("Building win expectancy table from proc_mlb__events + proc_mlb__games")
        table = build(conn)
    finally:
        conn.close()

    validate(table)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = f"{S3_ARTIFACTS}/sim/{run_id}/win_expectancy.json"
    save(table, output_path)
    context.log.info(f"Artifact: {output_path}")

    # Register in artifact registry
    from sim.infra.artifact_catalog import register_artifact
    conn = get_ducklake_connection()
    try:
        register_artifact(conn, "win_expectancy", run_id, output_path, table["metadata"])
    finally:
        conn.close()


@asset(
    group_name="training",
    deps=[proc_mlb__events, int_mlb__pitchers],
)
def train_pitcher_exit(context: AssetExecutionContext, config: PitcherExitConfig):
    """Train pitcher exit XGBoost binary classifier + isotonic calibration.

    Saves decomposed artifacts (booster.ubj + calibration.json + metadata.json).
    """
    load_secrets()
    from xg.pitcher_exit import build, save, validate

    conn = get_ducklake_connection()
    try:
        context.log.info("Training pitcher exit model from int_mlb__game_state + int_mlb__pitchers")
        result = build(conn, sweep=config.sweep)
        metrics = result["metadata"]["model_metrics"]
        context.log.info(
            f"Trained model: {result['metadata']['n_training_rows']:,} training rows, "
            f"{metrics['n_trees']} trees, AUC={metrics['auc']:.4f}, Brier={metrics['brier']:.4f}"
        )
    finally:
        conn.close()

    validate(result)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = f"{S3_ARTIFACTS}/pitcher_exit/{run_id}"
    save(result, output_path)
    context.log.info(f"Artifact: {output_path}")

    # Register in artifact registry
    from sim.infra.artifact_catalog import register_artifact
    conn = get_ducklake_connection()
    try:
        register_artifact(conn, "pitcher_exit", run_id, output_path, result["metadata"])
    finally:
        conn.close()


# ============================================================================
# Calibration artifacts (built from eval tables)
# ============================================================================


@asset(group_name="artifacts", deps=["calibration_eval"])
def build_n_lookup_artifact(context: AssetExecutionContext, config: CalibrationArtifactConfig):
    """Build per-state N allocation from a calibration eval."""
    if not config.eval_id:
        raise ValueError("eval_id is required")
    load_secrets()
    from sim.infra.artifact_catalog import register_artifact, get_slot_artifacts
    from sim.engine.estimators.calibration_builders import build_n_lookup
    from sim.infra.artifact_loaders import load_win_expectancy_table
    from sim.engine.lookups.win_expectancy import build_lookup_array

    conn = get_ducklake_connection()
    try:
        context.log.info(f"Building n_lookup from eval {config.eval_id}")
        slot_arts = get_slot_artifacts(conn, slot="prod")
        we_path = slot_arts.get("win_expectancy", {}).get("s3_path")
        we_array = None
        if we_path:
            we_array = build_lookup_array(load_win_expectancy_table(we_path))

        artifact = build_n_lookup(conn, config.eval_id, we_array=we_array)
        if not artifact.get("lookup"):
            context.log.info(f"No convergence data for eval {config.eval_id}")
            return

        s3_path = f"{S3_ARTIFACTS}/n_lookup/{config.eval_id}/n_lookup.json"
        _save_json_to_s3(artifact, s3_path, context.log.info)
        register_artifact(conn, "n_lookup", config.eval_id, s3_path, artifact["metadata"])
        context.add_output_metadata({"eval_id": config.eval_id, "n_states": artifact["metadata"]["n_states"]})
    finally:
        conn.close()


@asset(group_name="artifacts", deps=["calibration_eval"])
def build_stopping_thresholds_artifact(context: AssetExecutionContext, config: CalibrationArtifactConfig):
    """Build per-state SE stopping thresholds from a calibration eval."""
    if not config.eval_id:
        raise ValueError("eval_id is required")
    load_secrets()
    from sim.infra.artifact_catalog import register_artifact
    from sim.engine.estimators.calibration_builders import build_stopping_thresholds

    conn = get_ducklake_connection()
    try:
        context.log.info(f"Building stopping_thresholds from eval {config.eval_id}")
        artifact = build_stopping_thresholds(conn, config.eval_id)
        if not artifact.get("lookup"):
            context.log.info(f"No convergence data for eval {config.eval_id}")
            return

        s3_path = f"{S3_ARTIFACTS}/stopping_thresholds/{config.eval_id}/stopping_thresholds.json"
        _save_json_to_s3(artifact, s3_path, context.log.info)
        register_artifact(conn, "stopping_thresholds", config.eval_id, s3_path, artifact["metadata"])
        context.add_output_metadata({"eval_id": config.eval_id, "n_states": artifact["metadata"]["n_states"]})
    finally:
        conn.close()


@asset(group_name="artifacts", deps=["calibration_eval"])
def build_gamma_schedule_artifact(context: AssetExecutionContext, config: CalibrationArtifactConfig):
    """Build per-state SMC gamma tempering from a calibration eval."""
    if not config.eval_id:
        raise ValueError("eval_id is required")
    load_secrets()
    from sim.infra.artifact_catalog import register_artifact, get_slot_artifacts
    from sim.engine.estimators.calibration_builders import build_gamma_schedule
    from sim.infra.artifact_loaders import load_win_expectancy_table
    from sim.engine.lookups.win_expectancy import build_sensitivity_array

    conn = get_ducklake_connection()
    try:
        context.log.info(f"Building gamma_schedule from eval {config.eval_id}")
        slot_arts = get_slot_artifacts(conn, slot="prod")
        we_path = slot_arts.get("win_expectancy", {}).get("s3_path")
        sens_array = None
        if we_path:
            sens_array = build_sensitivity_array(load_win_expectancy_table(we_path))

        artifact = build_gamma_schedule(conn, config.eval_id, sensitivity_array=sens_array)
        if not artifact.get("lookup"):
            context.log.info(f"No convergence data for eval {config.eval_id}")
            return

        s3_path = f"{S3_ARTIFACTS}/gamma_schedule/{config.eval_id}/gamma_schedule.json"
        _save_json_to_s3(artifact, s3_path, context.log.info)
        register_artifact(conn, "gamma_schedule", config.eval_id, s3_path, artifact["metadata"])
        context.add_output_metadata({"eval_id": config.eval_id, "n_states": artifact["metadata"]["n_states"]})
    finally:
        conn.close()


@asset(group_name="artifacts", deps=["calibration_eval"])
def build_horizon_weights_artifact(context: AssetExecutionContext, config: CalibrationArtifactConfig):
    """Build per-state horizon weights from a calibration eval."""
    if not config.eval_id:
        raise ValueError("eval_id is required")
    load_secrets()
    from sim.infra.artifact_catalog import register_artifact
    from sim.engine.estimators.calibration_builders import build_horizon_weights

    conn = get_ducklake_connection()
    try:
        context.log.info(f"Building horizon_weights from eval {config.eval_id}")
        artifact = build_horizon_weights(conn, config.eval_id)
        if not artifact.get("lookup"):
            context.log.info(f"No horizon data for eval {config.eval_id}")
            return

        s3_path = f"{S3_ARTIFACTS}/horizon_weights/{config.eval_id}/horizon_weights.json"
        _save_json_to_s3(artifact, s3_path, context.log.info)
        register_artifact(conn, "horizon_weights", config.eval_id, s3_path, artifact["metadata"])
        context.add_output_metadata({"eval_id": config.eval_id, "n_states": artifact["metadata"]["n_states"]})
    finally:
        conn.close()
