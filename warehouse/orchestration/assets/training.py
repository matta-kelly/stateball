"""Training assets: feature selection, XGBoost training, sweep, validation, analysis."""
import json
import os
from datetime import datetime, timezone

from dagster import asset, AssetExecutionContext

from ._shared import (
    S3_ARTIFACTS, load_secrets, get_ducklake_connection, _xg_config_path,
)
from ._configs import (
    SelectFeaturesConfig, TrainXgboostConfig, AnalyzeFeaturesConfig,
)
from .mlb import feat_mlb__vectors


@asset(
    group_name="training",
)
def validate_artifact(context: AssetExecutionContext):
    """Validate an existing model artifact across splits and strata.

    Loads a registered artifact, evaluates it with Brier decomposition,
    stratified calibration, and bootstrap CIs. No model is trained.
    Report: s3://<artifact_path>/validation_report.json
    """
    load_secrets()
    from xg.core.config import load_config
    from xg.pa.validate import validate_artifact as _validate

    config_path = _xg_config_path("default.toml")
    cfg = load_config(config_path)
    cfg.data.connection_mode = "ducklake"

    # Get artifact_id from run config (must be provided at materialization)
    artifact_id = os.environ.get("VALIDATE_ARTIFACT_ID", "")
    if not artifact_id:
        raise ValueError("VALIDATE_ARTIFACT_ID env var required")

    context.log.info(f"Validating artifact: {artifact_id}")

    report = _validate(artifact_id, cfg, n_splits=10)

    context.log.info(
        f"Validation complete. Brier={report['brier_score']['mean']:.6f} "
        f"± {report['brier_score']['std']:.6f}"
    )
    context.log.info(
        f"Reliability={report['brier_reliability']['mean']:.6f}  "
        f"Resolution={report['brier_resolution']['mean']:.6f}"
    )
    context.log.info(f"Bootstrap 95%% CI: [{report['bootstrap_ci']['ci_lower']:.6f}, {report['bootstrap_ci']['ci_upper']:.6f}]")
    if "stratified_calibration" in report:
        for name, vals in report["stratified_calibration"].items():
            context.log.info(f"  {name} (n={vals['n']}): reliability={vals['reliability']:.6f}")


@asset(group_name="training")
def analyze_features(context: AssetExecutionContext, config: AnalyzeFeaturesConfig):
    """Post-hoc feature analysis: importance, SHAP attribution, correlation.

    Loads a trained model artifact from S3, reloads test data from DuckLake,
    and produces feature_analysis.json in the artifact directory.

    Resolves artifact path from the registry (test slot, xgboost_sim) by default.
    Override via launchpad config or ANALYZE_ARTIFACT_PATH env var.
    """
    load_secrets()
    from xg.pa.analyze import analyze
    from xg.core.config import load_config

    config_path = _xg_config_path("default.toml")
    cfg = load_config(config_path)
    cfg.data.connection_mode = "ducklake"

    artifact_path = config.artifact_path or os.environ.get("ANALYZE_ARTIFACT_PATH", "")
    if not artifact_path:
        from sim.infra.artifact_catalog import get_slot_artifacts
        conn = get_ducklake_connection()
        slot_artifacts = get_slot_artifacts(conn, slot=config.slot)
        entry = slot_artifacts.get(config.variant)
        if not entry:
            raise ValueError(
                f"No {config.variant} artifact in {config.slot} slot. "
                f"Register and promote an artifact first, or set artifact_path in config."
            )
        artifact_path = entry["s3_path"]
        context.log.info(f"Resolved from registry: {entry['artifact_id']} ({config.slot})")

    context.log.info(f"Analyzing features for: {artifact_path}")

    report = analyze(artifact_path, cfg, n_shap_samples=config.n_shap_samples)

    # Log summary
    top_gain = list(report["importance"]["gain"].items())[:10]
    context.log.info("Top 10 features by gain:")
    for feat, val in top_gain:
        context.log.info(f"  {feat}: {val:.4f}")

    context.log.info("Block importance (gain):")
    for block, val in report["importance"]["by_block"]["gain"].items():
        context.log.info(f"  {block}: {val:.4f}")

    context.log.info("Top 10 features by SHAP:")
    for feat, val in list(report["shap"]["mean_abs"].items())[:10]:
        context.log.info(f"  {feat}: {val:.6f}")

    n_high = len(report["correlation"]["high_pairs"])
    context.log.info(f"High-correlation pairs (|r| >= 0.9): {n_high}")


@asset(group_name="training", deps=[feat_mlb__vectors])
def select_features(context: AssetExecutionContext, config: SelectFeaturesConfig):
    """Three-stage feature selection: mRMR → Boruta-SHAP → ShapRFECV.

    Produces live and sim feature manifests. Registers the manifest as
    artifact_type='feature_manifest' in the registry.
    """
    load_secrets()
    from xg.core.config import load_config
    from xg.core.manifest import derive_sim_manifest, save_manifest
    from xg.pa.select import run_selection

    config_path = _xg_config_path("select.toml")
    cfg = load_config(config_path)
    cfg.data.connection_mode = "ducklake"
    cfg.output.dir = f"{S3_ARTIFACTS}/manifests"

    # Apply launchpad overrides (0 = keep TOML default)
    if config.mrmr_n_select:
        cfg.selection.mrmr_n_select = config.mrmr_n_select
    if config.boruta_n_iterations:
        cfg.selection.boruta_n_iterations = config.boruta_n_iterations
    if config.rfecv_min_features:
        cfg.selection.rfecv_min_features = config.rfecv_min_features
    if config.target:
        cfg.selection.target = config.target

    context.log.info("Starting feature selection")
    manifest = run_selection(cfg)

    # Save both manifests
    run_dir = f"{S3_ARTIFACTS}/manifests/{manifest.run_id}"
    save_manifest(manifest, f"{run_dir}/live_manifest.json")
    sim_manifest = derive_sim_manifest(manifest)
    save_manifest(sim_manifest, f"{run_dir}/sim_manifest.json")

    # Register in artifact registry (slot assignment is manual via the UI)
    from sim.infra.artifact_catalog import register_artifact
    conn = get_ducklake_connection()
    try:
        register_artifact(
            conn, "feature_manifest", manifest.run_id,
            run_dir, {
                "n_features": len(manifest.features),
                "n_sim_features": len(manifest.sim_features),
                "method": manifest.method,
                "features": manifest.features,
                "sim_features": manifest.sim_features,
            },
        )
        context.log.info(f"Registered feature_manifest: {manifest.run_id}")
    finally:
        conn.close()

    # Log summary
    for stage, meta in manifest.stage_results.items():
        context.log.info(f"  {stage}: {meta.get('input_count', '?')} → {meta.get('output_count', meta.get('confirmed_count', '?'))}")
    context.log.info(
        f"Selection complete: {len(manifest.features)} live, "
        f"{len(sim_manifest.features)} sim features"
    )
    context.log.info(f"Blocks: {', '.join(f'{k}({len(v)})' for k, v in manifest.blocks.items())}")


@asset(
    group_name="training",
    deps=[feat_mlb__vectors],
)
def train_xgboost(context: AssetExecutionContext, config: TrainXgboostConfig):
    """Train XGBoost PA outcome models (sim + live) on DuckLake feature vectors.

    Loads data once, trains two models sequentially:
      - live (all features including balls/strikes)
      - sim (drops count columns for simulation use)

    Artifacts written to s3://<S3_BUCKET>/stateball/artifacts/xgboost/{run_id}_{variant}/.
    All config params default to 0 = use TOML value. Override from Dagster launchpad.
    """
    load_secrets()
    from xg.core.config import load_config, slice_sim_data
    from xg.core.data import prepare_data
    from xg.pa.train import slice_data_to_features, run_experiment

    config_path = _xg_config_path("default.toml")
    cfg = load_config(config_path)

    cfg.data.connection_mode = "ducklake"
    cfg.output.dir = f"{S3_ARTIFACTS}/xgboost"

    # Apply run config overrides (0 = keep TOML default)
    if config.max_games:
        cfg.data.max_games = config.max_games
    if config.learning_rate:
        cfg.xgboost.learning_rate = config.learning_rate
    if config.n_estimators:
        cfg.xgboost.n_estimators = config.n_estimators
    if config.early_stopping_rounds:
        cfg.training.early_stopping_rounds = config.early_stopping_rounds
    if config.max_leaves:
        cfg.xgboost.max_leaves = config.max_leaves
    if config.subsample:
        cfg.xgboost.subsample = config.subsample
    if config.colsample_bytree:
        cfg.xgboost.colsample_bytree = config.colsample_bytree
    if config.max_bin:
        cfg.xgboost.max_bin = config.max_bin

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_default"
    context.log.info(f"Starting training run: {run_id}")

    # Resolve manifest: explicit path > registry > fallback to all features
    manifest = None
    live_manifest_dict = None
    sim_manifest_dict = None
    if config.manifest_path:
        from xg.core.manifest import derive_sim_manifest, load_manifest, manifest_to_dict
        manifest = load_manifest(config.manifest_path)
    else:
        from sim.infra.artifact_catalog import get_manifest_path
        conn = get_ducklake_connection()
        mpath = get_manifest_path(conn, slot="test")
        conn.close()
        if mpath:
            from xg.core.manifest import derive_sim_manifest, load_manifest, manifest_to_dict
            manifest = load_manifest(f"{mpath}/live_manifest.json")

    if manifest:
        from xg.core.manifest import derive_sim_manifest, manifest_to_dict
        data = prepare_data(cfg, feature_cols=manifest.features)
        sim_manifest = derive_sim_manifest(manifest)
        sim_data = slice_data_to_features(data, sim_manifest.features)
        live_manifest_dict = manifest_to_dict(manifest)
        sim_manifest_dict = manifest_to_dict(sim_manifest)
        context.log.info(
            f"Using manifest {manifest.run_id}: "
            f"{len(manifest.features)} live, {len(sim_manifest.features)} sim features"
        )
    else:
        data = prepare_data(cfg)
        sim_data = slice_sim_data(data)
        context.log.info(
            f"No manifest — using all features: "
            f"{len(data['feature_cols'])} live, {len(sim_data['feature_cols'])} sim"
        )

    # Train live model
    live_rid = f"{run_id}_live"
    context.log.info(f"Training live model: {live_rid}")
    live_results = run_experiment(
        cfg, config_path=config_path, run_id=live_rid, data=data,
        manifest_dict=live_manifest_dict,
    )
    context.log.info(
        f"Live complete. Brier: {live_results['brier_score_test']:.6f}  "
        f"Reliability: {live_results.get('brier_reliability', 'N/A')}  "
        f"Resolution: {live_results.get('brier_resolution', 'N/A')}"
    )

    # Train sim model (same config, fewer features)
    sim_rid = f"{run_id}_sim"
    context.log.info(f"Training sim model: {sim_rid}")
    sim_results = run_experiment(
        cfg, config_path=config_path, run_id=sim_rid, data=sim_data,
        manifest_dict=sim_manifest_dict,
    )
    context.log.info(
        f"Sim complete. Brier: {sim_results['brier_score_test']:.6f}  "
        f"Reliability: {sim_results.get('brier_reliability', 'N/A')}  "
        f"Resolution: {sim_results.get('brier_resolution', 'N/A')}"
    )

    # Register both artifacts
    from sim.infra.artifact_catalog import register_artifact
    conn = get_ducklake_connection()
    try:
        register_artifact(
            conn, "xgboost_live", live_rid,
            f"{S3_ARTIFACTS}/xgboost/{live_rid}", live_results,
        )
        context.log.info(f"Registered xgboost_live: {live_rid}")
        register_artifact(
            conn, "xgboost_sim", sim_rid,
            f"{S3_ARTIFACTS}/xgboost/{sim_rid}", sim_results,
        )
        context.log.info(f"Registered xgboost_sim: {sim_rid}")
    finally:
        conn.close()


def _sweep_and_retrain(context, cfg, data, study_name, label, config_path, run_id, manifest_dict=None):
    """Run one sweep + final retrain cycle. Returns (results, run_id_suffixed) or None."""
    from xg.core.io import get_s3fs
    from xg.pa.sweep import extract_frontier, final_retrain, run_sweep, select_best

    rid = f"{run_id}_{label.lower()}"
    context.log.info(f"=== {label} sweep ({len(data['feature_cols'])} features) ===")

    study = run_sweep(cfg, data, study_name=study_name)
    frontier = extract_frontier(study)
    context.log.info(f"{label} frontier: {len(frontier)} candidates")
    for i, pt in enumerate(frontier):
        context.log.info(
            f"  [{i}] reliability={pt['reliability']:.6f}  "
            f"resolution={pt['resolution']:.6f}  "
            f"inference={pt['inference_ms']:.1f}ms"
        )

    # Save frontier summary to S3
    fs = get_s3fs()
    s3_dir = f"{S3_ARTIFACTS}/xgboost/{rid}"
    with fs.open(f"{s3_dir}/frontier_summary.json", "w") as f:
        json.dump(frontier, f, indent=2)

    if not frontier:
        context.log.warning(f"{label}: no Pareto-optimal trials")
        return None

    best = select_best(frontier, max_inference_ms=cfg.sweep.max_inference_ms)
    if best is None:
        context.log.warning(f"{label}: no candidates survive inference constraint")
        return None

    context.log.info(
        f"{label}: retraining best (trial {best['trial_number']}, brier={best['brier_score']:.6f})"
    )
    results = final_retrain(
        cfg, dict(best["params"]), data,
        best_iteration=best["best_iteration"],
        config_path=config_path, run_id=rid,
        manifest_dict=manifest_dict,
    )
    context.log.info(f"{label} brier: {results['brier_score_test']:.6f}")
    return results, rid


@asset(group_name="training", deps=[feat_mlb__vectors])
def sweep_xgboost(context: AssetExecutionContext):
    """Unified sweep: shared data load, independent sweeps for sim + live models."""
    load_secrets()
    from xg.core.config import load_config, slice_sim_data
    from xg.core.data import prepare_data
    from xg.pa.train import slice_data_to_features

    config_path = _xg_config_path("sweep.toml")
    cfg = load_config(config_path)
    cfg.data.connection_mode = "ducklake"
    cfg.output.dir = f"{S3_ARTIFACTS}/xgboost"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_sweep"
    context.log.info(f"Starting unified sweep run: {run_id}")

    # Resolve manifest (same pattern as train_xgboost)
    manifest = None
    live_manifest_dict = None
    sim_manifest_dict = None
    from sim.infra.artifact_catalog import get_manifest_path
    conn = get_ducklake_connection()
    mpath = get_manifest_path(conn, slot="test")
    conn.close()
    if mpath:
        from xg.core.manifest import derive_sim_manifest, load_manifest, manifest_to_dict
        manifest = load_manifest(f"{mpath}/live_manifest.json")

    if manifest:
        from xg.core.manifest import derive_sim_manifest, manifest_to_dict
        data = prepare_data(cfg, feature_cols=manifest.features)
        sim_manifest = derive_sim_manifest(manifest)
        sim_data = slice_data_to_features(data, sim_manifest.features)
        live_manifest_dict = manifest_to_dict(manifest)
        sim_manifest_dict = manifest_to_dict(sim_manifest)
        context.log.info(
            f"Using manifest {manifest.run_id}: "
            f"{len(manifest.features)} live, {len(sim_manifest.features)} sim features"
        )
    else:
        data = prepare_data(cfg)
        sim_data = slice_sim_data(data)
        context.log.info(
            f"No manifest — using all features: "
            f"{len(data['feature_cols'])} live, {len(sim_data['feature_cols'])} sim"
        )

    # Two independent sweeps + retrains
    live_out = _sweep_and_retrain(
        context, cfg, data, "stateball_xgb_live", "Live", config_path, run_id,
        manifest_dict=live_manifest_dict,
    )
    sim_out = _sweep_and_retrain(
        context, cfg, sim_data, "stateball_xgb_sim", "Sim", config_path, run_id,
        manifest_dict=sim_manifest_dict,
    )

    # Register artifacts
    from sim.infra.artifact_catalog import register_artifact
    xgb_base = f"{S3_ARTIFACTS}/xgboost"
    conn = get_ducklake_connection()
    try:
        if live_out:
            live_results, live_rid = live_out
            register_artifact(conn, "xgboost_live", live_rid, f"{xgb_base}/{live_rid}", live_results)
            context.log.info(f"Registered xgboost_live: {live_rid}")
        if sim_out:
            sim_results, sim_rid = sim_out
            register_artifact(conn, "xgboost_sim", sim_rid, f"{xgb_base}/{sim_rid}", sim_results)
            context.log.info(f"Registered xgboost_sim: {sim_rid}")
    finally:
        conn.close()
