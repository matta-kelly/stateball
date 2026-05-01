"""
Validate an existing model artifact across splits and strata.

Replaces kfold_evaluate / run_cv. Instead of training K models from scratch,
loads a registered artifact and evaluates it rigorously.

Usage:
    .venv/bin/python -m xg.validate <artifact_id>
    .venv/bin/python -m xg.validate <artifact_id> --config xg/configs/default.toml
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from xg.core.config import ExperimentConfig, load_config
from xg.core.data import prepare_data
from xg.core.io import get_s3fs, log_memory
from xg.core.evaluate import bootstrap_ci, brier_decomposition, full_evaluate, stratified_calibration

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("xg.validate")


# ---------------------------------------------------------------------------
# Strata builders
# ---------------------------------------------------------------------------

def build_strata(X_test: np.ndarray, feature_cols: list[str]) -> dict[str, np.ndarray]:
    """Build boolean masks for stratified evaluation from feature columns.

    Creates strata based on game state features when available:
    - Inning groups: early (1-3), mid (4-6), late (7+)
    - Score differential: ahead (run_diff > 0), tied (0), behind (< 0)
    """
    col_idx = {name: i for i, name in enumerate(feature_cols)}
    strata = {}

    if "inning" in col_idx:
        inning = X_test[:, col_idx["inning"]]
        strata["early_inning"] = inning <= 3
        strata["mid_inning"] = (inning >= 4) & (inning <= 6)
        strata["late_inning"] = inning >= 7

    if "run_diff" in col_idx:
        rd = X_test[:, col_idx["run_diff"]]
        strata["ahead"] = rd > 0
        strata["tied"] = rd == 0
        strata["behind"] = rd < 0
        strata["close_game"] = np.abs(rd) <= 2
        strata["blowout"] = np.abs(rd) > 4

    if "times_through_order" in col_idx:
        tto = X_test[:, col_idx["times_through_order"]]
        strata["tto_1"] = tto == 1
        strata["tto_2"] = tto == 2
        strata["tto_3_plus"] = tto >= 3

    return strata


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_artifact(
    artifact_id: str,
    cfg: ExperimentConfig,
    n_splits: int = 10,
    strata_names: list[str] | None = None,
) -> dict:
    """Load an existing artifact and evaluate it across reshuffled splits + strata.

    1. Resolve artifact from registry
    2. Load model via sim.artifacts.load_model
    3. Load + prepare test data
    4. For each of N splits: reshuffle test set, run full_evaluate with strata
    5. Aggregate: mean/std of Brier components across splits
    6. Bootstrap CI on full dataset
    7. Save validation_report.json to artifact's S3 directory

    Returns the full report dict.
    """
    logger.info("Validating artifact: %s (%d splits)", artifact_id, n_splits)

    # Resolve artifact path from registry
    from sim.registry import get_artifact
    artifact = get_artifact(artifact_id)
    s3_path = artifact["s3_path"]
    logger.info("Artifact S3 path: %s", s3_path)

    # Load model
    from sim.artifacts import load_model
    model_bundle = load_model(s3_path)
    model = model_bundle.model
    logger.info("Model loaded: %d classes", len(model_bundle.class_names))

    # Try to load manifest from artifact directory
    from xg.core.manifest import try_load_artifact_manifest
    manifest = try_load_artifact_manifest(s3_path)
    if manifest:
        logger.info("Manifest loaded: %s (%d features)", manifest.run_id, len(manifest.features))
    else:
        logger.info("No manifest found (legacy artifact)")

    # Prepare test data
    data = prepare_data(cfg)
    X_test = data["X_test"]
    y_test = data["y_test"]
    class_names = data["class_names"]

    # Align data columns to model's feature list
    data_cols = data["feature_cols"]
    model_features = model_bundle.feature_names
    if model_features and data_cols != model_features:
        col_idx = [data_cols.index(f) for f in model_features]
        X_test = X_test[:, col_idx]
        dropped = set(data_cols) - set(model_features)
        logger.info(
            "Aligned data (%d cols) to model (%d features), dropped: %s",
            len(data_cols), len(model_features), sorted(dropped),
        )
        data_cols = model_features

    # Build strata (uses aligned column list)
    strata = build_strata(X_test, data_cols)
    if strata_names:
        strata = {k: v for k, v in strata.items() if k in strata_names}
    logger.info("Strata: %s", list(strata.keys()))

    # Multi-split evaluation
    int_classes = list(range(len(class_names)))
    split_results = []
    rng = np.random.RandomState(42)

    for i in range(n_splits):
        log_memory(f"split {i+1}/{n_splits}")
        # Reshuffle test indices
        idx = rng.permutation(len(y_test))
        X_shuf = X_test[idx]
        y_shuf = y_test[idx]

        result = full_evaluate(
            model, X_shuf, y_shuf, class_names,
            strata={k: v[idx] for k, v in strata.items()} if strata else None,
            model_bundle=model,
        )
        split_results.append(result)
        logger.info(
            "  Split %d/%d: brier=%.6f  reliability=%.6f  resolution=%.6f",
            i + 1, n_splits,
            result["brier_score_test"],
            result["brier_reliability"],
            result["brier_resolution"],
        )

    # Aggregate across splits
    def _agg(key):
        vals = [r[key] for r in split_results if key in r]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    report = {
        "artifact_id": artifact_id,
        "s3_path": s3_path,
        "n_splits": n_splits,
        "test_samples": len(y_test),
        "classes": class_names,
        "brier_score": _agg("brier_score_test"),
        "brier_skill": _agg("brier_skill"),
        "brier_reliability": _agg("brier_reliability"),
        "brier_resolution": _agg("brier_resolution"),
        "brier_uncertainty": _agg("brier_uncertainty"),
        "manifest": {
            "run_id": manifest.run_id,
            "method": manifest.method,
            "n_features": len(manifest.features),
            "blocks": list(manifest.blocks.keys()),
        } if manifest else None,
    }

    # Bootstrap CI on full (unshuffled) test set
    y_proba = model.predict_proba(X_test)
    ci = bootstrap_ci(y_test, y_proba, int_classes)
    report["bootstrap_ci"] = ci

    # Full decomposition on unshuffled set
    decomp = brier_decomposition(y_test, y_proba, int_classes)
    report["full_decomposition"] = decomp

    # Stratified calibration on full set
    if strata:
        strat_cal = stratified_calibration(y_test, y_proba, int_classes, strata)
        report["stratified_calibration"] = strat_cal

    # Save report to S3
    try:
        fs = get_s3fs()
        report_path = f"{s3_path.rstrip('/')}/validation_report.json"
        with fs.open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Validation report saved: %s", report_path)
    except Exception:
        logger.warning("Could not save to S3 — saving locally", exc_info=True)
        local_path = REPO_ROOT / "xg" / "experiments" / f"validate_{artifact_id}.json"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Validation report saved locally: %s", local_path)

    # Log summary
    logger.info("=" * 60)
    logger.info("Validation complete: %s", artifact_id)
    logger.info("  Brier: %.6f +/- %.6f", report["brier_score"]["mean"], report["brier_score"]["std"])
    logger.info("  Reliability: %.6f +/- %.6f", report["brier_reliability"]["mean"], report["brier_reliability"]["std"])
    logger.info("  Resolution: %.6f +/- %.6f", report["brier_resolution"]["mean"], report["brier_resolution"]["std"])
    logger.info("  Bootstrap 95%% CI: [%.6f, %.6f]", ci["ci_lower"], ci["ci_upper"])
    if strata and "stratified_calibration" in report:
        for name, vals in report["stratified_calibration"].items():
            logger.info("  %s (n=%d): reliability=%.6f", name, vals["n"], vals["reliability"])
    logger.info("=" * 60)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("xg").setLevel(logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python -m xg.validate <artifact_id> [--config path/to/config.toml]")
        sys.exit(1)

    artifact_id = sys.argv[1]

    config_path = Path(__file__).parent.parent / "configs" / "default.toml"
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = Path(sys.argv[idx + 1])

    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    cfg = load_config(config_path)
    validate_artifact(artifact_id, cfg)
