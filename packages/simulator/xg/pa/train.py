"""
XGBoost training pipeline for PA outcome probability estimation.

Config-driven. All experiment parameters live in TOML files under
xg/configs/. No hardcoded decisions in this module.

Usage:
    .venv/bin/python -m xg.train                          # default config
    .venv/bin/python -m xg.train xg/configs/custom.toml   # custom config
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from xg.core.config import CalibrationConfig, ExperimentConfig, TrainingConfig, load_config, slice_sim_data
from xg.core.data import prepare_data
from xg.core.io import get_s3fs, log_memory
from xg.core.evaluate import full_evaluate, plot_calibration_curves

from xg.core.io import REPO_ROOT  # noqa: E402

logger = logging.getLogger("xg.train")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

class _LogProgress(xgb.callback.TrainingCallback):
    """Route XGBoost eval metrics through Python logging so Dagster captures them."""

    def __init__(self, period: int = 50):
        super().__init__()
        self.period = period

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period == 0:
            parts = [f"[{epoch}]"]
            for dataset, metrics in evals_log.items():
                for metric, values in metrics.items():
                    parts.append(f"{dataset}-{metric}:{values[-1]:.5f}")
            logger.info("  ".join(parts))
        return False


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    xgb_params: dict,
    training_cfg: TrainingConfig,
) -> XGBClassifier:
    """Fit XGBClassifier with early stopping on a validation slice."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=training_cfg.early_stopping_validation_fraction,
        random_state=xgb_params.get("random_state", 42),
        stratify=y_train,
    )

    model = XGBClassifier(
        early_stopping_rounds=training_cfg.early_stopping_rounds,
        callbacks=[_LogProgress(period=50)],
        **xgb_params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("Best iteration: %d, best score: %.6f", model.best_iteration, model.best_score)
    return model


# ---------------------------------------------------------------------------
# Calibrate
# ---------------------------------------------------------------------------

def calibrate(
    model: XGBClassifier,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    cal_cfg: CalibrationConfig,
) -> CalibratedClassifierCV:
    """Fit post-hoc calibration on held-out calibration set."""
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(model),
        method=cal_cfg.method,
        cv=cal_cfg.cv,
    )
    calibrated.fit(X_calib, y_calib)
    logger.info("Calibration complete (%s, cv=%d)", cal_cfg.method, cal_cfg.cv)
    return calibrated


# ---------------------------------------------------------------------------
# Export decomposed artifacts
# ---------------------------------------------------------------------------

def _export_decomposed(model, feature_names, write_fn):
    """Extract XGBoost booster + isotonic calibration tables + ONNX from CalibratedClassifierCV.

    Args:
        model: A fitted CalibratedClassifierCV wrapping FrozenEstimator(XGBClassifier).
        feature_names: Ordered list of feature names (needed for ONNX input shape).
        write_fn: Callable(filename, data_bytes_or_dict) that writes to the run dir.
                  For JSON dicts, pass the dict. For binary, pass bytes.
    """
    try:
        cc = model.calibrated_classifiers_[0]
        inner_clf = cc.estimator.estimator  # FrozenEstimator → XGBClassifier
        booster = inner_clf.get_booster()

        # Save booster in universal binary JSON
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
            booster.save_model(tmp.name)
            with open(tmp.name, "rb") as f:
                write_fn("model.ubj", f.read())
            os.unlink(tmp.name)

        # Extract isotonic calibration tables
        cal_tables = {}
        for i, cal in enumerate(cc.calibrators):
            cal_tables[str(i)] = {
                "x": cal.X_thresholds_.tolist(),
                "y": cal.y_thresholds_.tolist(),
            }
        write_fn("calibration.json", cal_tables)

        logger.info("Exported decomposed artifacts: model.ubj + calibration.json (%d classes)", len(cal_tables))

        # Export ONNX model (optional — requires onnxmltools)
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_types = [("X", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_xgboost(booster, initial_types=initial_types, target_opset=15)
            write_fn("model.onnx", onnx_model.SerializeToString())
            logger.info("Exported ONNX model: model.onnx (%d features)", len(feature_names))
        except ImportError:
            logger.info("onnxmltools not installed — skipping ONNX export")
        except Exception:
            logger.warning("ONNX export failed", exc_info=True)

    except Exception:
        logger.warning("Could not export decomposed artifacts — saving joblib only", exc_info=True)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_experiment(
    model,
    label_encoder: LabelEncoder,
    cfg: ExperimentConfig,
    results: dict,
    config_path: Path | None,
    feature_names: list[str],
    run_id: str | None = None,
    manifest_dict: dict | None = None,
):
    """Save model + frozen config copy + results. Supports local paths and s3://."""
    output_dir = cfg.output.dir
    is_s3 = output_dir.startswith("s3://")

    if run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = cfg.output.name or (config_path.stem if config_path else "experiment")
        run_id = f"{ts}_{name}"

    # Attach metadata to results
    results["run_id"] = run_id
    results["timestamp"] = datetime.now().isoformat()
    results["n_features"] = len(feature_names)
    results["feature_names"] = feature_names
    results["xgboost_version"] = xgb.__version__
    results["sklearn_version"] = sklearn.__version__
    if manifest_dict:
        results["manifest_run_id"] = manifest_dict.get("run_id")

    if is_s3:
        return _save_experiment_s3(
            model, label_encoder, results, config_path, output_dir, run_id,
            manifest_dict=manifest_dict,
        )
    else:
        return _save_experiment_local(
            model, label_encoder, results, config_path, output_dir, run_id,
            manifest_dict=manifest_dict,
        )


def _save_experiment_local(model, label_encoder, results, config_path, output_dir, run_id, manifest_dict=None):
    """Save experiment artifacts to local filesystem."""
    out = Path(output_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    joblib.dump({"model": model, "label_encoder": label_encoder}, model_path)

    # Decomposed artifacts (booster + calibration tables)
    def _write_local(filename, data):
        path = run_dir / filename
        if isinstance(data, dict):
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, "wb") as f:
                f.write(data)
    _export_decomposed(model, results["feature_names"], _write_local)

    if manifest_dict is not None:
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest_dict, f, indent=2)

    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "config.toml")

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    cal_path = None
    if "calibration_by_class" in results:
        try:
            cal_path = run_dir / "calibration_curves.png"
            plot_calibration_curves(results["calibration_by_class"], save_path=str(cal_path))
        except ImportError:
            pass

    logger.info("Experiment saved: %s", run_dir)
    return run_dir


def _save_experiment_s3(model, label_encoder, results, config_path, output_dir, run_id, manifest_dict=None):
    """Save experiment artifacts to S3 via s3fs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = get_s3fs()
    run_dir = f"{output_dir.rstrip('/')}/{run_id}"

    # Model artifact
    with fs.open(f"{run_dir}/model.joblib", "wb") as f:
        joblib.dump({"model": model, "label_encoder": label_encoder}, f)

    # Decomposed artifacts (booster + calibration tables)
    def _write_s3(filename, data):
        if isinstance(data, dict):
            with fs.open(f"{run_dir}/{filename}", "w") as f:
                json.dump(data, f, indent=2)
        else:
            with fs.open(f"{run_dir}/{filename}", "wb") as f:
                f.write(data)
    _export_decomposed(model, results["feature_names"], _write_s3)

    if manifest_dict is not None:
        with fs.open(f"{run_dir}/manifest.json", "w") as f:
            json.dump(manifest_dict, f, indent=2)

    # Frozen config copy
    if config_path and Path(config_path).exists():
        with open(config_path, "rb") as src, fs.open(f"{run_dir}/config.toml", "wb") as dst:
            shutil.copyfileobj(src, dst)

    # Results JSON
    with fs.open(f"{run_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Calibration curves
    if "calibration_by_class" in results:
        try:
            fig = plot_calibration_curves(results["calibration_by_class"], return_fig=True)
            if fig:
                with fs.open(f"{run_dir}/calibration_curves.png", "wb") as f:
                    fig.savefig(f, dpi=150, bbox_inches="tight", format="png")
                plt.close(fig)
        except ImportError:
            pass

    logger.info("Experiment saved: %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def slice_data_to_features(data: dict, target_features: list[str]) -> dict:
    """Slice a data dict to a specific feature list (generalizes slice_sim_data)."""
    keep_indices = [i for i, c in enumerate(data["feature_cols"]) if c in target_features]
    return {
        **data,
        "X_train": data["X_train"][:, keep_indices],
        "X_calib": data["X_calib"][:, keep_indices],
        "X_test": data["X_test"][:, keep_indices],
        "feature_cols": [data["feature_cols"][i] for i in keep_indices],
    }


def run_experiment(
    cfg: ExperimentConfig,
    config_path: Path | None = None,
    run_id: str | None = None,
    data: dict | None = None,
    manifest_dict: dict | None = None,
) -> dict:
    """Execute full training pipeline: train → calibrate → evaluate → save.

    Args:
        data: Pre-loaded data dict from prepare_data(). If None, loads from
              config. Pass this to avoid redundant data loading when training
              multiple model variants (e.g. sim + live).
        manifest_dict: Serialized FeatureManifest to embed in artifact dir.
    """
    if data is None:
        data = prepare_data(cfg)
    log_memory("run_experiment start")

    xgb_params = asdict(cfg.xgboost)
    model = train(data["X_train"], data["y_train"], xgb_params, cfg.training)
    log_memory("run_experiment train done")

    calibrated = calibrate(model, data["X_calib"], data["y_calib"], cfg.calibration)

    results = full_evaluate(
        calibrated, data["X_test"], data["y_test"], data["class_names"],
        model_bundle=calibrated,
    )
    log_memory("run_experiment eval done")

    # Attach metadata
    results["classes"] = data["class_names"]
    results["best_iteration"] = model.best_iteration
    results["best_score"] = float(model.best_score)
    results["sampling_enabled"] = cfg.sampling.enabled
    results["total_pas"] = data["total_pas"]
    results["train_pas"] = data["train_pas"]
    results["calibrate_pas"] = data["calibrate_pas"]
    results["test_pas"] = data["test_pas"]
    results["train_games"] = data["train_games"]
    results["date_range"] = data["date_range"]
    results["seasons"] = data["seasons"]

    if config_path or run_id:
        save_experiment(
            calibrated, data["le"], cfg, results, config_path,
            data["feature_cols"], run_id=run_id, manifest_dict=manifest_dict,
        )

    return results


def run_experiment_pair(
    cfg: ExperimentConfig,
    config_path: Path | None = None,
    run_id_base: str | None = None,
    manifest=None,
) -> tuple[dict, dict]:
    """Train both sim and live model variants from one data load.

    Loads data once (live feature superset), slices for sim, trains both.
    Produces two saved runs: {run_id_base}_live and {run_id_base}_sim.

    Args:
        manifest: FeatureManifest object. When provided, uses manifest.features
            for live data and manifest.sim_features for sim data instead of
            resolve_features() + slice_sim_data().
    """
    if manifest is not None:
        from xg.core.manifest import derive_sim_manifest, manifest_to_dict
        data = prepare_data(cfg, feature_cols=manifest.features)
        sim_data = slice_data_to_features(data, manifest.sim_features)
        live_manifest_dict = manifest_to_dict(manifest)
        sim_manifest = derive_sim_manifest(manifest)
        sim_manifest_dict = manifest_to_dict(sim_manifest)
    else:
        data = prepare_data(cfg)
        sim_data = slice_sim_data(data)
        live_manifest_dict = None  # noqa: F841
        sim_manifest_dict = None  # noqa: F841

    logger.info(
        "Live features: %d, Sim features: %d (dropped %d sim-excluded)",
        len(data["feature_cols"]),
        len(sim_data["feature_cols"]),
        len(data["feature_cols"]) - len(sim_data["feature_cols"]),
    )

    if run_id_base is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = cfg.output.name or (config_path.stem if config_path else "default")
        run_id_base = f"{ts}_{name}"

    logger.info("--- Training live model ---")
    live_results = run_experiment(
        cfg, config_path,
        run_id=f"{run_id_base}_live",
        data=data,
        manifest_dict=live_manifest_dict,
    )

    logger.info("--- Training sim model ---")
    sim_results = run_experiment(
        cfg, config_path,
        run_id=f"{run_id_base}_sim",
        data=sim_data,
        manifest_dict=sim_manifest_dict,
    )

    return live_results, sim_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path):
    """Load config and train both sim + live model variants."""
    logger.info("Config: %s", config_path)
    cfg = load_config(config_path)
    run_experiment_pair(cfg, config_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("xg").setLevel(logging.DEBUG)

    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent.parent / "configs" / "default.toml"
    )
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)
    main(config_path)
