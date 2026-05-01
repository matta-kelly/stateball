"""
Multi-objective hyperparameter sweep for the XGBoost PA outcome model.

Replaces the single-objective tune.py. Maps the Pareto frontier across
calibration (reliability), discrimination (resolution), and inference speed
using Optuna's NSGA-II sampler.

Usage:
    .venv/bin/python -m xg.sweep                          # sweep only
    .venv/bin/python -m xg.sweep --final                  # sweep + retrain best
    .venv/bin/python -m xg.sweep xg/configs/sweep.toml    # custom config
"""

from __future__ import annotations

from dataclasses import replace
import gc
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from xg.core.config import ExperimentConfig, load_config
from xg.core.data import prepare_data
from xg.core.io import log_memory
from xg.core.evaluate import brier_decomposition, inference_timing
from xg.pa.train import calibrate, save_experiment

from xg.core.io import REPO_ROOT  # noqa: E402

logger = logging.getLogger("xg.sweep")


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def make_objective(
    cfg: ExperimentConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    feature_cols: list[str],
):
    """Return a multi-objective Optuna objective closure.

    Each trial returns (reliability, -resolution, inference_ms) — all minimized.

    Calib is sub-split into cal_fit (for isotonic fitting) and cal_eval (for
    metric evaluation) to prevent data leakage. The split is deterministic and
    shared across all trials for fair comparison.
    """
    s = cfg.sweep

    # Sub-split calib: fit isotonic on cal_fit, evaluate metrics on cal_eval
    X_cal_fit, X_cal_eval, y_cal_fit, y_cal_eval = train_test_split(
        X_calib, y_calib,
        test_size=s.sweep_eval_fraction,
        random_state=cfg.split.seed + 1,
        stratify=y_calib,
    )
    logger.info(
        "Sweep calib sub-split: fit=%d  eval=%d  (%.0f%% held out)",
        len(y_cal_fit), len(y_cal_eval), s.sweep_eval_fraction * 100,
    )

    # Subsample training data for faster sweep trials (ranking only).
    # Evaluation always uses the full cal/test sets for stable metrics.
    if s.sweep_sample_fraction < 1.0:
        X_train_sweep, _, y_train_sweep, _ = train_test_split(
            X_train, y_train,
            train_size=s.sweep_sample_fraction,
            random_state=cfg.split.seed + 2,
            stratify=y_train,
        )
        logger.info(
            "Sweep training subsample: %d rows (%.0f%% of %d)",
            len(y_train_sweep), s.sweep_sample_fraction * 100, len(y_train),
        )
    else:
        X_train_sweep = X_train
        y_train_sweep = y_train

    # Sweep calibration uses fewer CV folds — cal_fit is smaller than full
    # calib, so rare classes may not survive cv=5. This is for trial ranking,
    # not the final model (which calibrates on full calib with cfg.calibration).
    sweep_cal_cfg = replace(cfg.calibration, cv=2)

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        params = {
            "objective": cfg.xgboost.objective,
            "eval_metric": cfg.xgboost.eval_metric,
            "n_estimators": trial.suggest_int("n_estimators", s.n_estimators_range[0], s.n_estimators_range[1]),
            "max_delta_step": cfg.xgboost.max_delta_step,
            "random_state": cfg.xgboost.random_state,
            "n_jobs": cfg.xgboost.n_jobs,
            "tree_method": cfg.xgboost.tree_method,
            "min_child_weight": trial.suggest_int("min_child_weight", s.min_child_weight_range[0], s.min_child_weight_range[1]),
            "learning_rate": trial.suggest_float("learning_rate", s.learning_rate_range[0], s.learning_rate_range[1], log=True),
            "subsample": trial.suggest_float("subsample", s.subsample_range[0], s.subsample_range[1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", s.colsample_bytree_range[0], s.colsample_bytree_range[1]),
            "reg_lambda": trial.suggest_float("reg_lambda", s.reg_lambda_range[0], s.reg_lambda_range[1], log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", s.reg_alpha_range[0], s.reg_alpha_range[1], log=True),
        }

        # Tree structure: lossguide sweeps max_leaves, depthwise sweeps max_depth
        if cfg.xgboost.grow_policy == "lossguide":
            params["grow_policy"] = "lossguide"
            params["max_leaves"] = trial.suggest_int("max_leaves", s.max_leaves_range[0], s.max_leaves_range[1])
            params["max_depth"] = 0
            params["max_bin"] = cfg.xgboost.max_bin
        else:
            params["max_depth"] = trial.suggest_int("max_depth", s.max_depth_range[0], s.max_depth_range[1])

        log_memory(f"trial {trial.number + 1} start")
        logger.info("--- Trial %d ---", trial.number + 1)
        complexity_label = f"max_leaves={params.get('max_leaves', '-')}" if cfg.xgboost.grow_policy == "lossguide" else f"max_depth={params['max_depth']}"
        logger.debug(
            "  n_est=%d  %s  lr=%.4f  subsample=%.2f  colsample=%.2f",
            params["n_estimators"], complexity_label, params["learning_rate"],
            params["subsample"], params["colsample_bytree"],
        )

        # Early stopping split from (possibly subsampled) training data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_sweep, y_train_sweep,
            test_size=cfg.training.early_stopping_validation_fraction,
            random_state=params["random_state"],
            stratify=y_train_sweep,
        )

        # No Optuna pruning — Trial.report() is not supported for multi-objective.
        # XGBoost early stopping handles cutting short bad configs.
        model = XGBClassifier(
            early_stopping_rounds=cfg.training.early_stopping_rounds,
            **params,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

        # Calibrate on cal_fit, evaluate on held-out cal_eval (no leakage)
        calibrated = calibrate(model, X_cal_fit, y_cal_fit, sweep_cal_cfg)
        y_proba = calibrated.predict_proba(X_cal_eval)
        int_classes = list(range(y_proba.shape[1]))

        # Brier decomposition on held-out eval set
        decomp = brier_decomposition(y_cal_eval, y_proba, int_classes)
        reliability = decomp["reliability"]
        resolution = decomp["resolution"]

        # Inference timing
        timing = inference_timing(calibrated, X_cal_eval[:2000])
        inf_ms = timing["ms_per_batch"]

        # Store for later retrieval
        trial.set_user_attr("best_iteration", model.best_iteration)
        trial.set_user_attr("reliability", reliability)
        trial.set_user_attr("resolution", resolution)
        trial.set_user_attr("inference_ms", inf_ms)
        trial.set_user_attr("brier_score", decomp["brier_score"])

        logger.info(
            "  reliability=%.6f  resolution=%.6f  inference=%.1fms  brier=%.6f",
            reliability, resolution, inf_ms, decomp["brier_score"],
        )

        del model, calibrated
        gc.collect()
        log_memory(f"trial {trial.number + 1} done (post-gc)")

        # All minimized: reliability (lower=better), -resolution (lower=more discriminative), inference_ms (lower=faster)
        return reliability, -resolution, inf_ms

    return objective


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_sweep(
    cfg: ExperimentConfig,
    data: dict,
    study_name: str | None = None,
) -> optuna.Study:
    """Create and run a multi-objective Optuna study."""
    s = cfg.sweep

    study = optuna.create_study(
        study_name=study_name or s.study_name,
        storage=s.storage,
        directions=["minimize", "minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(),
        load_if_exists=True,
    )

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            logger.info("Trial %d/%d pruned", trial.number + 1, s.n_trials)
        else:
            logger.info(
                "Trial %d/%d done  objectives=%s",
                trial.number + 1, s.n_trials,
                [f"{v:.4f}" for v in trial.values],
            )

    study.optimize(
        make_objective(cfg, data["X_train"], data["y_train"], data["X_calib"], data["y_calib"], data["feature_cols"]),
        n_trials=s.n_trials,
        timeout=s.timeout,
        callbacks=[_log_trial],
    )

    logger.info("=" * 60)
    logger.info("Sweep complete: %d trials", len(study.trials))
    frontier = extract_frontier(study)
    logger.info("Pareto frontier: %d candidates", len(frontier))
    for i, pt in enumerate(frontier):
        logger.info(
            "  [%d] reliability=%.6f  resolution=%.6f  inference=%.1fms  (trial %d)",
            i, pt["reliability"], pt["resolution"], pt["inference_ms"], pt["trial_number"],
        )
    logger.info("=" * 60)

    return study


# ---------------------------------------------------------------------------
# Frontier extraction
# ---------------------------------------------------------------------------

def extract_frontier(study: optuna.Study) -> list[dict]:
    """Extract Pareto-optimal trials from a multi-objective study."""
    frontier = []
    for trial in study.best_trials:
        frontier.append({
            "params": trial.params,
            "reliability": trial.user_attrs["reliability"],
            "resolution": trial.user_attrs["resolution"],
            "inference_ms": trial.user_attrs["inference_ms"],
            "brier_score": trial.user_attrs["brier_score"],
            "best_iteration": trial.user_attrs["best_iteration"],
            "trial_number": trial.number,
        })
    # Sort by Brier score (best combined calibration + discrimination first)
    frontier.sort(key=lambda x: x["brier_score"])
    return frontier


# ---------------------------------------------------------------------------
# Frontier selection
# ---------------------------------------------------------------------------

def select_best(
    frontier: list[dict],
    max_inference_ms: float | None = None,
    inference_speed_weight: float = 0.003,
) -> dict | None:
    """Select best candidate from Pareto frontier.

    Strategy:
    1. Filter by hard inference constraint (if set)
    2. Score each candidate: brier + λ*log(inference_ms)
       - λ=0 recovers pure Brier selection
       - λ>0 penalizes slow models with diminishing returns
         (80ms→10ms matters a lot, 10ms→4ms barely registers)
    3. Pick lowest score

    Returns None if no candidates survive the filter.
    """
    candidates = frontier
    if max_inference_ms is not None:
        candidates = [c for c in candidates if c["inference_ms"] <= max_inference_ms]

    if not candidates:
        return None

    def _score(c):
        brier = c["brier_score"]
        inf_ms = max(c["inference_ms"], 0.1)  # floor to avoid log(0)
        return brier + inference_speed_weight * math.log(inf_ms)

    ranked = sorted(candidates, key=_score)

    if inference_speed_weight > 0:
        best = ranked[0]
        pure_brier_best = sorted(candidates, key=lambda c: c["brier_score"])[0]
        if best["trial_number"] != pure_brier_best["trial_number"]:
            logger.info(
                "Speed-aware selection: trial %d (%.1fms, brier=%.6f) over "
                "trial %d (%.1fms, brier=%.6f) — λ=%.4f",
                best["trial_number"], best["inference_ms"], best["brier_score"],
                pure_brier_best["trial_number"], pure_brier_best["inference_ms"],
                pure_brier_best["brier_score"], inference_speed_weight,
            )

    return ranked[0]


# ---------------------------------------------------------------------------
# Final retrain
# ---------------------------------------------------------------------------

def final_retrain(
    cfg: ExperimentConfig,
    trial_params: dict,
    data: dict,
    best_iteration: int,
    config_path: Path | None = None,
    run_id: str | None = None,
    manifest_dict: dict | None = None,
) -> dict:
    """Retrain with best params on full train set, calibrate, evaluate on test.

    Args:
        best_iteration: The early-stopped round count from the sweep trial.
                        This is the actual number of trees to train — not
                        n_estimators (which is the max ceiling).
    """
    from xg.core.evaluate import full_evaluate

    remaining_params = {k: v for k, v in trial_params.items() if k != "n_estimators"}

    params = {
        "objective": cfg.xgboost.objective,
        "eval_metric": cfg.xgboost.eval_metric,
        "n_estimators": best_iteration,
        "max_delta_step": cfg.xgboost.max_delta_step,
        "random_state": cfg.xgboost.random_state,
        "n_jobs": cfg.xgboost.n_jobs,
        "tree_method": cfg.xgboost.tree_method,
        "grow_policy": cfg.xgboost.grow_policy,
        "max_bin": cfg.xgboost.max_bin,
        **remaining_params,
    }

    # lossguide needs max_depth=0 explicitly (not stored in trial params)
    if cfg.xgboost.grow_policy == "lossguide" and "max_depth" not in remaining_params:
        params["max_depth"] = 0

    logger.info("Final retrain: n_estimators=%d", best_iteration)
    logger.debug("Params: %s", params)

    # Train on FULL training set (no early stopping holdout)
    model = XGBClassifier(**params)
    model.fit(data["X_train"], data["y_train"], verbose=50)
    log_memory("final_retrain fit")

    # Calibrate
    calibrated = calibrate(model, data["X_calib"], data["y_calib"], cfg.calibration)

    # Evaluate on test set
    results = full_evaluate(
        calibrated, data["X_test"], data["y_test"], data["class_names"],
        model_bundle=calibrated,
    )

    # Attach metadata
    results["best_iteration"] = best_iteration
    results["sweep_mode"] = "final_retrain"
    results["sweep_params"] = {**remaining_params, "n_estimators": best_iteration}
    results["classes"] = data["class_names"]
    results["sampling_enabled"] = cfg.sampling.enabled
    results["total_pas"] = data["total_pas"]
    results["train_pas"] = data["train_pas"]
    results["calibrate_pas"] = data["calibrate_pas"]
    results["test_pas"] = data["test_pas"]
    results["train_games"] = data["train_games"]
    results["date_range"] = data["date_range"]
    results["seasons"] = data["seasons"]

    # Save
    save_experiment(
        calibrated, data["le"], cfg, results, config_path, data["feature_cols"],
        run_id=run_id, manifest_dict=manifest_dict,
    )

    return results


# ---------------------------------------------------------------------------
# Sim data slicing (delegated to xg.config)
# ---------------------------------------------------------------------------

from xg.core.config import slice_sim_data  # noqa: E402 — re-export for backward compat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_sweep_and_select(cfg, data, study_name, label, config_path, do_final, manifest_dict=None):
    """Run sweep for one model variant, optionally final-retrain the winner."""
    logger.info("=" * 60)
    logger.info("=== %s sweep (%d features) ===", label, len(data["feature_cols"]))
    logger.info("=" * 60)

    study = run_sweep(cfg, data, study_name=study_name)
    frontier = extract_frontier(study)

    summary_path = REPO_ROOT / "xg" / f"sweep_summary_{label.lower()}.json"
    with open(summary_path, "w") as f:
        json.dump(frontier, f, indent=2)
    logger.info("%s frontier saved: %s (%d candidates)", label, summary_path, len(frontier))

    if not (do_final and frontier):
        return None

    best = select_best(
        frontier,
        max_inference_ms=cfg.sweep.max_inference_ms,
        inference_speed_weight=cfg.sweep.inference_speed_weight,
    )
    if best is None:
        logger.warning("%s: no candidates survive inference constraint", label)
        return None

    logger.info("%s: retraining best (trial %d, brier=%.6f)",
                label, best["trial_number"], best["brier_score"])
    return final_retrain(
        cfg, dict(best["params"]), data, best["best_iteration"],
        config_path=config_path, manifest_dict=manifest_dict,
    )


def main():
    args = sys.argv[1:]
    final = "--final" in args
    args = [a for a in args if a != "--final"]

    manifest_path = None
    if "--manifest" in args:
        idx = args.index("--manifest")
        manifest_path = args[idx + 1]
        args = [a for a in args if a not in ("--manifest", manifest_path)]

    config_path = (
        Path(args[0])
        if args
        else Path(__file__).parent.parent / "configs" / "sweep.toml"
    )
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    logger.info("Config: %s", config_path)
    logger.info("Mode: %s", "sweep + final retrain (sim + live)" if final else "sweep only")

    cfg = load_config(config_path)

    # Load data — manifest-driven or full feature set
    live_manifest_dict = None
    sim_manifest_dict = None
    if manifest_path:
        from xg.core.manifest import derive_sim_manifest, load_manifest, manifest_to_dict
        from xg.pa.train import slice_data_to_features
        manifest = load_manifest(manifest_path)
        data = prepare_data(cfg, feature_cols=manifest.features)
        sim_data = slice_data_to_features(data, manifest.sim_features)
        live_manifest_dict = manifest_to_dict(manifest)
        sim_manifest_dict = manifest_to_dict(derive_sim_manifest(manifest))
        logger.info("Using manifest %s", manifest.run_id)
    else:
        data = prepare_data(cfg)
        sim_data = slice_sim_data(data)

    logger.info("Live features: %d, Sim features: %d (dropped %d count cols)",
                len(data["feature_cols"]), len(sim_data["feature_cols"]),
                len(data["feature_cols"]) - len(sim_data["feature_cols"]))

    # Two independent sweeps
    _run_sweep_and_select(cfg, data, "stateball_xgb_live", "Live", config_path, final, manifest_dict=live_manifest_dict)
    _run_sweep_and_select(cfg, sim_data, "stateball_xgb_sim", "Sim", config_path, final, manifest_dict=sim_manifest_dict)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("xg").setLevel(logging.DEBUG)
    main()
