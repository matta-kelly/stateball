"""Three-stage feature selection: mRMR → SAGE rank → SAGE CV.

Produces a FeatureManifest recording which features were selected, why,
and how. The manifest travels with the model through training, validation,
and analysis.

Usage:
    .venv/bin/python -m xg.select                              # default config
    .venv/bin/python -m xg.select xg/configs/select.toml       # custom config
    .venv/bin/python -m xg.select select.toml --output-dir s3://bucket/path
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from xg.core.config import (
    SIM_EXCLUDED_FEATURES,
    ExperimentConfig,
    SelectionConfig,
    load_config,
)
from xg.core.data import prepare_data
from xg.core.io import log_memory
from xg.core.evaluate import brier_decomposition
from xg.core.manifest import (
    FeatureManifest,
    build_blocks,
    derive_sim_manifest,
    save_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("xg.select")

# Fast XGBoost params for selection stages — ranking quality, not production.
_FAST_XGB_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fit_fast_xgb(X: np.ndarray, y: np.ndarray, n_classes: int) -> xgb.Booster:
    """Fit a fast XGBoost model for importance estimation (not production)."""
    params = {**_FAST_XGB_PARAMS, "num_class": n_classes}
    n_est = params.pop("n_estimators")
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(params, dtrain, num_boost_round=n_est)


def _make_predict_fn(booster: xgb.Booster, n_classes: int) -> Callable:
    """Wrap a booster as a probability callable for SAGE.

    Renormalizes output to guard against floating-point precision issues
    that can cause SAGE's cross-entropy loss to reject inputs.
    """
    def predict_fn(X: np.ndarray) -> np.ndarray:
        probs = booster.predict(xgb.DMatrix(X))
        probs = probs.reshape(-1, n_classes)
        probs = np.clip(probs, 1e-12, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
    return predict_fn


# ---------------------------------------------------------------------------
# Stage 1: mRMR (filter)
# ---------------------------------------------------------------------------


def _run_mrmr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_select: int,
) -> tuple[list[str], dict]:
    """Minimum Redundancy Maximum Relevance — filter stage.

    Ranks features by mutual information with target, penalized by
    redundancy with already-selected features. No model fitting needed.
    """
    from mrmr import mrmr_classif

    t0 = time.perf_counter()

    # mRMR requires DataFrame input
    df = pd.DataFrame(X_train, columns=feature_names)
    y_series = pd.Series(y_train, name="target")

    n_select = min(n_select, len(feature_names))
    selected = mrmr_classif(X=df, y=y_series, K=n_select, show_progress=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        "mRMR: %d → %d features (%.1fs)",
        len(feature_names), len(selected), elapsed,
    )

    meta = {
        "input_count": len(feature_names),
        "output_count": len(selected),
        "time_seconds": round(elapsed, 1),
    }
    return selected, meta


# ---------------------------------------------------------------------------
# Stage 2: SAGE rank (replaces Boruta-SHAP)
# ---------------------------------------------------------------------------


def _run_sage_rank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_classes: int,
    cfg: SelectionConfig,
) -> tuple[list[str], dict]:
    """SAGE importance ranking — replaces Boruta-SHAP.

    Fits one fast XGBoost, computes SAGE importance on an eval subsample,
    returns top cfg.sage_n_select features sorted by SAGE value descending.

    SAGE measures loss degradation when each feature is removed. Unlike
    Boruta's max-shadow comparison, features are not penalized by other
    features' dominance — each contributes its own marginal loss impact.
    Negative values mean the feature adds noise; those are ranked last.
    """
    from xg.pa.sage import compute_sage_importance

    t0 = time.perf_counter()
    rng = np.random.default_rng(42)

    n_eval = min(cfg.sage_n_eval, len(X_train))
    n_bg = min(cfg.sage_background_n, len(X_train))

    eval_idx = rng.choice(len(X_train), n_eval, replace=False)
    bg_idx = rng.choice(len(X_train), n_bg, replace=False)

    booster = _fit_fast_xgb(X_train, y_train, n_classes)
    predict_fn = _make_predict_fn(booster, n_classes)

    values, stds = compute_sage_importance(
        predict_fn,
        X_train[eval_idx],
        y_train[eval_idx],
        X_train[bg_idx],
        feature_names,
        n_permutations=None,
        thresh=0.025,
    )

    # Sort descending — features with negative SAGE (noise) rank last
    ranked = sorted(feature_names, key=lambda f: values[f], reverse=True)
    n_select = min(cfg.sage_n_select, len(ranked))
    selected = ranked[:n_select]

    elapsed = time.perf_counter() - t0
    logger.info(
        "SAGE rank: %d → %d features (top: %s, %.1fs)",
        len(feature_names), n_select, selected[:3], elapsed,
    )

    meta = {
        "input_count": len(feature_names),
        "output_count": n_select,
        "importance": {f: float(values[f]) for f in feature_names},
        "std": {f: float(stds[f]) for f in feature_names},
        "time_seconds": round(elapsed, 1),
    }
    return selected, meta


# ---------------------------------------------------------------------------
# Stage 3: SAGE CV (replaces ShapRFECV)
# ---------------------------------------------------------------------------


def _run_sage_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_classes: int,
    cfg: SelectionConfig,
) -> tuple[list[str], dict]:
    """SAGE-ranked subset CV — replaces ShapRFECV.

    Re-computes SAGE on the stage-2 feature set to produce a fresh ranking
    on the smaller set, then evaluates every prefix size from cv_min_features
    to len(feature_names) via K-fold CV Brier score. Applies the
    one-standard-error rule to select the most parsimonious set whose Brier
    is within 1 std of the minimum.
    """
    from xg.pa.sage import compute_sage_importance

    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    classes = list(range(n_classes))

    # Fresh SAGE on the smaller feature set for stable ranking
    n_eval = min(cfg.sage_n_eval, len(X_train))
    n_bg = min(cfg.sage_background_n, len(X_train))

    eval_idx = rng.choice(len(X_train), n_eval, replace=False)
    bg_idx = rng.choice(len(X_train), n_bg, replace=False)

    booster = _fit_fast_xgb(X_train, y_train, n_classes)
    predict_fn = _make_predict_fn(booster, n_classes)

    values, _ = compute_sage_importance(
        predict_fn,
        X_train[eval_idx],
        y_train[eval_idx],
        X_train[bg_idx],
        feature_names,
    )

    # Re-rank on this smaller set
    ranked = sorted(feature_names, key=lambda f: values[f], reverse=True)

    # CV Brier for each prefix size: cv_min_features → len(features)
    sizes = list(range(cfg.cv_min_features, len(ranked) + 1))
    skf = StratifiedKFold(n_splits=cfg.cv_n_folds, shuffle=True, random_state=42)

    brier_by_size: list[tuple[int, float, float]] = []  # (n, mean_brier, std_brier)

    for n in sizes:
        subset = ranked[:n]
        feat_idx = [feature_names.index(f) for f in subset]
        X_sub = X_train[:, feat_idx]

        fold_briers: list[float] = []
        for train_idx, val_idx in skf.split(X_sub, y_train):
            b = _fit_fast_xgb(X_sub[train_idx], y_train[train_idx], n_classes)
            preds = _make_predict_fn(b, n_classes)(X_sub[val_idx])
            decomp = brier_decomposition(y_train[val_idx], preds, classes)
            fold_briers.append(decomp["brier_score"])

        brier_by_size.append((n, float(np.mean(fold_briers)), float(np.std(fold_briers))))

        if n % 5 == 0 or n == sizes[0] or n == sizes[-1]:
            logger.info(
                "  SAGE CV n=%d: brier=%.6f ± %.6f",
                n, brier_by_size[-1][1], brier_by_size[-1][2],
            )

    # One-standard-error rule: smallest set within 1 std of minimum Brier
    brier_means = [b[1] for b in brier_by_size]
    best_idx = int(np.argmin(brier_means))
    threshold = brier_means[best_idx] + brier_by_size[best_idx][2]
    optimal_idx = next(i for i, (_, b, _) in enumerate(brier_by_size) if b <= threshold)

    optimal_n = brier_by_size[optimal_idx][0]
    optimal_features = ranked[:optimal_n]

    elapsed = time.perf_counter() - t0
    logger.info(
        "SAGE CV: %d → %d features (best brier=%.6f at n=%d, 1-SE → n=%d, %.1fs)",
        len(feature_names), optimal_n,
        brier_by_size[best_idx][1], brier_by_size[best_idx][0],
        optimal_n, elapsed,
    )

    meta = {
        "input_count": len(feature_names),
        "output_count": optimal_n,
        "best_brier": brier_by_size[best_idx][1],
        "best_n_features": brier_by_size[best_idx][0],
        "optimal_n_features": optimal_n,
        "brier_per_n": [
            {"n": n, "mean_brier": mb, "std_brier": sb}
            for n, mb, sb in brier_by_size
        ],
        "time_seconds": round(elapsed, 1),
    }
    return optimal_features, meta


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_selection(
    cfg: ExperimentConfig,
    data: dict | None = None,
) -> FeatureManifest:
    """Run three-stage feature selection. Returns live manifest."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_select"
    logger.info("Feature selection run: %s", run_id)

    if data is None:
        data = prepare_data(cfg)
    log_memory("selection: data loaded")

    all_features = data["feature_cols"]
    y_train = data["y_train"]
    n_classes = data["n_classes"]
    target = cfg.selection.target  # "live" | "sim"
    logger.info("Starting with %d features (target=%s)", len(all_features), target)

    # Pre-filter SIM_EXCLUDED_FEATURES when target=="sim" so all three stages
    # find the best features in a count-free world.
    if target == "sim":
        sim_excluded_present = [f for f in all_features if f in SIM_EXCLUDED_FEATURES]
        selection_cols = [f for f in all_features if f not in SIM_EXCLUDED_FEATURES]
        keep_idx = [all_features.index(f) for f in selection_cols]
        X_train = data["X_train"][:, keep_idx]
        logger.info(
            "target=sim: pre-filtered %d SIM_EXCLUDED features, running selection on %d",
            len(sim_excluded_present), len(selection_cols),
        )
    else:
        sim_excluded_present = []
        selection_cols = all_features
        X_train = data["X_train"]

    # --- Stage 1: mRMR ---
    logger.info("=== Stage 1: mRMR ===")
    mrmr_features, mrmr_meta = _run_mrmr(
        X_train, y_train, selection_cols, cfg.selection.mrmr_n_select,
    )
    log_memory("selection: mRMR done")

    mrmr_idx = [selection_cols.index(f) for f in mrmr_features]
    X_stage2 = X_train[:, mrmr_idx]

    # --- Stage 2: SAGE rank ---
    logger.info("=== Stage 2: SAGE rank ===")
    sage_features, sage_rank_meta = _run_sage_rank(
        X_stage2, y_train, mrmr_features, n_classes, cfg.selection,
    )
    log_memory("selection: SAGE rank done")

    sage_idx = [mrmr_features.index(f) for f in sage_features]
    X_stage3 = X_stage2[:, sage_idx]

    # --- Stage 3: SAGE CV ---
    logger.info("=== Stage 3: SAGE CV ===")
    final_features, sage_cv_meta = _run_sage_cv(
        X_stage3, y_train, sage_features, n_classes, cfg.selection,
    )
    log_memory("selection: SAGE CV done")

    # Build dropped dict with reasons
    dropped: dict[str, dict] = {}
    mrmr_set = set(mrmr_features)
    sage_set = set(sage_features)
    final_set = set(final_features)

    for f in sim_excluded_present:
        dropped[f] = {"reason": "sim_target_excluded", "stage": 0}

    for f in selection_cols:
        if f in final_set:
            continue
        if f not in mrmr_set:
            dropped[f] = {"reason": "mrmr_filtered", "stage": 1}
        elif f not in sage_set:
            sage_val = sage_rank_meta["importance"].get(f)
            dropped[f] = {"reason": "sage_ranked_out", "stage": 2, "sage_value": sage_val}
        else:
            dropped[f] = {"reason": "cv_eliminated", "stage": 3}

    # Build manifest — sim is primary when target=="sim"
    if target == "sim":
        sim_features = final_features
        live_features = final_features + sim_excluded_present
        sim_excluded_list = sorted(sim_excluded_present)
    else:
        live_features = final_features
        sim_features = [f for f in final_features if f not in SIM_EXCLUDED_FEATURES]
        sim_excluded_list = sorted(SIM_EXCLUDED_FEATURES)

    blocks = build_blocks(live_features)

    manifest = FeatureManifest(
        run_id=run_id,
        method="mrmr_sage_cv",
        source_feature_count=len(all_features),
        warehouse_columns=all_features,
        features=live_features,
        blocks=blocks,
        dropped=dropped,
        stage_results={
            "mrmr": mrmr_meta,
            "sage_rank": sage_rank_meta,
            "sage_cv": sage_cv_meta,
        },
        sim_excluded=sim_excluded_list,
        sim_features=sim_features,
        metadata={
            "n_samples": len(X_train),
            "n_classes": n_classes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        target=target,
    )

    output_dir = cfg.output.dir
    run_dir = f"{output_dir.rstrip('/')}/{run_id}"
    save_manifest(manifest, f"{run_dir}/live_manifest.json")

    sim_manifest = derive_sim_manifest(manifest)
    save_manifest(sim_manifest, f"{run_dir}/sim_manifest.json")

    logger.info(
        "Selection complete (target=%s): %d → %d live features, %d sim features",
        target, len(all_features), len(live_features), len(sim_features),
    )
    logger.info("Blocks: %s", {k: len(v) for k, v in blocks.items()})

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("xg").setLevel(logging.DEBUG)

    args = sys.argv[1:]

    output_dir = None
    if "--output-dir" in args:
        idx = args.index("--output-dir")
        output_dir = args[idx + 1]
        args = [a for a in args if a not in ("--output-dir", output_dir)]

    config_path = (
        Path(args[0])
        if args
        else Path(__file__).parent.parent / "configs" / "select.toml"
    )
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    logger.info("Config: %s", config_path)
    cfg = load_config(config_path)

    if output_dir:
        cfg.output.dir = output_dir

    manifest = run_selection(cfg)
    logger.info("Final features (%d): %s", len(manifest.features), manifest.features)
