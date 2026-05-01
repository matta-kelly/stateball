"""Train pitcher exit model: XGBoost binary classifier + isotonic calibration.

Predicts P(pulled after this PA) for any pitcher. Saved as decomposed
artifacts (booster.ubj + calibration.json + metadata.json) — same pattern
as the main XGBoost PA outcome model. Single-row inference at sim runtime.

Usage:
    # Cluster (DuckLake) — called by Dagster asset
    # Local dev:
    python -m xg.pitcher_exit --db-path stateball.duckdb
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sim.tables.re_table import RE_TABLE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature definitions — must match training query output column names.
# Order matters: this is the model's input contract.
# ---------------------------------------------------------------------------

FEATURES = [
    # Game state (sim engine tracks these)
    "pitcher_bf_game",
    "starter_flag",
    "outing_runs",
    "inning",
    "run_diff",
    "outs",
    "runners_on",
    "outing_walks",
    "outing_hits",
    "outing_k",
    "times_through_order",
    "outing_whip",
    "pitcher_recent_whip",
    "current_re",
    "re_diff",
    # Pitcher profile (static per pitcher per game, from ASOF join)
    "avg_bf_per_app",
    "pit_rest_days",
]


# ---------------------------------------------------------------------------
# Training data query
# ---------------------------------------------------------------------------

def _detect_table_prefix(conn) -> str:
    """Detect DuckLake vs local DuckDB table prefix."""
    try:
        conn.execute("SELECT 1 FROM lakehouse.main.int_mlb__game_state LIMIT 0")
        return "lakehouse.main"
    except Exception:
        return "main"


def _build_training_query(prefix: str) -> str:
    """SQL that produces training data with pull labels.

    Reads from int_mlb__game_state (outing counters, BF, WHIP, TTO, etc.)
    with ASOF join to int_mlb__pitchers for profile features.
    Derives: starter_flag, runners_on, pull label.
    RE features (current_re, re_diff) computed in Python post-query.
    """
    return f"""
    WITH terminal_pas AS (
        SELECT
            game_pk,
            at_bat_number,
            game_date,
            pitcher_id,
            is_bottom,
            inning,
            outs,
            runner_1b,
            runner_2b,
            runner_3b,
            run_diff,
            times_through_order,
            pitcher_bf_game,
            pitcher_outing_walks,
            pitcher_outing_hits,
            pitcher_outing_k,
            pitcher_outing_runs,
            pitcher_outing_whip,
            pitcher_recent_whip,
            target
        FROM {prefix}.int_mlb__game_state
        WHERE target IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY game_pk, at_bat_number
            ORDER BY pitch_number DESC
        ) = 1
    ),

    with_labels AS (
        SELECT
            t.*,
            CASE WHEN t.pitcher_id = FIRST_VALUE(t.pitcher_id) OVER (
                PARTITION BY t.game_pk, t.is_bottom
                ORDER BY t.at_bat_number
            ) THEN 1 ELSE 0 END AS starter_flag,
            t.runner_1b + t.runner_2b + t.runner_3b AS runners_on,
            CASE
                WHEN LEAD(t.pitcher_id) OVER (
                    PARTITION BY t.game_pk, t.is_bottom
                    ORDER BY t.at_bat_number
                ) IS NULL THEN 0
                WHEN LEAD(t.pitcher_id) OVER (
                    PARTITION BY t.game_pk, t.is_bottom
                    ORDER BY t.at_bat_number
                ) != t.pitcher_id THEN 1
                ELSE 0
            END AS pulled
        FROM terminal_pas t
    )

    SELECT
        s.game_pk,
        s.game_date,
        s.pitcher_id,
        s.is_bottom,
        s.at_bat_number,
        s.pitcher_bf_game,
        s.starter_flag,
        COALESCE(s.pitcher_outing_runs, 0) AS outing_runs,
        s.inning,
        s.run_diff,
        s.outs,
        s.runners_on,
        s.runner_1b,
        s.runner_2b,
        s.runner_3b,
        COALESCE(s.pitcher_outing_walks, 0) AS outing_walks,
        COALESCE(s.pitcher_outing_hits, 0) AS outing_hits,
        COALESCE(s.pitcher_outing_k, 0) AS outing_k,
        s.times_through_order,
        s.pitcher_outing_whip AS outing_whip,
        COALESCE(s.pitcher_recent_whip, 0.0) AS pitcher_recent_whip,
        COALESCE(pp.career_avg_bf_per_app, 20.0) AS avg_bf_per_app,
        COALESCE(pp.pitcher_rest_days, -1) AS pit_rest_days,
        s.pulled
    FROM with_labels s
    ASOF JOIN {prefix}.int_mlb__pitchers pp
        ON s.pitcher_id = pp.pitcher_id AND s.game_date >= pp.game_date
    WHERE s.game_date >= '2018-01-01'
    """


def _add_re_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute current_re and re_diff from runners + outs using RE_TABLE."""
    runners_bitmask = (
        df["runner_1b"].values
        + df["runner_2b"].values * 2
        + df["runner_3b"].values * 4
    ).astype(int)
    outs = df["outs"].values.astype(int).clip(0, 2)

    df["current_re"] = RE_TABLE[runners_bitmask, outs]

    # re_diff: change in RE from previous PA (same game, same half-inning)
    prev_re = df.groupby(["game_pk", "is_bottom"])["current_re"].shift(1)
    # Default previous state = bases empty, 0 outs
    df["re_diff"] = df["current_re"] - prev_re.fillna(RE_TABLE[0, 0])

    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _split_data(df: pd.DataFrame) -> tuple:
    """Prepare features and split into train/cal/test.

    Returns (X_train, y_train, X_cal, y_cal, X_test, y_test, pos_weight).
    """
    from sklearn.model_selection import train_test_split

    X = df[FEATURES].values.astype(np.float32)
    y = df["pulled"].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0)

    logger.info("Training data: %d rows, %.1f%% positive (pulled)", len(y), y.mean() * 100)

    # Split: 70% train, 15% calibration, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval,
    )  # 0.176 of 0.85 ≈ 0.15 of total

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    return X_train, y_train, X_cal, y_cal, X_test, y_test, pos_weight


def _train_and_evaluate(
    X_train, y_train, X_cal, y_cal, X_test, y_test,
    pos_weight: float,
    xgb_params: dict | None = None,
) -> tuple:
    """Train one XGBoost model, calibrate, evaluate.

    Returns (booster, calibration_table, metrics_dict).
    """
    from sklearn.calibration import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from xgboost import XGBClassifier

    defaults = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_leaves": 8,
        "max_bin": 128,
        "random_state": 42,
    }
    params = {**defaults, **(xgb_params or {})}

    model = XGBClassifier(
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=20,
        **params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_cal, y_cal)],
        verbose=False,
    )
    booster = model.get_booster()
    n_trees = booster.num_boosted_rounds()

    # Isotonic calibration on held-out set
    raw_probs = model.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(raw_probs, y_cal)

    # Evaluate on test set
    test_raw = model.predict_proba(X_test)[:, 1]
    test_cal = calibrator.predict(test_raw)

    auc = roc_auc_score(y_test, test_cal)
    brier = brier_score_loss(y_test, test_cal)
    ll = log_loss(y_test, test_cal)

    cal_table = {
        "x": calibrator.X_thresholds_.tolist(),
        "y": calibrator.y_thresholds_.tolist(),
    }

    importances = dict(zip(FEATURES, model.feature_importances_.tolist()))

    metrics = {
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "log_loss": round(ll, 4),
        "n_train": len(X_train),
        "n_cal": len(X_cal),
        "n_test": len(X_test),
        "pos_rate": round(float(y_train.mean()), 4),
        "n_trees": n_trees,
        "xgb_params": {k: v for k, v in params.items() if k != "random_state"},
        "feature_importances": importances,
    }

    return booster, cal_table, metrics


# ---------------------------------------------------------------------------
# Hyperparameter sweep — Optuna NSGA-II multi-objective
# ---------------------------------------------------------------------------

# Search space bounds
_SWEEP_SPACE = {
    "max_leaves": (4, 32),
    "learning_rate": (0.05, 0.4),
    "n_estimators": (50, 500),
    "min_child_weight": (10, 200),
    "subsample": (0.5, 1.0),
    "reg_lambda": (0.1, 50.0),
}
_SWEEP_N_TRIALS = 30
_SWEEP_SAMPLE_FRACTION = 0.3


def _run_sweep(
    X_train, y_train, X_cal, y_cal, X_test, y_test, pos_weight: float,
) -> list[dict]:
    """Multi-objective Optuna sweep: minimize (Brier, μs/row).

    Trains on a subsample for speed. Calibration and eval use full sets.
    Returns all completed trials with metrics.
    """
    import gc

    import optuna
    from sklearn.model_selection import train_test_split

    from xg.core.evaluate import inference_timing

    # Subsample training data — sweep ranks configs, doesn't produce final model
    X_sweep, _, y_sweep, _ = train_test_split(
        X_train, y_train,
        train_size=_SWEEP_SAMPLE_FRACTION,
        random_state=42,
        stratify=y_train,
    )
    sweep_pos_weight = (y_sweep == 0).sum() / max((y_sweep == 1).sum(), 1)
    logger.info("Sweep subsample: %d rows (%.0f%% of train)", len(X_sweep), _SWEEP_SAMPLE_FRACTION * 100)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = {
            "grow_policy": "lossguide",
            "max_depth": 0,
            "tree_method": "hist",
            "max_bin": 128,
            "max_leaves": trial.suggest_int("max_leaves", *_SWEEP_SPACE["max_leaves"]),
            "learning_rate": trial.suggest_float("learning_rate", *_SWEEP_SPACE["learning_rate"], log=True),
            "n_estimators": trial.suggest_int("n_estimators", *_SWEEP_SPACE["n_estimators"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *_SWEEP_SPACE["min_child_weight"]),
            "subsample": trial.suggest_float("subsample", *_SWEEP_SPACE["subsample"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *_SWEEP_SPACE["reg_lambda"], log=True),
        }

        logger.info("--- Trial %d ---", trial.number + 1)
        logger.debug(
            "  leaves=%d  lr=%.3f  n_est=%d  mcw=%d  sub=%.2f  lambda=%.1f",
            params["max_leaves"], params["learning_rate"], params["n_estimators"],
            params["min_child_weight"], params["subsample"], params["reg_lambda"],
        )

        booster, cal_table, metrics = _train_and_evaluate(
            X_sweep, y_sweep, X_cal, y_cal, X_test, y_test,
            pos_weight=sweep_pos_weight,
            xgb_params=params,
        )

        timing = inference_timing({"booster": booster}, X_test[:500])
        us_per_row = timing["ms_per_row"] * 1000

        trial.set_user_attr("auc", metrics["auc"])
        trial.set_user_attr("brier", metrics["brier"])
        trial.set_user_attr("n_trees", metrics["n_trees"])
        trial.set_user_attr("us_per_row", round(us_per_row, 2))

        logger.info(
            "  trees=%d  AUC=%.4f  Brier=%.4f  %.1fμs/row",
            metrics["n_trees"], metrics["auc"], metrics["brier"], us_per_row,
        )

        del booster
        gc.collect()

        # Minimize both: lower Brier = better quality, lower μs = faster
        return metrics["brier"], us_per_row

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42),
    )
    study.optimize(objective, n_trials=_SWEEP_N_TRIALS)

    # Extract all trials
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results.append({
                "trial": trial.number,
                "params": dict(trial.params),
                "brier": trial.values[0],
                "us_per_row": trial.values[1],
                "auc": trial.user_attrs["auc"],
                "n_trees": trial.user_attrs["n_trees"],
            })

    # Log summary
    results.sort(key=lambda r: r["brier"])
    logger.info("")
    logger.info("%-6s %-6s %-5s %-6s %-7s %-7s %-8s", "trial", "leaves", "lr", "trees", "AUC", "Brier", "μs/row")
    logger.info("-" * 55)
    for r in results:
        logger.info(
            "%-6d %-6d %-5.2f %-6d %-7.4f %-7.4f %-8.2f",
            r["trial"], r["params"]["max_leaves"], r["params"]["learning_rate"],
            r["n_trees"], r["auc"], r["brier"], r["us_per_row"],
        )

    return results


def _extract_frontier(results: list[dict]) -> list[dict]:
    """Extract Pareto-optimal trials (non-dominated on Brier × μs/row)."""
    frontier = []
    for r in results:
        dominated = any(
            o["brier"] <= r["brier"] and o["us_per_row"] <= r["us_per_row"]
            and (o["brier"] < r["brier"] or o["us_per_row"] < r["us_per_row"])
            for o in results
        )
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda r: r["us_per_row"])
    return frontier


def _select_best(frontier: list[dict]) -> dict:
    """Pick the speed/quality knee from the Pareto frontier.

    Strategy: lowest Brier that's within 1.5x inference time of the fastest
    Pareto candidate. Biases hard toward speed.
    """
    fastest_us = min(r["us_per_row"] for r in frontier)
    threshold = fastest_us * 1.5

    # Among speed-eligible, pick lowest Brier
    eligible = [r for r in frontier if r["us_per_row"] <= threshold]
    if not eligible:
        eligible = frontier

    best = min(eligible, key=lambda r: r["brier"])
    logger.info(
        "Selected: trial %d, leaves=%d lr=%.3f trees=%d (AUC=%.4f, Brier=%.4f, %.1fμs/row)",
        best["trial"], best["params"]["max_leaves"], best["params"]["learning_rate"],
        best["n_trees"], best["auc"], best["brier"], best["us_per_row"],
    )
    return best


# ---------------------------------------------------------------------------
# Top-level training entry point
# ---------------------------------------------------------------------------

def _train_model(df: pd.DataFrame, sweep: bool = False) -> tuple:
    """Train XGBoost binary classifier + isotonic calibration.

    Args:
        df: Training dataframe with FEATURES columns and 'pulled' label.
        sweep: If True, run Optuna NSGA-II sweep to find fastest well-calibrated model.

    Returns (booster, calibration_table, metrics_dict).
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test, pos_weight = _split_data(df)

    if sweep:
        logger.info("Running Optuna NSGA-II sweep (%d trials)...", _SWEEP_N_TRIALS)
        results = _run_sweep(X_train, y_train, X_cal, y_cal, X_test, y_test, pos_weight)
        frontier = _extract_frontier(results)
        logger.info("Pareto frontier: %d candidates from %d trials", len(frontier), len(results))
        best = _select_best(frontier)

        # Retrain winner on full training data
        retrain_params = {
            **best["params"],
            "grow_policy": "lossguide",
            "max_depth": 0,
            "tree_method": "hist",
            "max_bin": 128,
            "n_estimators": best["n_trees"],
        }
        logger.info(
            "Retraining winner: leaves=%d lr=%.3f n_trees=%d",
            retrain_params["max_leaves"], retrain_params["learning_rate"], retrain_params["n_estimators"],
        )
        booster, cal_table, metrics = _train_and_evaluate(
            X_train, y_train, X_cal, y_cal, X_test, y_test,
            pos_weight=pos_weight,
            xgb_params=retrain_params,
        )
        metrics["sweep_results"] = results
        metrics["pareto_frontier"] = frontier
    else:
        booster, cal_table, metrics = _train_and_evaluate(
            X_train, y_train, X_cal, y_cal, X_test, y_test,
            pos_weight=pos_weight,
        )

    logger.info(
        "XGBoost trained: %d trees, AUC=%.4f, Brier=%.4f, LogLoss=%.4f",
        metrics["n_trees"], metrics["auc"], metrics["brier"], metrics["log_loss"],
    )

    return booster, cal_table, metrics


# ---------------------------------------------------------------------------
# Public API: build / validate / save
# ---------------------------------------------------------------------------

def build(conn, sweep: bool = False) -> dict:
    """End-to-end: query → derive RE features → train → package.

    Args:
        conn: DuckDB connection (local or DuckLake).
        sweep: If True, run hyperparameter sweep to find leanest model.

    Returns:
        {"booster": xgb.Booster, "calibration": dict, "metadata": dict}
    """
    prefix = _detect_table_prefix(conn)
    logger.info("Training pitcher exit model from %s", prefix)

    query = _build_training_query(prefix)
    df = conn.execute(query).fetchdf()
    logger.info("Training data: %d rows", len(df))

    if len(df) < 1000:
        raise ValueError(f"Insufficient training data: {len(df)} rows (need >= 1000)")

    df = _add_re_features(df)

    booster, cal_table, metrics = _train_model(df, sweep=sweep)

    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": f"{prefix}.int_mlb__game_state + {prefix}.int_mlb__pitchers",
        "n_training_rows": len(df),
        "model_metrics": metrics,
        "features": FEATURES,
    }

    return {
        "booster": booster,
        "calibration": cal_table,
        "metadata": metadata,
    }


def validate(result: dict) -> None:
    """Validate training output: booster exists, calibration is sane, metrics pass."""
    errors = []

    if result.get("booster") is None:
        errors.append("No booster in result")

    cal = result.get("calibration", {})
    if not cal.get("x") or not cal.get("y"):
        errors.append("Calibration table missing or empty")
    elif len(cal["x"]) != len(cal["y"]):
        errors.append(f"Calibration x/y length mismatch: {len(cal['x'])} vs {len(cal['y'])}")

    metrics = result.get("metadata", {}).get("model_metrics", {})
    auc = metrics.get("auc", 0)
    if auc < 0.6:
        errors.append(f"AUC too low: {auc} (need >= 0.6)")

    brier = metrics.get("brier", 1)
    if brier > 0.25:
        errors.append(f"Brier score too high: {brier} (need <= 0.25)")

    if errors:
        raise ValueError(
            f"Validation failed ({len(errors)} errors):\n" + "\n".join(errors)
        )

    logger.info(
        "Validation passed: AUC=%.4f, Brier=%.4f, %d trees, %d features",
        auc, brier, metrics.get("n_trees", 0), len(result.get("metadata", {}).get("features", [])),
    )


def save(result: dict, path: str) -> str:
    """Save pitcher exit model artifacts to a directory (local or s3://).

    Writes:
        <path>/model.ubj          — XGBoost booster
        <path>/model.onnx         — ONNX export for fast inference
        <path>/calibration.json   — isotonic calibration thresholds
        <path>/metadata.json      — features, metrics, build info
    """
    import tempfile

    booster = result["booster"]
    cal_table = result["calibration"]
    metadata = result["metadata"]
    feature_names = metadata.get("features", FEATURES)

    # Export ONNX from booster (same pattern as PA outcome model)
    onnx_bytes = None
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_types = [("X", FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_xgboost(booster, initial_types=initial_types, target_opset=15)
        onnx_bytes = onnx_model.SerializeToString()
        logger.info("Exported ONNX model: model.onnx (%d features)", len(feature_names))
    except ImportError:
        logger.info("onnxmltools not installed — skipping ONNX export")
    except Exception:
        logger.warning("ONNX export failed", exc_info=True)

    if path.startswith("s3://"):
        import s3fs

        fs = s3fs.S3FileSystem(
            key=os.environ.get("S3_ACCESS_KEY_ID", ""),
            secret=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
            endpoint_url=os.environ.get("S3_ENDPOINT", ""),
        )

        # Booster
        with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
            booster.save_model(tmp.name)
            with open(tmp.name, "rb") as f:
                with fs.open(f"{path}/model.ubj", "wb") as s3f:
                    s3f.write(f.read())
            os.unlink(tmp.name)

        # ONNX
        if onnx_bytes is not None:
            with fs.open(f"{path}/model.onnx", "wb") as f:
                f.write(onnx_bytes)

        # Calibration
        with fs.open(f"{path}/calibration.json", "w") as f:
            json.dump(cal_table, f, indent=2)

        # Metadata
        with fs.open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    else:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # Booster
        booster.save_model(str(out / "model.ubj"))

        # ONNX
        if onnx_bytes is not None:
            with open(out / "model.onnx", "wb") as f:
                f.write(onnx_bytes)

        # Calibration
        with open(out / "calibration.json", "w") as f:
            json.dump(cal_table, f, indent=2)

        # Metadata
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info("Saved pitcher exit model to %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import duckdb

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train pitcher exit model")
    parser.add_argument(
        "--db-path",
        default="stateball.duckdb",
        help="Path to local DuckDB file (default: stateball.duckdb)",
    )
    parser.add_argument(
        "--output",
        default="sim/data/pitcher_exit",
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run hyperparameter sweep to find leanest effective model",
    )
    args = parser.parse_args()

    conn = duckdb.connect(str(args.db_path), read_only=True)
    try:
        result = build(conn, sweep=args.sweep)
    finally:
        conn.close()

    validate(result)
    save(result, args.output)
