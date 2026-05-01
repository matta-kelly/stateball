"""
Data loading, splitting, and preparation for XGBoost training pipelines.

Extracted from the training monolith. All pipelines (train, sweep, validate)
use prepare_data() as their entry point.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xg.core.config import ExperimentConfig, resolve_features
from xg.core.io import REPO_ROOT, log_memory

logger = logging.getLogger("xg.data")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(cfg: ExperimentConfig) -> pd.DataFrame:
    """Load feature vectors from DuckDB or DuckLake with optional filtering.

    When sampling is enabled, PA sampling happens server-side via QUALIFY
    so only one pitch per plate appearance enters Python memory.
    """
    logger.info("load: mode=%s, table=%s", cfg.data.connection_mode, cfg.data.table)
    log_memory("load start")

    if cfg.data.connection_mode == "ducklake":
        from orchestration.lib import get_ducklake_connection
        conn = get_ducklake_connection()
        table = f"lakehouse.{cfg.data.table}"
    else:
        db_path = Path(cfg.data.db_path)
        if not db_path.is_absolute():
            db_path = REPO_ROOT / db_path
        conn = duckdb.connect(str(db_path), read_only=True)
        table = cfg.data.table

    logger.info("load: connection established")
    log_memory("load (post-connect)")

    where_clauses: list[str] = []
    if cfg.data.seasons:
        seasons_str = ", ".join(str(s) for s in cfg.data.seasons)
        where_clauses.append(f"EXTRACT(YEAR FROM game_date::DATE) IN ({seasons_str})")
    if cfg.data.date_range and len(cfg.data.date_range) == 2:
        where_clauses.append(f"game_date >= '{cfg.data.date_range[0]}'")
        where_clauses.append(f"game_date <= '{cfg.data.date_range[1]}'")
    if cfg.data.min_game_date:
        where_clauses.append(f"game_date >= '{cfg.data.min_game_date}'")

    where = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # PA sampling: one random pitch per plate appearance, server-side.
    # Uses a narrow CTE for the window function (3 key columns only) then
    # joins back for the full row — avoids DuckDB carrying all columns through
    # the partition/sort which OOMs on large datasets.
    if cfg.sampling.enabled:
        seed = cfg.sampling.seed
        query = (
            f"WITH sampled AS ("
            f"  SELECT game_pk, at_bat_number, pitch_number"
            f"  FROM {table}{where}"
            f"  QUALIFY ROW_NUMBER() OVER ("
            f"    PARTITION BY game_pk, at_bat_number"
            f"    ORDER BY hash(pitch_number + {seed})"
            f"  ) = 1"
            f") SELECT t.* FROM {table} t"
            f" INNER JOIN sampled s USING (game_pk, at_bat_number, pitch_number)"
        )
    else:
        query = f"SELECT * FROM {table}{where}"

    logger.info("load: executing query (sampling=%s, filters=%s)", cfg.sampling.enabled, where.strip() or "none")
    log_memory("load (pre-fetchdf)")

    df = conn.execute(query).fetchdf()
    conn.close()
    log_memory("load (post-fetchdf)")

    if cfg.data.max_games:
        game_pks = df["game_pk"].unique()
        if len(game_pks) > cfg.data.max_games:
            rng = np.random.RandomState(42)
            keep = rng.choice(game_pks, cfg.data.max_games, replace=False)
            df = df[df["game_pk"].isin(set(keep))]

    if cfg.sampling.enabled:
        logger.info("Loaded %d PAs (sampled), %d games", len(df), df["game_pk"].nunique())
    else:
        logger.info("Loaded %d pitches, %d games", len(df), df["game_pk"].nunique())
    return df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split(df: pd.DataFrame, cfg: ExperimentConfig):
    """Stratified split into train/calibrate/test.

    Two-level stratified split preserves class proportions in every partition:
    1. Hold out test_fraction as the untouched test set
    2. Split remaining dev into train and calibrate
    """
    seed = cfg.split.seed

    dev_df, test_df = train_test_split(
        df, test_size=cfg.split.test_fraction,
        random_state=seed, stratify=df["target"],
    )
    train_df, calib_df = train_test_split(
        dev_df, test_size=cfg.split.calibrate_fraction,
        random_state=seed, stratify=dev_df["target"],
    )

    return train_df, calib_df, test_df


# ---------------------------------------------------------------------------
# Data preparation (shared by train, sweep, and validate pipelines)
# ---------------------------------------------------------------------------

def prepare_data(cfg: ExperimentConfig, feature_cols: list[str] | None = None) -> dict:
    """Load, sample, filter rare classes, encode, split. Shared by all pipelines.

    Args:
        cfg: Experiment config (data source, split params, etc.).
        feature_cols: Explicit feature list (from manifest). When provided,
            skips resolve_features() and uses this list directly.
    """
    df = load(cfg)

    if feature_cols is None:
        feature_cols = resolve_features(cfg.features, df_columns=list(df.columns))
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not in data: {missing}")
    logger.info("Features (%d): %s...%s", len(feature_cols), feature_cols[:5], feature_cols[-2:])

    # Filter rare classes — threshold accounts for calibration set needing
    # >= cv examples per class after stratified split.
    calib_share = (1 - cfg.split.test_fraction) * cfg.split.calibrate_fraction
    min_samples = max(20, int(np.ceil(cfg.calibration.cv / calib_share)))
    class_counts = df["target"].value_counts()
    rare = class_counts[class_counts < min_samples].index
    if len(rare) > 0:
        n_before = len(df)
        df = df[~df["target"].isin(rare)]
        logger.info("Filtered %d rare classes (%s): %d -> %d samples", len(rare), list(rare), n_before, len(df))

    le = LabelEncoder()
    le.fit(df["target"])
    logger.info("Classes (%d): %s", len(le.classes_), list(le.classes_))

    train_df, calib_df, test_df = split(df, cfg)
    logger.info("Train: %d  Calibrate: %d  Test: %d", len(train_df), len(calib_df), len(test_df))

    X_train = train_df[feature_cols].values
    y_train = le.transform(train_df["target"])
    X_calib = calib_df[feature_cols].values
    y_calib = le.transform(calib_df["target"])
    X_test = test_df[feature_cols].values
    y_test = le.transform(test_df["target"])

    # Capture metadata before freeing DataFrames
    train_games = int(train_df["game_pk"].nunique())
    total_pas = len(df)
    train_pas = len(train_df)
    calibrate_pas = len(calib_df)
    test_pas = len(test_df)
    date_range = [str(df["game_date"].min())[:10], str(df["game_date"].max())[:10]]
    seasons = sorted(df["game_date"].str[:4].unique().astype(int).tolist())

    # Free source DataFrames — numpy arrays are independent copies
    del df, train_df, calib_df, test_df
    gc.collect()
    log_memory("prepare_data (post-gc)")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_calib": X_calib, "y_calib": y_calib,
        "X_test": X_test, "y_test": y_test,
        "le": le,
        "class_names": list(le.classes_),
        "feature_cols": feature_cols,
        "n_classes": len(le.classes_),
        "train_games": train_games,
        "total_pas": total_pas,
        "train_pas": train_pas,
        "calibrate_pas": calibrate_pas,
        "test_pas": test_pas,
        "date_range": date_range,
        "seasons": seasons,
    }
