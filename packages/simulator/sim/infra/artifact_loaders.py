"""Artifact loading for the game simulator.

Loads baserunning tables and XGBoost model bundles from local paths or
S3 URIs. Validates model compatibility against the canonical
feature/outcome definitions in training.config.

Same s3fs + env var pattern used by the artifact builders and training
pipeline. Works from local dev (.env via seed-env.sh) and prod (k8s
secrets) without code changes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 access — same pattern as build_baserunning_table.py and xgboost.py
# ---------------------------------------------------------------------------


def _get_fs():
    import s3fs

    return s3fs.S3FileSystem(
        key=os.environ.get("S3_ACCESS_KEY_ID", ""),
        secret=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
        endpoint_url=os.environ.get("S3_ENDPOINT", ""),
    )


def _read_json(path: str) -> dict:
    if path.startswith("s3://"):
        fs = _get_fs()
        with fs.open(path, "r") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Lookup table loading
# ---------------------------------------------------------------------------


def load_baserunning_table(path: str) -> dict:
    """Load baserunning transition table from local path or s3:// URI.

    Returns the dict consumed by state.resolve_outcome():
    {"transitions": {key: [{post_bases, runs_scored, outs_added, p}, ...]}, "metadata": {...}}
    """
    table = _read_json(path)
    n_keys = len(table.get("transitions", {}))
    logger.info("Loaded baserunning table: %d keys from %s", n_keys, path)
    return table


def _compile_trees(booster) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract XGBoost tree structure into flat numpy arrays for pure-numpy traversal.

    Returns (feature_idx, threshold, left_child, right_child, leaf_value),
    each shaped (n_trees, max_nodes).
    """
    df = booster.trees_to_dataframe()
    n_trees = int(df["Tree"].max()) + 1
    max_nodes = int(df["Node"].max()) + 1

    feature_idx = np.full((n_trees, max_nodes), -1, dtype=np.int16)
    threshold = np.zeros((n_trees, max_nodes), dtype=np.float32)
    left_child = np.zeros((n_trees, max_nodes), dtype=np.int16)
    right_child = np.zeros((n_trees, max_nodes), dtype=np.int16)
    leaf_value = np.zeros((n_trees, max_nodes), dtype=np.float32)

    for _, row in df.iterrows():
        t = int(row["Tree"])
        n = int(row["Node"])
        if row["Feature"] == "Leaf":
            leaf_value[t, n] = row["Gain"]
        else:
            # Feature names are "f0", "f1", ... — extract index
            feature_idx[t, n] = int(row["Feature"][1:])
            threshold[t, n] = row["Split"]
            # Yes/No are "tree-node" strings like "0-3"
            left_child[t, n] = int(str(row["Yes"]).split("-")[1])
            right_child[t, n] = int(str(row["No"]).split("-")[1])

    return feature_idx, threshold, left_child, right_child, leaf_value


@dataclass(frozen=True)
class PitcherExitModel:
    """Pitcher exit binary classifier — ONNX inference + isotonic calibration.

    Uses ONNX runtime for compiled C++ tree traversal when available.
    Falls back to pure numpy traversal for old artifacts without model.onnx.
    """

    cal_x: Any               # float64 — isotonic X thresholds
    cal_y: Any               # float64 — isotonic Y thresholds
    feature_names: list[str]
    onnx_session: Any = None  # ort.InferenceSession (None = numpy fallback)
    onnx_input_name: str = ""
    # Numpy tree fallback fields (populated only when ONNX unavailable)
    tree_feature_idx: Any = None
    tree_threshold: Any = None
    tree_left: Any = None
    tree_right: Any = None
    tree_leaf_value: Any = None
    n_trees: int = 0
    max_depth: int = 0

    def _extract_raw_probs(self, X: np.ndarray) -> np.ndarray:
        """Run inference on (n, n_features) float32 matrix → (n,) raw probabilities."""
        if self.onnx_session is not None:
            outputs = self.onnx_session.run(None, {self.onnx_input_name: X})
            probs = outputs[1]  # probability tensor
            if probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1].astype(np.float64)
            return probs.ravel().astype(np.float64)

        # Numpy tree fallback
        n_samples = X.shape[0]
        scores = np.zeros(n_samples, dtype=np.float64)
        sample_idx = np.arange(n_samples)
        for t in range(self.n_trees):
            nodes = np.zeros(n_samples, dtype=np.int16)
            for _ in range(self.max_depth):
                feat = self.tree_feature_idx[t, nodes]
                is_leaf = feat == -1
                if is_leaf.all():
                    break
                feat_vals = X[sample_idx, np.where(is_leaf, 0, feat)]
                go_left = feat_vals < self.tree_threshold[t, nodes]
                next_nodes = np.where(go_left, self.tree_left[t, nodes], self.tree_right[t, nodes])
                nodes = np.where(is_leaf, nodes, next_nodes)
            scores += self.tree_leaf_value[t, nodes]
        return 1.0 / (1.0 + np.exp(-scores))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """(n, n_features) → calibrated P(pulled) array."""
        raw_probs = self._extract_raw_probs(X.astype(np.float32))
        return np.interp(raw_probs, self.cal_x, self.cal_y)

    def predict_transposed(self, features_T: np.ndarray) -> np.ndarray:
        """(n_features, n_samples) → calibrated P(pulled) array."""
        X = np.ascontiguousarray(features_T.T, dtype=np.float32)
        raw_probs = self._extract_raw_probs(X)
        return np.interp(raw_probs, self.cal_x, self.cal_y).astype(np.float32)

    def predict(self, features: dict[str, float]) -> float:
        """Single-row predict for scalar engine."""
        row = np.array(
            [features.get(f, 0.0) for f in self.feature_names],
            dtype=np.float32,
        ).reshape(1, -1)
        return float(self.predict_batch(row)[0])


def load_pitcher_exit_model(path: str) -> PitcherExitModel:
    """Load pitcher exit model from a run directory (local or s3://).

    Expects: metadata.json, calibration.json, and either model.onnx (preferred)
    or model.ubj (numpy tree fallback).
    """
    path = path.rstrip("/")

    # Load metadata + calibration (always needed)
    metadata = _read_json(f"{path}/metadata.json")
    feature_names = metadata.get("features", [])
    if not feature_names:
        raise ValueError(f"No features in {path}/metadata.json")

    cal_raw = _read_json(f"{path}/calibration.json")
    cal_x = np.array(cal_raw["x"], dtype=np.float64)
    cal_y = np.array(cal_raw["y"], dtype=np.float64)

    # Try ONNX first (fast path)
    onnx_session = None
    onnx_input_name = ""
    try:
        import onnxruntime as ort
        onnx_data = _read_binary(f"{path}/model.onnx")
        onnx_session = ort.InferenceSession(onnx_data)
        onnx_input_name = onnx_session.get_inputs()[0].name
        logger.info(
            "Loaded pitcher exit model: ONNX, %d features, "
            "%d calibration points from %s",
            len(feature_names), len(cal_x), path,
        )
        return PitcherExitModel(
            cal_x=cal_x, cal_y=cal_y, feature_names=feature_names,
            onnx_session=onnx_session, onnx_input_name=onnx_input_name,
        )
    except Exception:
        logger.info("No ONNX model found, falling back to numpy tree traversal")

    # Numpy tree fallback — compile booster into flat arrays
    import tempfile
    import xgboost as xgb

    ubj_data = _read_binary(f"{path}/model.ubj")
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
        tmp.write(ubj_data)
        tmp.flush()
        booster = xgb.Booster(model_file=tmp.name)
        os.unlink(tmp.name)

    feat_idx, thresh, left, right, leaf_val = _compile_trees(booster)
    n_trees = len(feat_idx)

    max_depth = 0
    for tree_str in booster.get_dump():
        for line in tree_str.split("\n"):
            if ":" in line:
                d = len(line) - len(line.lstrip("\t"))
                max_depth = max(max_depth, d)
    max_depth = max(max_depth, 1)

    logger.info(
        "Loaded pitcher exit model: %d trees, depth %d, %d features, "
        "%d calibration points from %s (numpy fallback)",
        n_trees, max_depth, len(feature_names), len(cal_x), path,
    )

    return PitcherExitModel(
        cal_x=cal_x, cal_y=cal_y, feature_names=feature_names,
        tree_feature_idx=feat_idx, tree_threshold=thresh,
        tree_left=left, tree_right=right, tree_leaf_value=leaf_val,
        n_trees=n_trees, max_depth=max_depth,
    )


def load_win_expectancy_table(path: str) -> dict:
    """Load win expectancy lookup table from local path or s3:// URI.

    Returns the dict with hierarchical P(home_win) by game state:
    {"levels": {"full": {...}, "no_bases": {...}, "coarse": {...}, "baseline": {...}}}
    """
    table = _read_json(path)
    n_full = len(table.get("levels", {}).get("full", {}))
    n_no_bases = len(table.get("levels", {}).get("no_bases", {}))
    n_coarse = len(table.get("levels", {}).get("coarse", {}))
    logger.info(
        "Loaded win expectancy table: %d full / %d no_bases / %d coarse keys from %s",
        n_full, n_no_bases, n_coarse, path,
    )
    return table



def _normalize_lookup(config: dict, legacy_key: str | None = None) -> dict:
    """Normalize artifact format: ensure 'lookup' key exists.

    New format: {"lookup": {...}, "metadata": {...}}
    Legacy format: {"n_lookup": {...}, "stopping_thresholds": {...}, ...}
    """
    if "lookup" in config:
        return config
    if legacy_key and legacy_key in config:
        config["lookup"] = config.pop(legacy_key)
    elif "h_lookup" in config:
        config["lookup"] = config.pop("h_lookup")
    return config


def load_n_lookup(path: str) -> dict:
    """Load per-state N allocation artifact."""
    config = _normalize_lookup(_read_json(path), legacy_key="n_lookup")
    n_states = len(config.get("lookup", {}))
    logger.info("Loaded n_lookup: %d states from %s", n_states, path)
    return config


def load_stopping_thresholds(path: str) -> dict:
    """Load per-state SE stopping thresholds artifact."""
    config = _normalize_lookup(_read_json(path), legacy_key="stopping_thresholds")
    n_states = len(config.get("lookup", {}))
    logger.info("Loaded stopping_thresholds: %d states from %s", n_states, path)
    return config


def load_gamma_schedule(path: str) -> dict:
    """Load per-state SMC gamma tempering artifact."""
    config = _normalize_lookup(_read_json(path), legacy_key="gamma_schedule")
    n_states = len(config.get("lookup", {}))
    logger.info("Loaded gamma_schedule: %d states from %s", n_states, path)
    return config


def load_horizon_weights(path: str) -> dict:
    """Load per-state horizon cutoff artifact."""
    config = _normalize_lookup(_read_json(path))
    n_states = len(config.get("lookup", {}))
    logger.info("Loaded horizon_weights: %d states from %s", n_states, path)
    return config


def build_blowout_thresholds(
    we_table: dict,
    p_threshold: float = 0.99,
    margin: int = 1,
) -> tuple:
    """Derive blowout pruning thresholds from win expectancy table.

    Scans the coarse level (inning|half|run_diff) to find the minimum
    run_diff magnitude at which P(home_win) >= p_threshold (or <= 1-p),
    then adds `margin` runs for safety.

    Returns:
        (win_thresholds, loss_thresholds) — two int16 arrays shaped (max_inn+1, 2)
        where axis 1 is half_idx (0=Top, 1=Bot). win_thresholds[inn][half] is the
        positive run_diff at which home is considered a blowout winner.
        loss_thresholds[inn][half] is the negative run_diff for blowout loss.
    """
    import numpy as np

    coarse = we_table.get("levels", {}).get("coarse", {})
    if not coarse:
        raise ValueError("WE table has no coarse level — cannot build thresholds")

    max_inning = 9  # extras clamp to 9
    half_map = {"Top": 0, "Bot": 1}
    no_trigger = 16  # fallback: never triggers

    # Initialize with safe defaults
    win_thresh = np.full((max_inning + 1, 2), no_trigger, dtype=np.int16)
    loss_thresh = np.full((max_inning + 1, 2), -no_trigger, dtype=np.int16)

    # Collect all entries by (inning, half)
    by_state: dict[tuple[int, int], dict[int, float]] = {}
    for key, val in coarse.items():
        parts = key.split("|")
        inn = int(parts[0])
        half_idx = half_map.get(parts[1])
        rd = int(parts[2])
        if half_idx is None or inn < 1 or inn > max_inning:
            continue
        by_state.setdefault((inn, half_idx), {})[rd] = val["p_home_win"]

    for (inn, half_idx), rd_map in by_state.items():
        # Win threshold: min positive rd where p >= p_threshold
        for rd in sorted(rd_map.keys()):
            if rd > 0 and rd_map[rd] >= p_threshold:
                win_thresh[inn, half_idx] = rd + margin
                break

        # Loss threshold: max negative rd where p <= (1 - p_threshold)
        for rd in sorted(rd_map.keys(), reverse=True):
            if rd < 0 and rd_map[rd] <= (1 - p_threshold):
                loss_thresh[inn, half_idx] = rd - margin
                break

    logger.info("Blowout thresholds (p=%.2f, margin=%d):", p_threshold, margin)
    for inn in range(1, max_inning + 1):
        logger.info(
            "  Inn %d: Top win=%+d loss=%+d | Bot win=%+d loss=%+d",
            inn,
            int(win_thresh[inn, 0]), int(loss_thresh[inn, 0]),
            int(win_thresh[inn, 1]), int(loss_thresh[inn, 1]),
        )

    return win_thresh, loss_thresh


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelBundle:
    """Self-describing model artifact.

    Bundles the model with its feature contract and outcome labels.
    Prefers decomposed format (booster + calibration tables) over sklearn wrapper.
    feature_names defines the exact columns (in order) the model expects.
    outcome_labels defines the classes the model can predict.
    """

    booster: Any                    # xgb.Booster (native) or None if legacy
    calibration_tables: dict | None # {str(class_idx): {"x": ndarray, "y": ndarray}}
    feature_names: list[str]
    outcome_labels: list[str]
    model: Any = None               # CalibratedClassifierCV (legacy fallback)
    onnx_session: Any = None        # ort.InferenceSession or None


def _file_exists(path: str) -> bool:
    """Check if a local or S3 path exists."""
    if path.startswith("s3://"):
        fs = _get_fs()
        return fs.exists(path)
    return os.path.exists(path)


def _read_binary(path: str) -> bytes:
    """Read binary from local or S3 path."""
    if path.startswith("s3://"):
        fs = _get_fs()
        with fs.open(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def load_model(run_dir: str) -> ModelBundle:
    """Load model + metadata from a run directory.

    Tries decomposed format first (model.ubj + calibration.json), falls back
    to legacy joblib. The decomposed format is faster and doesn't need sklearn.

    Args:
        run_dir: Local path or s3:// URI to the run directory
                 (e.g. "s3://<bucket>/stateball/artifacts/xgboost/20260306_052117_default").
    """
    run_dir = run_dir.rstrip("/")

    # Load results metadata (always needed)
    results_path = f"{run_dir}/results.json"
    results = _read_json(results_path)

    feature_names = results.get("feature_names", [])
    if not feature_names:
        raise ValueError(f"No feature_names in {results_path}")

    # Try decomposed format first
    ubj_path = f"{run_dir}/model.ubj"
    cal_path = f"{run_dir}/calibration.json"

    if _file_exists(ubj_path) and _file_exists(cal_path):
        import tempfile

        import numpy as np
        import xgboost as xgb

        # Load booster from UBJ
        ubj_data = _read_binary(ubj_path)
        with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
            tmp.write(ubj_data)
            tmp.flush()
            booster = xgb.Booster(model_file=tmp.name)
            os.unlink(tmp.name)

        # Load calibration tables
        cal_raw = _read_json(cal_path)
        cal_tables = {
            int(k): {"x": np.array(v["x"], dtype=np.float64),
                      "y": np.array(v["y"], dtype=np.float64)}
            for k, v in cal_raw.items()
        }

        outcome_labels = results.get("classes", [])
        if not outcome_labels:
            raise ValueError(f"No 'classes' in {results_path}")

        # Try loading ONNX model (preferred inference path)
        onnx_session = None
        onnx_path = f"{run_dir}/model.onnx"
        try:
            if _file_exists(onnx_path):
                import onnxruntime as ort
                onnx_data = _read_binary(onnx_path)
                onnx_session = ort.InferenceSession(onnx_data)
                logger.info("Loaded ONNX model from %s", onnx_path)
        except ImportError:
            logger.info("onnxruntime not installed — using native booster")
        except Exception as e:
            logger.info("ONNX model not available (%s) — using native booster", e)

        bundle = ModelBundle(
            booster=booster,
            calibration_tables=cal_tables,
            feature_names=feature_names,
            outcome_labels=outcome_labels,
            onnx_session=onnx_session,
        )
        logger.info(
            "Loaded decomposed model: %d features, %d outcomes, %d calibration classes, onnx=%s, run=%s",
            len(feature_names), len(outcome_labels), len(cal_tables),
            onnx_session is not None,
            results.get("run_id", "unknown"),
        )
    else:
        # Legacy fallback: joblib
        import joblib

        model_path = f"{run_dir}/model.joblib"
        if model_path.startswith("s3://"):
            fs = _get_fs()
            with fs.open(model_path, "rb") as f:
                artifact = joblib.load(f)
        else:
            artifact = joblib.load(model_path)

        model = artifact["model"]
        label_encoder = artifact["label_encoder"]
        outcome_labels = list(label_encoder.classes_)

        bundle = ModelBundle(
            booster=None,
            calibration_tables=None,
            feature_names=feature_names,
            outcome_labels=outcome_labels,
            model=model,
        )
        logger.info(
            "Loaded legacy model (joblib): %d features, %d outcomes, run=%s",
            len(feature_names), len(outcome_labels),
            results.get("run_id", "unknown"),
        )

    _log_compatibility(bundle)
    return bundle


def _log_compatibility(bundle: ModelBundle):
    """Log compatibility between model and current config definitions."""
    from xg.core.config import FEATURE_BLOCKS, OUTCOME_CLASSES

    # Resolve canonical feature set
    config_features = set()
    for block in FEATURE_BLOCKS.values():
        config_features.update(block)

    model_features = set(bundle.feature_names)
    model_outcomes = set(bundle.outcome_labels)
    known_outcomes = set(OUTCOME_CLASSES)

    # Features the model uses that aren't in current config
    extra_features = model_features - config_features
    if extra_features:
        logger.warning(
            "Model uses %d features not in current config: %s",
            len(extra_features),
            sorted(extra_features),
        )

    # Features in config that the model doesn't use
    missing_features = config_features - model_features
    if missing_features:
        logger.info(
            "Config has %d features the model doesn't use: %s",
            len(missing_features),
            sorted(missing_features),
        )

    # Outcome compatibility
    extra_outcomes = model_outcomes - known_outcomes
    if extra_outcomes:
        logger.warning(
            "Model predicts outcomes not in OUTCOME_CLASSES: %s",
            sorted(extra_outcomes),
        )

    unknown_outcomes = known_outcomes - model_outcomes
    if unknown_outcomes:
        logger.info(
            "OUTCOME_CLASSES has %d outcomes the model doesn't predict: %s",
            len(unknown_outcomes),
            sorted(unknown_outcomes),
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def list_runs(base: str | None = None) -> dict:
    """List available artifact runs on S3.

    Returns {"sim": [run_id, ...], "xgboost": [run_id, ...]}.
    """
    if base is None:
        _bucket = os.environ.get("S3_BUCKET", "dazoo")
        base = f"s3://{_bucket}/stateball/artifacts"
    fs = _get_fs()
    base = base.rstrip("/")
    result = {}
    for category in ("sim", "xgboost"):
        prefix = f"{base}/{category}/"
        try:
            # s3fs returns full paths; strip prefix to get run_ids
            dirs = fs.ls(prefix.lstrip("s3://").lstrip("/"))
            result[category] = sorted(
                d.split("/")[-1] for d in dirs
            )
        except FileNotFoundError:
            result[category] = []
    return result
