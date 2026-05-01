"""XGBoost probability source for the game simulator.

Wraps a trained ModelBundle into a ProbSource callable that the engine
can use. Assembles feature vectors from batter/pitcher profiles, game
state, and in-game context — then runs inference.

Two inference paths:
  - Decomposed (preferred): native XGBoost booster + numpy isotonic calibration.
    No sklearn at runtime. Static profile features cached per matchup.
  - Legacy: sklearn CalibratedClassifierCV.predict_proba() via joblib pickle.

Feature name translation: the model expects names like ``bat_season_ba``
and ``pit_career_whip``, but profile dicts from init.py carry raw column
names (``season_ba_vs_r``). This module resolves the mismatch at factory
time by classifying each feature into one of four patterns and building
a closure that strips the prefix and applies platoon resolution based on
the current matchup handedness.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import xgboost as xgb

from sim.infra.artifact_loaders import ModelBundle
from sim.engine.core.state import GameState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pitch types (lowercase) — mirrors training/config.py::PITCH_TYPES
# ---------------------------------------------------------------------------

_PT_LOWER: list[str] = [
    "ff", "si", "fc", "sl", "cu", "kc", "sv", "st", "ch", "fs",
]

# ---------------------------------------------------------------------------
# Game state → feature extractors
# ---------------------------------------------------------------------------

_STATE_EXTRACTORS: dict[str, Callable[[GameState], float]] = {
    "inning": lambda s: float(s.inning),
    "is_bottom": lambda s: float(s.half),
    "outs": lambda s: float(s.outs),
    "runner_1b": lambda s: float((s.bases >> 0) & 1),
    "runner_2b": lambda s: float((s.bases >> 1) & 1),
    "runner_3b": lambda s: float((s.bases >> 2) & 1),
    "run_diff": lambda s: float(
        (s.home_score - s.away_score) if s.half == 1
        else (s.away_score - s.home_score)
    ),
    "is_home": lambda s: float(s.half),
}

# ---------------------------------------------------------------------------
# Dynamic feature names — game state + in-game context that changes per PA.
# Everything NOT in this set is static per (batter, pitcher) matchup.
# ---------------------------------------------------------------------------

_DYNAMIC_FEATURES: set[str] = {
    # Game state
    *_STATE_EXTRACTORS.keys(),
    # In-game context (assembled by engine._build_context)
    "times_through_order",
    "batter_prior_pa",
    "pitcher_bf_game", "batter_ab_vs_pitcher",
    # Pitcher outing counters
    "pitcher_outing_walks", "pitcher_outing_hits",
    "pitcher_outing_k", "pitcher_outing_runs",
    "pitcher_outing_whip", "pitcher_recent_whip",
}

# ---------------------------------------------------------------------------
# Non-platoon features — these exist in profiles WITHOUT a _vs_{hand} suffix.
# Everything else gets platoon resolution.
# ---------------------------------------------------------------------------

_NO_PLATOON: set[str] = {
    # Batter aggregates
    "s_pa", "c_pa",
    # Pitcher aggregates
    "s_bf", "c_bf",
    # Pitcher physical: velocity, spin, delivery (not split by hand)
    *(f"season_{pt}_velo" for pt in _PT_LOWER),
    *(f"career_{pt}_velo" for pt in _PT_LOWER),
    *(f"season_{pt}_spin" for pt in _PT_LOWER),
    *(f"career_{pt}_spin" for pt in _PT_LOWER),
    "season_arm_angle", "career_arm_angle",
    "season_extension", "career_extension",
}


# ---------------------------------------------------------------------------
# Resolver factories
# ---------------------------------------------------------------------------


def _make_direct_resolver(source: str, key: str) -> Callable:
    """Look up ``key`` directly from batter or pitcher dict (no platoon)."""
    if source == "batter":
        def resolve(b, p, s, c):
            val = b.get(key)
            return float(val) if val is not None else 0.0
    else:
        def resolve(b, p, s, c):
            val = p.get(key)
            return float(val) if val is not None else 0.0
    return resolve


def _resolve_hand(batter_hand: str | None, pitcher_hand: str | None, for_pitcher: bool) -> str:
    """Determine the opponent hand suffix for platoon resolution.

    For batter features: use pitcher's throwing hand.
    For pitcher features: use batter's batting side, resolving switch
    hitters (S) to the opposite of the pitcher's hand.
    """
    if for_pitcher:
        # Pitcher features keyed by batter hand
        bh = (batter_hand or "R").upper()
        if bh == "S":
            # Switch hitter bats opposite of pitcher hand
            ph = (pitcher_hand or "R").upper()
            return "l" if ph == "R" else "r"
        return bh.lower()
    else:
        # Batter features keyed by pitcher hand
        return (pitcher_hand or "R").lower()


def _make_platoon_resolver(
    source: str, base: str, warned: set[str]
) -> Callable:
    """Look up ``{base}_vs_{opponent_hand}`` from the source dict."""
    if source == "batter":
        def resolve(b, p, s, c):
            hand = _resolve_hand(None, p.get("hand"), for_pitcher=False)
            key = f"{base}_vs_{hand}"
            val = b.get(key)
            if val is not None:
                return float(val)
            if key not in warned:
                warned.add(key)
                logger.warning("Platoon feature %r not found in batter dict — defaulting to 0.0", key)
            return 0.0
    else:
        def resolve(b, p, s, c):
            hand = _resolve_hand(b.get("hand"), p.get("hand"), for_pitcher=True)
            key = f"{base}_vs_{hand}"
            val = p.get(key)
            if val is not None:
                return float(val)
            if key not in warned:
                warned.add(key)
                logger.warning("Platoon feature %r not found in pitcher dict — defaulting to 0.0", key)
            return 0.0
    return resolve


def _make_context_resolver(name: str, warned: set[str]) -> Callable:
    """Look up ``name`` from the context dict."""
    def resolve(b, p, s, c):
        val = c.get(name)
        if val is not None:
            return float(val)
        if name not in warned:
            warned.add(name)
            logger.warning("Context feature %r not found — defaulting to 0.0", name)
        return 0.0
    return resolve


# ---------------------------------------------------------------------------
# Resolver classification
# ---------------------------------------------------------------------------


def _build_resolvers(feature_names: list[str]) -> list[Callable]:
    """Build resolver closures for all features (same as before)."""
    resolvers: list[Callable] = []
    warned: set[str] = set()

    for name in feature_names:
        if name in _STATE_EXTRACTORS:
            extractor = _STATE_EXTRACTORS[name]
            resolvers.append(
                lambda b, p, s, c, _e=extractor: _e(s)
            )
        elif name.startswith("bat_"):
            base = name[4:]
            if base in _NO_PLATOON:
                resolvers.append(_make_direct_resolver("batter", base))
            else:
                resolvers.append(_make_platoon_resolver("batter", base, warned))
        elif name.startswith("pit_"):
            base = name[4:]
            if base in _NO_PLATOON:
                resolvers.append(_make_direct_resolver("pitcher", base))
            else:
                resolvers.append(_make_platoon_resolver("pitcher", base, warned))
        else:
            resolvers.append(_make_context_resolver(name, warned))

    return resolvers


# ---------------------------------------------------------------------------
# Isotonic calibration — shared by all inference paths
# ---------------------------------------------------------------------------


def _compile_calibration(
    cal_tables: dict, n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-class isotonic curves into contiguous arrays.

    Returns (flat_x, flat_y, offsets) where offsets[k]:offsets[k+1]
    slice the curve for class k.
    """
    xs, ys = [], []
    offsets = np.empty(n_classes + 1, dtype=np.int32)
    offsets[0] = 0
    for k in range(n_classes):
        x, y = cal_tables[k]["x"], cal_tables[k]["y"]
        xs.append(x)
        ys.append(y)
        offsets[k + 1] = offsets[k] + len(x)
    return np.concatenate(xs), np.concatenate(ys), offsets


def _calibrate(
    raw_probs: np.ndarray,
    flat_x: np.ndarray,
    flat_y: np.ndarray,
    offsets: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Apply isotonic calibration and normalize.

    Works for both scalar (1, n_classes) and batch (N, n_classes) inputs.
    """
    calibrated = np.empty_like(raw_probs, dtype=np.float64)
    for k in range(n_classes):
        lo, hi = offsets[k], offsets[k + 1]
        calibrated[..., k] = np.interp(
            raw_probs[..., k], flat_x[lo:hi], flat_y[lo:hi],
        )
    totals = calibrated.sum(axis=-1, keepdims=True)
    calibrated /= np.where(totals > 0, totals, 1.0)
    return calibrated


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_prob_source(bundle: ModelBundle) -> Callable:
    """Create a prob_source from a ModelBundle.

    Prefers the decomposed path (native booster + isotonic calibration) if
    available. Falls back to legacy sklearn predict_proba.

    In both paths, static features (batter/pitcher profiles) are computed
    once per matchup and cached. Only dynamic features (game state + context)
    are recomputed per PA.
    """
    feature_names = bundle.feature_names
    n_features = len(feature_names)
    resolvers = _build_resolvers(feature_names)

    # Classify features as static or dynamic
    static_indices: list[int] = []
    static_resolvers: list[Callable] = []
    dynamic_indices: list[int] = []
    dynamic_resolvers: list[Callable] = []

    for i, name in enumerate(feature_names):
        if name in _DYNAMIC_FEATURES:
            dynamic_indices.append(i)
            dynamic_resolvers.append(resolvers[i])
        else:
            static_indices.append(i)
            static_resolvers.append(resolvers[i])

    static_idx_arr = np.array(static_indices, dtype=np.intp)
    dynamic_idx_arr = np.array(dynamic_indices, dtype=np.intp)
    n_static = len(static_indices)
    n_dynamic = len(dynamic_indices)

    logger.info(
        "Built %d feature resolvers (%d static cached, %d dynamic per-PA)",
        n_features, n_static, n_dynamic,
    )

    # Matchup cache: (batter_id, pitcher_id, batter_hand, pitcher_hand) → static vector
    _cache: dict[tuple, np.ndarray] = {}

    # Helper: assemble feature vector (shared by all paths)
    def _assemble_features(batter, pitcher, state, context):
        cache_key = (batter["player_id"], pitcher["player_id"],
                     batter.get("hand"), pitcher.get("hand"))
        if cache_key not in _cache:
            vec = np.empty(n_static, dtype=np.float32)
            for j, resolver in enumerate(static_resolvers):
                vec[j] = resolver(batter, pitcher, state, context)
            _cache[cache_key] = vec

        X = np.empty((1, n_features), dtype=np.float32)
        X[0, static_idx_arr] = _cache[cache_key]
        for j, resolver in enumerate(dynamic_resolvers):
            X[0, dynamic_idx_arr[j]] = resolver(batter, pitcher, state, context)
        return X

    # Pre-compile calibration tables (shared by ONNX and booster paths)
    cal_tables = bundle.calibration_tables
    n_classes = len(bundle.outcome_labels)
    if cal_tables is not None:
        flat_x, flat_y, cal_offsets = _compile_calibration(cal_tables, n_classes)

    if bundle.onnx_session is not None:
        # --- ONNX path (fastest): no DMatrix overhead ---
        session = bundle.onnx_session
        input_name = session.get_inputs()[0].name

        logger.info("Using ONNX inference (onnxruntime + isotonic calibration)")

        def prob_source(
            batter: dict,
            pitcher: dict,
            state: GameState,
            context: dict,
        ) -> list[float]:
            X = _assemble_features(batter, pitcher, state, context)
            outputs = session.run(None, {input_name: X})
            calibrated = _calibrate(outputs[1], flat_x, flat_y, cal_offsets, n_classes)
            return calibrated[0].tolist()

    elif bundle.booster is not None:
        # --- Native booster path: DMatrix overhead but no sklearn ---
        booster = bundle.booster

        logger.info("Using decomposed inference (native booster + isotonic calibration)")

        def prob_source(
            batter: dict,
            pitcher: dict,
            state: GameState,
            context: dict,
        ) -> list[float]:
            X = _assemble_features(batter, pitcher, state, context)
            dmatrix = xgb.DMatrix(X, feature_names=feature_names)
            raw_probs = booster.predict(dmatrix)  # (1, n_classes)
            calibrated = _calibrate(raw_probs, flat_x, flat_y, cal_offsets, n_classes)
            return calibrated[0].tolist()

    else:
        # --- Legacy path: sklearn CalibratedClassifierCV ---
        model = bundle.model

        logger.info("Using legacy inference (sklearn CalibratedClassifierCV)")

        def prob_source(
            batter: dict,
            pitcher: dict,
            state: GameState,
            context: dict,
        ) -> list[float]:
            X = _assemble_features(batter, pitcher, state, context)
            proba = model.predict_proba(X)
            return proba[0].tolist()

    return prob_source


# ---------------------------------------------------------------------------
# Batch prob source (vectorized engine)
# ---------------------------------------------------------------------------


def make_batch_prob_source(bundle: ModelBundle) -> Callable:
    """Create a batch prob_source for the vectorized engine.

    Returns a callable:
        (unique_batter_dicts, unique_pitcher_dicts, group_idx, dynamic_arrays, n_active)
        → (n_active, n_classes) ndarray

    The caller groups sims by unique matchup and passes only the unique
    batter/pitcher dicts plus a group_idx array mapping each sim to its
    matchup group. Static features are cached per matchup. Typically 1-5
    unique matchups per PA step vs 1000 sims.

    Only supports ONNX path (the batch engine requires it).
    """
    if bundle.onnx_session is None:
        raise ValueError("Batch prob source requires ONNX model")

    feature_names = bundle.feature_names
    n_features = len(feature_names)
    resolvers = _build_resolvers(feature_names)

    # Classify features
    static_indices: list[int] = []
    static_resolvers: list[Callable] = []
    dynamic_names: list[str] = []
    dynamic_indices: list[int] = []

    for i, name in enumerate(feature_names):
        if name in _DYNAMIC_FEATURES:
            dynamic_indices.append(i)
            dynamic_names.append(name)
        else:
            static_indices.append(i)
            static_resolvers.append(resolvers[i])

    static_idx_arr = np.array(static_indices, dtype=np.intp)
    dynamic_idx_arr = np.array(dynamic_indices, dtype=np.intp)

    # ONNX session
    session = bundle.onnx_session
    input_name = session.get_inputs()[0].name
    n_classes = len(bundle.outcome_labels)
    flat_x, flat_y, cal_offsets = _compile_calibration(
        bundle.calibration_tables, n_classes,
    )

    # Shared matchup cache: (batter_id, pitcher_id, batter_hand, pitcher_hand) → static feature vec
    _cache: dict[tuple, np.ndarray] = {}

    def _ensure_cached(batter: dict, pitcher: dict) -> tuple:
        cache_key = (batter["player_id"], pitcher["player_id"],
                     batter.get("hand"), pitcher.get("hand"))
        if cache_key not in _cache:
            vec = np.empty(len(static_indices), dtype=np.float32)
            for j, resolver in enumerate(static_resolvers):
                vec[j] = resolver(batter, pitcher, None, None)
            _cache[cache_key] = vec
        return cache_key

    logger.info(
        "Built batch prob source: %d features (%d static, %d dynamic), ONNX",
        n_features, len(static_indices), len(dynamic_indices),
    )

    def batch_prob_source(
        unique_batter_dicts: list[dict],
        unique_pitcher_dicts: list[dict],
        group_idx: np.ndarray,
        dynamic_arrays: dict[str, np.ndarray],
        n_active: int,
    ) -> np.ndarray:
        """Run batched inference for N active sims.

        Args:
            unique_batter_dicts: One dict per unique matchup (typically 1-5).
            unique_pitcher_dicts: One dict per unique matchup.
            group_idx: (n_active,) int array mapping each sim to its unique matchup index.
            dynamic_arrays: Dict of (n_active,) float32 arrays for dynamic features.
            n_active: Number of active sims.

        Returns (n_active, n_classes) calibrated probability matrix.
        """
        n_unique = len(unique_batter_dicts)

        # Ensure cache populated for each unique matchup (1-5 iterations)
        unique_cache_keys = [
            _ensure_cached(unique_batter_dicts[g], unique_pitcher_dicts[g])
            for g in range(n_unique)
        ]

        # Build static feature matrix for unique matchups: (n_unique, n_static)
        static_matrix = np.array(
            [_cache[ck] for ck in unique_cache_keys], dtype=np.float32,
        )

        # Broadcast to all sims via group_idx — one fancy-index op
        X = np.empty((n_active, n_features), dtype=np.float32)
        X[:, static_idx_arr] = static_matrix[group_idx]

        # Fill dynamic features from arrays
        for k, name in enumerate(dynamic_names):
            X[:, dynamic_idx_arr[k]] = dynamic_arrays[name]

        # Single ONNX call for all active sims
        outputs = session.run(None, {input_name: X})
        raw_probs = outputs[1]  # (n_active, n_classes)

        return _calibrate(raw_probs, flat_x, flat_y, cal_offsets, n_classes)

    return batch_prob_source
