"""Build calibration artifacts from DuckLake eval tables.

Each builder queries sim_eval_* tables for a specific eval_id and
produces a standalone artifact dict. Four artifacts:
  - n_lookup: per-state sim count (adaptive N)
  - stopping_thresholds: per-state SE target (SMC stopping)
  - gamma_schedule: per-state tempering (SMC resampling)
  - horizon_weights: per-state half-inning cutoff (truncated MC)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from sim.engine.estimators.buckets import bucket_to_rd

logger = logging.getLogger(__name__)


def _prefix(conn) -> str:
    """Detect DuckLake schema prefix (lakehouse.main, lh.main, or main)."""
    for prefix in ("lakehouse.main", "lh.main"):
        try:
            conn.execute(f"SELECT 1 FROM {prefix}.sim_eval LIMIT 0")
            return prefix
        except Exception:
            continue
    return "main"


# ---------------------------------------------------------------------------
# n_lookup — per-state sim count
# ---------------------------------------------------------------------------


def build_n_lookup(
    conn,
    eval_id: str,
    *,
    we_array: np.ndarray | None = None,
    target_se: float = 0.02,
) -> dict:
    """Build per-state N allocation from convergence data.

    For each (inning, rd_bucket), picks the level with the lowest
    stabilization_n, then floors by CI-based minimum N.
    """
    pfx = _prefix(conn)
    rows = conn.execute(f"""
        SELECT inning, rd_bucket, level, stabilization_n,
               std_at_25, std_at_50, std_at_100, std_at_200, std_at_500
        FROM {pfx}.sim_eval_convergence
        WHERE eval_id = ?
    """, [eval_id]).fetchall()

    # Group by state_key, pick best (lowest stabilization_n) level
    best: dict[str, tuple[int, float | None]] = {}  # state_key → (stab_n, std_at_stab)
    for inning, rd_bucket, level, stab_n, *stds in rows:
        if stab_n is None:
            continue
        sk = f"{inning}|{rd_bucket}"
        if sk not in best or stab_n < best[sk][0]:
            # Find std at stabilization_n
            std_map = dict(zip([25, 50, 100, 200, 500], stds))
            std_at_stab = std_map.get(stab_n) or std_map.get(500)
            best[sk] = (stab_n, std_at_stab)

    lookup: dict[str, int] = {}
    for sk, (stab_n, _) in best.items():
        n = stab_n
        ci_floor = _ci_floor(sk, we_array, target_se)
        if ci_floor is not None:
            n = max(n, ci_floor)
        lookup[sk] = n

    n_values = list(lookup.values())
    return {
        "metadata": {
            "eval_id": eval_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_states": len(lookup),
            "target_se": target_se,
            "n_min": min(n_values) if n_values else 0,
            "n_max": max(n_values) if n_values else 0,
        },
        "lookup": lookup,
    }


# ---------------------------------------------------------------------------
# stopping_thresholds — per-state SE target
# ---------------------------------------------------------------------------


def build_stopping_thresholds(conn, eval_id: str) -> dict:
    """Build per-state SE stopping thresholds from convergence data.

    Uses the std at stabilization_n for the best level per state.
    """
    pfx = _prefix(conn)
    rows = conn.execute(f"""
        SELECT inning, rd_bucket, level, stabilization_n,
               std_at_25, std_at_50, std_at_100, std_at_200, std_at_500
        FROM {pfx}.sim_eval_convergence
        WHERE eval_id = ?
    """, [eval_id]).fetchall()

    best: dict[str, float] = {}
    best_stab: dict[str, int] = {}
    for inning, rd_bucket, level, stab_n, *stds in rows:
        if stab_n is None:
            continue
        sk = f"{inning}|{rd_bucket}"
        if sk not in best_stab or stab_n < best_stab[sk]:
            best_stab[sk] = stab_n
            std_map = dict(zip([25, 50, 100, 200, 500], stds))
            std_val = std_map.get(stab_n) or std_map.get(500)
            if std_val is not None:
                best[sk] = round(std_val, 6)

    return {
        "metadata": {
            "eval_id": eval_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_states": len(best),
        },
        "lookup": best,
    }


# ---------------------------------------------------------------------------
# gamma_schedule — per-state SMC tempering
# ---------------------------------------------------------------------------


def build_gamma_schedule(
    conn,
    eval_id: str,
    *,
    sensitivity_array: np.ndarray | None = None,
) -> dict:
    """Build per-state gamma from sim spread vs WE sensitivity.

    γ = clamp(sim_std / sensitivity, 0.1, 0.5).
    Falls back to 1.0 if sensitivity_array not provided.
    """
    pfx = _prefix(conn)
    rows = conn.execute(f"""
        SELECT inning, rd_bucket, level, stabilization_n,
               std_at_25, std_at_50, std_at_100, std_at_200, std_at_500
        FROM {pfx}.sim_eval_convergence
        WHERE eval_id = ?
    """, [eval_id]).fetchall()

    # Get sim_std per state (std at stabilization_n for best level)
    sim_stds: dict[str, float] = {}
    best_stab: dict[str, int] = {}
    for inning, rd_bucket, level, stab_n, *stds in rows:
        if stab_n is None:
            continue
        sk = f"{inning}|{rd_bucket}"
        if sk not in best_stab or stab_n < best_stab[sk]:
            best_stab[sk] = stab_n
            std_map = dict(zip([25, 50, 100, 200, 500], stds))
            std_val = std_map.get(stab_n) or std_map.get(500)
            if std_val is not None:
                sim_stds[sk] = std_val

    gamma: dict[str, float] = {}
    if sensitivity_array is None or not sim_stds:
        gamma = {k: 1.0 for k in sim_stds}
    else:
        for sk, sim_std in sim_stds.items():
            parts = sk.split("|", 1)
            if len(parts) != 2:
                gamma[sk] = 1.0
                continue
            inning = int(parts[0])
            rd = bucket_to_rd(parts[1])
            inn_idx = min(inning, sensitivity_array.shape[0] - 1)
            rd_idx = int(np.clip(rd + 15, 0, sensitivity_array.shape[2] - 1))
            sens = float(np.mean(np.abs(sensitivity_array[inn_idx, :, rd_idx])))
            if sens > 1e-6:
                raw = sim_std / sens
                gamma[sk] = round(float(np.clip(raw, 0.1, 0.5)), 4)
            else:
                gamma[sk] = 1.0

    return {
        "metadata": {
            "eval_id": eval_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_states": len(gamma),
        },
        "lookup": gamma,
    }


# ---------------------------------------------------------------------------
# horizon_weights — per-state per-horizon weight schedule
# ---------------------------------------------------------------------------


def build_horizon_weights(
    conn,
    eval_id: str,
    *,
    min_games: int = 30,
    max_horizons: int = 15,
) -> dict:
    """Build per-state horizon weight schedule using error covariance.

    For each state bucket, computes the covariance matrix of prediction
    errors (pred_we - actual_outcome) across horizons, inverts it, and
    extracts minimum-variance weights: w ∝ Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1).

    This is the Markowitz minimum-variance portfolio applied to WE
    estimates at different horizons. Accounts for correlated errors
    between adjacent horizons (which share the same game trajectory).

    Only includes horizons where the sim's Brier score beats the entry
    WE baseline (improvement gating).
    """
    pfx = _prefix(conn)

    rows = conn.execute(f"""
        SELECT
            g.entry_inning, g.entry_rd_bucket,
            g.game_pk, g.actual_home_win, g.entry_we_baseline,
            h.horizon, h.pred_we
        FROM {pfx}.sim_eval_horizons h
        JOIN {pfx}.sim_eval_games g
            ON h.eval_id = g.eval_id AND h.game_pk = g.game_pk
        WHERE h.eval_id = ?
          AND h.horizon LIKE '+%%hi'
          AND h.pred_we IS NOT NULL
          AND g.actual_home_win IS NOT NULL
          AND g.entry_we_baseline IS NOT NULL
    """, [eval_id]).fetchall()

    from collections import defaultdict

    # Group: state_key → game_pk → {horizon: pred_we, outcome, entry_we}
    games_by_state: dict[str, dict[int, dict]] = defaultdict(dict)
    for inning, rd_bucket, game_pk, outcome, entry_we, horizon, pred_we in rows:
        if inning is None or rd_bucket is None:
            continue
        sk = f"{inning}|{rd_bucket}"
        if game_pk not in games_by_state[sk]:
            games_by_state[sk][game_pk] = {
                "outcome": float(outcome),
                "entry_we": float(entry_we),
                "horizons": {},
            }
        games_by_state[sk][game_pk]["horizons"][horizon] = float(pred_we)

    def _hz_num(k: str) -> int:
        return int(k[1:-2])

    lookup: dict[str, dict[str, float]] = {}
    for sk, game_data in games_by_state.items():
        if len(game_data) < min_games:
            continue

        # Find horizons present in enough games
        all_hz: set[str] = set()
        for gd in game_data.values():
            all_hz.update(gd["horizons"].keys())
        hi_keys = sorted(
            [k for k in all_hz if k.endswith("hi")],
            key=_hz_num,
        )[:max_horizons]

        # Filter to horizons where sim beats entry WE (Brier gating)
        entry_brier = None
        valid_hz = []
        for hz in hi_keys:
            games_with_hz = [
                gd for gd in game_data.values()
                if hz in gd["horizons"]
            ]
            if len(games_with_hz) < min_games:
                continue

            hz_brier = sum(
                (gd["horizons"][hz] - gd["outcome"]) ** 2
                for gd in games_with_hz
            ) / len(games_with_hz)

            if entry_brier is None:
                entry_brier = sum(
                    (gd["entry_we"] - gd["outcome"]) ** 2
                    for gd in games_with_hz
                ) / len(games_with_hz)

            if hz_brier < entry_brier:
                valid_hz.append(hz)

        if not valid_hz:
            continue

        # Build error matrix: (n_games × n_horizons)
        # Only games that have all valid horizons
        complete_games = [
            gd for gd in game_data.values()
            if all(hz in gd["horizons"] for hz in valid_hz)
        ]
        if len(complete_games) < min_games:
            continue

        n = len(complete_games)
        h = len(valid_hz)
        errors = np.zeros((n, h))
        for i, gd in enumerate(complete_games):
            for j, hz in enumerate(valid_hz):
                errors[i, j] = gd["horizons"][hz] - gd["outcome"]

        # Covariance matrix of errors — Ledoit-Wolf shrinkage for stability
        # with small samples and near-singular matrices
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(errors).covariance_
        if cov.ndim == 0:
            # Single horizon
            cov = np.array([[float(cov)]])

        # Invert — use pseudo-inverse for numerical stability
        try:
            cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            continue

        # Minimum-variance weights: w = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1)
        ones = np.ones(h)
        raw_w = cov_inv @ ones
        denom = ones @ raw_w
        if abs(denom) < 1e-12:
            continue
        w = raw_w / denom

        # Clamp negatives to 0 and renormalize
        w = np.maximum(w, 0.0)
        w_sum = w.sum()
        if w_sum < 1e-12:
            continue
        w = w / w_sum

        weights = {hz: round(float(w[j]), 4) for j, hz in enumerate(valid_hz) if w[j] > 0.001}
        if weights:
            lookup[sk] = weights

    return {
        "metadata": {
            "eval_id": eval_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "min_games": min_games,
            "n_states": len(lookup),
        },
        "lookup": lookup,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ci_floor(
    state_key: str, we_array: np.ndarray | None, target_se: float,
) -> int | None:
    """Minimum N for target SE: N >= p*(1-p) / target_se²."""
    if we_array is None or target_se <= 0:
        return None
    parts = state_key.split("|", 1)
    if len(parts) != 2:
        return None
    inning = int(parts[0])
    rd = bucket_to_rd(parts[1])
    inn_idx = min(inning, we_array.shape[0] - 1)
    rd_idx = int(np.clip(rd + 15, 0, we_array.shape[4] - 1))
    p = float(we_array[inn_idx, 0, 0, 0, rd_idx])
    variance = p * (1 - p)
    return int(np.ceil(variance / (target_se ** 2)))
