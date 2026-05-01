"""Vectorized game simulation engine — runs N sims in parallel arrays.

Instead of looping 1000 times through simulate_game(), this module runs
all sims simultaneously at each PA step using NumPy arrays. The parallelism
axis is across simulations at the same PA step.

The scalar engine (engine.py) is preserved for testing and single-sim use.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

import numpy as np

from sim.engine.core.engine import (
    GameResult,
    GameTimings,
    _WALK_OUTCOMES,
    _HIT_OUTCOMES,
    _K_OUTCOMES,
    _STARTER_BF_THRESHOLD,
    _RELIEVER_BF_THRESHOLD,
    _BASE_PULL_PROB,
)
from sim.game_inputs.game import GameInput
from sim.engine.core.state import GameState
from sim.engine.lookups.re_table import RE_TABLE

logger = logging.getLogger(__name__)

# Outcome classification sets
_WALK_SET: frozenset[str] = _WALK_OUTCOMES
_HIT_SET: frozenset[str] = _HIT_OUTCOMES
_K_SET: frozenset[str] = _K_OUTCOMES


# ---------------------------------------------------------------------------
# Batch state
# ---------------------------------------------------------------------------


class HorizonData:
    """State snapshots at +1 PA and every half-inning boundary.

    Uniform 6-field format: inning, half, outs, bases, home_score, away_score.
    The engine records raw per-sim data; horizon semantics and aggregation
    are handled by the eval layer.
    """

    def __init__(self, n_sims: int, max_halves: int = 40):
        # +1 PA snapshot (filled once at pa_step == 1)
        self.pa1_inning = np.zeros(n_sims, dtype=np.int16)
        self.pa1_half = np.zeros(n_sims, dtype=np.int8)
        self.pa1_outs = np.zeros(n_sims, dtype=np.int8)
        self.pa1_bases = np.zeros(n_sims, dtype=np.int8)
        self.pa1_home_score = np.zeros(n_sims, dtype=np.int16)
        self.pa1_away_score = np.zeros(n_sims, dtype=np.int16)
        self.pa1_active = np.zeros(n_sims, dtype=bool)
        # Half-inning boundary snapshots (filled on each half change)
        self.hi_inning = np.zeros((n_sims, max_halves), dtype=np.int16)
        self.hi_half = np.zeros((n_sims, max_halves), dtype=np.int8)
        self.hi_outs = np.zeros((n_sims, max_halves), dtype=np.int8)
        self.hi_bases = np.zeros((n_sims, max_halves), dtype=np.int8)
        self.hi_home_score = np.zeros((n_sims, max_halves), dtype=np.int16)
        self.hi_away_score = np.zeros((n_sims, max_halves), dtype=np.int16)
        self.hi_count = np.zeros(n_sims, dtype=np.int8)
        # WE at each half-inning boundary — MLMC level values
        self.hi_we = np.full((n_sims, max_halves), -1.0, dtype=np.float64)


@dataclass
class BatchState:
    """Parallel game state for N simulations. All fields are (N,) arrays."""

    n: int
    inning: np.ndarray
    half: np.ndarray
    outs: np.ndarray
    bases: np.ndarray
    away_score: np.ndarray
    home_score: np.ndarray
    away_batter_idx: np.ndarray
    home_batter_idx: np.ndarray
    away_pitcher_bf: np.ndarray
    home_pitcher_bf: np.ndarray
    away_pitcher_runs: np.ndarray
    home_pitcher_runs: np.ndarray
    away_pitcher_tto: np.ndarray
    home_pitcher_tto: np.ndarray
    away_starter_pulled: np.ndarray
    home_starter_pulled: np.ndarray
    active: np.ndarray
    pruned: np.ndarray
    pruned_at_pa: np.ndarray


@dataclass
class BatchOutings:
    """Per-sim pitcher outing counters. All fields are (N,) arrays."""

    home_walks: np.ndarray
    home_hits: np.ndarray
    home_k: np.ndarray
    home_runs: np.ndarray
    away_walks: np.ndarray
    away_hits: np.ndarray
    away_k: np.ndarray
    away_runs: np.ndarray
    # Ring buffer for recent outcomes (for recent_whip)
    # (N, 9) int8 — stores outcome index, -1 = empty
    home_recent: np.ndarray
    home_recent_ptr: np.ndarray  # (N,) int8 — next write position
    away_recent: np.ndarray
    away_recent_ptr: np.ndarray


def batch_from_scalar(state: GameState, n: int) -> BatchState:
    """Broadcast a scalar GameState into N parallel copies."""
    return BatchState(
        n=n,
        inning=np.full(n, state.inning, dtype=np.int16),
        half=np.full(n, state.half, dtype=np.int8),
        outs=np.full(n, state.outs, dtype=np.int8),
        bases=np.full(n, state.bases, dtype=np.int8),
        away_score=np.full(n, state.away_score, dtype=np.int16),
        home_score=np.full(n, state.home_score, dtype=np.int16),
        away_batter_idx=np.full(n, state.away_batter_idx, dtype=np.int8),
        home_batter_idx=np.full(n, state.home_batter_idx, dtype=np.int8),
        away_pitcher_bf=np.full(n, state.away_pitcher_bf, dtype=np.int16),
        home_pitcher_bf=np.full(n, state.home_pitcher_bf, dtype=np.int16),
        away_pitcher_runs=np.full(n, state.away_pitcher_runs, dtype=np.int16),
        home_pitcher_runs=np.full(n, state.home_pitcher_runs, dtype=np.int16),
        away_pitcher_tto=np.full(n, state.away_pitcher_tto, dtype=np.int8),
        home_pitcher_tto=np.full(n, state.home_pitcher_tto, dtype=np.int8),
        away_starter_pulled=np.full(n, state.away_starter_pulled, dtype=bool),
        home_starter_pulled=np.full(n, state.home_starter_pulled, dtype=bool),
        active=np.ones(n, dtype=bool),
        pruned=np.zeros(n, dtype=bool),
        pruned_at_pa=np.zeros(n, dtype=np.int16),
    )


def _make_batch_outings(n: int) -> BatchOutings:
    """Create zeroed outing counters for N sims."""
    return BatchOutings(
        home_walks=np.zeros(n, dtype=np.int16),
        home_hits=np.zeros(n, dtype=np.int16),
        home_k=np.zeros(n, dtype=np.int16),
        home_runs=np.zeros(n, dtype=np.int16),
        away_walks=np.zeros(n, dtype=np.int16),
        away_hits=np.zeros(n, dtype=np.int16),
        away_k=np.zeros(n, dtype=np.int16),
        away_runs=np.zeros(n, dtype=np.int16),
        home_recent=np.full((n, 9), -1, dtype=np.int8),
        home_recent_ptr=np.zeros(n, dtype=np.int8),
        away_recent=np.full((n, 9), -1, dtype=np.int8),
        away_recent_ptr=np.zeros(n, dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# Vectorized game_over
# ---------------------------------------------------------------------------


def batch_game_over(bs: BatchState) -> np.ndarray:
    """Return bool mask of sims that have ended."""
    late = bs.inning >= 9

    # Walk-off: bottom 9+, home leads
    walkoff = late & (bs.half == 1) & (bs.home_score > bs.away_score)

    # Full inning complete (9+ just ended): top of next, scores differ
    complete = (late & (bs.half == 0) & (bs.inning > 9)
                & (bs.home_score != bs.away_score) & (bs.outs == 0))

    return walkoff | complete


# ---------------------------------------------------------------------------
# Vectorized outcome sampling
# ---------------------------------------------------------------------------


def _batch_sample_outcomes(
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample outcome indices from (N, n_classes) probability matrix."""
    cum_probs = np.cumsum(probs, axis=1)
    rolls = rng.random(probs.shape[0])
    return (rolls[:, None] < cum_probs).argmax(axis=1).astype(np.int16)


# ---------------------------------------------------------------------------
# Vectorized baserunning resolution (compiled table)
# ---------------------------------------------------------------------------


def _batch_resolve_outcomes(
    outcome_indices: np.ndarray,
    bases: np.ndarray,
    outs: np.ndarray,
    compiled,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve baserunning for all sims using the compiled lookup table.

    Returns (post_bases, runs_scored, outs_added, valid_mask).
    """
    key_idx = (outcome_indices.astype(np.int32) * 24
               + bases.astype(np.int32) * 3
               + outs.astype(np.int32))

    # Clamp to valid key range
    np.clip(key_idx, 0, compiled.n_keys - 1, out=key_idx)

    n_t = compiled.n_transitions[key_idx]
    valid = n_t > 0

    # Sample transition index: argmax on (roll < cum_probs) per row
    rolls = rng.random(len(key_idx))
    cp_rows = compiled.cum_probs[key_idx]  # (N, max_t)
    t_idx = (rolls[:, None] < cp_rows).argmax(axis=1)
    t_idx = np.minimum(t_idx, np.maximum(n_t - 1, 0))

    post_bases = compiled.post_bases[key_idx, t_idx]
    runs_scored = compiled.runs_scored[key_idx, t_idx]
    outs_added = compiled.outs_added[key_idx, t_idx]

    return post_bases, runs_scored, outs_added, valid


# ---------------------------------------------------------------------------
# Vectorized state transitions
# ---------------------------------------------------------------------------


def _update_score_and_pitcher_bf(
    bs: BatchState, idx: np.ndarray, half_mask: np.ndarray, runs: np.ndarray,
):
    """Update score and pitcher BF/runs for a subset of sims.

    half_mask: bool array (same length as idx) where True = top half (away batting).
    """
    top = half_mask
    bot = ~half_mask
    if top.any():
        idx_top = idx[top]
        bs.away_score[idx_top] += runs[top]
        bs.home_pitcher_bf[idx_top] += 1
        bs.home_pitcher_runs[idx_top] += runs[top]
    if bot.any():
        idx_bot = idx[bot]
        bs.home_score[idx_bot] += runs[bot]
        bs.away_pitcher_bf[idx_bot] += 1
        bs.away_pitcher_runs[idx_bot] += runs[bot]


def _batch_apply_pa(
    bs: BatchState,
    active_idx: np.ndarray,
    post_bases: np.ndarray,
    runs_scored: np.ndarray,
    outs_added: np.ndarray,
):
    """Apply resolved PA results to batch state. Modifies bs in place."""
    new_outs = bs.outs[active_idx] + outs_added

    flip = new_outs >= 3
    stay = ~flip

    # --- Mid-inning updates ---
    if stay.any():
        idx_s = active_idx[stay]
        bs.bases[idx_s] = post_bases[stay]
        bs.outs[idx_s] = new_outs[stay]
        _update_score_and_pitcher_bf(bs, idx_s, bs.half[idx_s] == 0, runs_scored[stay])

    # --- Inning flip ---
    if flip.any():
        idx_f = active_idx[flip]
        _update_score_and_pitcher_bf(bs, idx_f, bs.half[idx_f] == 0, runs_scored[flip])

        # Flip half/inning
        bs.outs[idx_f] = 0
        bs.bases[idx_f] = 0
        was_top = bs.half[idx_f] == 0
        was_bot = ~was_top
        bs.half[idx_f[was_top]] = 1   # top → bottom
        bs.half[idx_f[was_bot]] = 0   # bottom → top
        bs.inning[idx_f[was_bot]] += 1  # next inning


def _batch_advance_lineup(bs: BatchState, active_idx: np.ndarray):
    """Advance batting team's lineup cursor. Modifies bs in place."""
    top = bs.half[active_idx] == 0
    bot = ~top

    if top.any():
        idx_t = active_idx[top]
        new_idx = (bs.away_batter_idx[idx_t] + 1) % 9
        wrapped = new_idx == 0
        bs.away_batter_idx[idx_t] = new_idx
        if wrapped.any():
            bs.home_pitcher_tto[idx_t[wrapped]] += 1

    if bot.any():
        idx_b = active_idx[bot]
        new_idx = (bs.home_batter_idx[idx_b] + 1) % 9
        wrapped = new_idx == 0
        bs.home_batter_idx[idx_b] = new_idx
        if wrapped.any():
            bs.away_pitcher_tto[idx_b[wrapped]] += 1


def _batch_update_outings(
    outings: BatchOutings,
    active_idx: np.ndarray,
    outcome_indices: np.ndarray,
    runs_scored: np.ndarray,
    half: np.ndarray,
    walk_indices: np.ndarray,
    hit_indices: np.ndarray,
    k_indices: np.ndarray,
):
    """Update outing counters for all active sims. Modifies outings in place."""
    is_walk = walk_indices[outcome_indices]
    is_hit = hit_indices[outcome_indices]
    is_k = k_indices[outcome_indices]

    top = half == 0  # away batting → home pitcher defending
    bot = ~top

    if top.any():
        idx_t = active_idx[top]
        outings.home_walks[idx_t] += is_walk[top].astype(np.int16)
        outings.home_hits[idx_t] += is_hit[top].astype(np.int16)
        outings.home_k[idx_t] += is_k[top].astype(np.int16)
        outings.home_runs[idx_t] += runs_scored[top]
        ptrs = outings.home_recent_ptr[idx_t]
        outings.home_recent[idx_t, ptrs] = outcome_indices[top].astype(np.int8)
        outings.home_recent_ptr[idx_t] = (ptrs + 1) % 9

    if bot.any():
        idx_b = active_idx[bot]
        outings.away_walks[idx_b] += is_walk[bot].astype(np.int16)
        outings.away_hits[idx_b] += is_hit[bot].astype(np.int16)
        outings.away_k[idx_b] += is_k[bot].astype(np.int16)
        outings.away_runs[idx_b] += runs_scored[bot]
        ptrs = outings.away_recent_ptr[idx_b]
        outings.away_recent[idx_b, ptrs] = outcome_indices[bot].astype(np.int8)
        outings.away_recent_ptr[idx_b] = (ptrs + 1) % 9


# ---------------------------------------------------------------------------
# Outing context helpers
# ---------------------------------------------------------------------------


def _batch_outing_whip(walks: np.ndarray, hits: np.ndarray, bf: np.ndarray) -> np.ndarray:
    """WHIP for current outing."""
    return np.where(bf > 0, (walks + hits).astype(np.float32) / bf, 0.0)


def _batch_recent_whip(
    recent: np.ndarray, walk_indices: np.ndarray, hit_indices: np.ndarray,
) -> np.ndarray:
    """Recent WHIP from ring buffer. Fully vectorized — no column loop."""
    valid = recent >= 0  # (N, 9)
    safe_oi = np.where(valid, recent, 0)  # safe indexing (replace -1 with 0)
    is_wh = (walk_indices[safe_oi] | hit_indices[safe_oi]) & valid  # (N, 9)
    wh = is_wh.sum(axis=1).astype(np.float32)
    n_valid = valid.sum(axis=1).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_valid > 0, wh / n_valid, 0.0)


# ---------------------------------------------------------------------------
# Pitcher exit helpers
# ---------------------------------------------------------------------------


def _reset_pitcher_outing(
    bs: BatchState, outings: BatchOutings, pull_idx: np.ndarray, defending_home: bool,
):
    """Reset pitcher state and outing counters for pulled pitchers."""
    if defending_home:
        bs.home_starter_pulled[pull_idx] = True
        bs.home_pitcher_bf[pull_idx] = 0
        bs.home_pitcher_runs[pull_idx] = 0
        bs.home_pitcher_tto[pull_idx] = 1
        outings.home_walks[pull_idx] = 0
        outings.home_hits[pull_idx] = 0
        outings.home_k[pull_idx] = 0
        outings.home_runs[pull_idx] = 0
        outings.home_recent[pull_idx] = -1
        outings.home_recent_ptr[pull_idx] = 0
    else:
        bs.away_starter_pulled[pull_idx] = True
        bs.away_pitcher_bf[pull_idx] = 0
        bs.away_pitcher_runs[pull_idx] = 0
        bs.away_pitcher_tto[pull_idx] = 1
        outings.away_walks[pull_idx] = 0
        outings.away_hits[pull_idx] = 0
        outings.away_k[pull_idx] = 0
        outings.away_runs[pull_idx] = 0
        outings.away_recent[pull_idx] = -1
        outings.away_recent_ptr[pull_idx] = 0


def _vectorized_placeholder_exit(
    bf_a: np.ndarray, sp: np.ndarray, check_active: np.ndarray,
) -> np.ndarray:
    """Vectorized placeholder pull probability. No Python loop."""
    bf_vals = bf_a[check_active].astype(np.float64)
    is_starter = ~sp[check_active]
    threshold = np.where(is_starter, _STARTER_BF_THRESHOLD, _RELIEVER_BF_THRESHOLD)
    return np.where(bf_vals >= threshold, _BASE_PULL_PROB, 0.0)


def _build_exit_features_transposed(
    pitcher_exit_model,
    bs: BatchState,
    shared: dict,
    halves: np.ndarray,
    home_prev_re: np.ndarray,
    away_prev_re: np.ndarray,
    home_pitcher_idx: np.ndarray,
    away_pitcher_idx: np.ndarray,
    home_pitcher_profiles: tuple[np.ndarray, np.ndarray],
    away_pitcher_profiles: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Build (n_features, n_active) transposed feature matrix for all active sims.

    Computes exit features once for both halves using shared_arrays directly.
    Returns float32 array ready for predict_transposed (no column_stack,
    no ascontiguousarray, no per-half split).
    """
    active_idx = shared["active_idx"]
    n = len(active_idx)

    # Shared arrays are already merged home/away — use directly
    bf = shared["def_bf"].astype(np.float32)
    tto = shared["def_tto"].astype(np.float32)
    o_walks = shared["def_walks"].astype(np.float32)
    o_hits = shared["def_hits"].astype(np.float32)
    o_k = shared["def_k"].astype(np.float32)
    o_runs = shared["def_runs"].astype(np.float32)
    outing_whip = shared["def_whip"].astype(np.float32)
    recent_whip = shared["def_recent_whip"].astype(np.float32)
    run_diff = shared["run_diff"].astype(np.float32)

    inning = bs.inning[active_idx].astype(np.float32)
    outs = bs.outs[active_idx].astype(np.float32)

    # starter_flag depends on half
    is_top = halves == 0
    starter_flag = np.where(
        is_top, ~bs.home_starter_pulled[active_idx], ~bs.away_starter_pulled[active_idx],
    ).astype(np.float32)

    # Runners / RE
    bases = bs.bases[active_idx].astype(np.int32)
    runners_on = ((bases & 1) + ((bases >> 1) & 1) + ((bases >> 2) & 1)).astype(np.float32)
    runners_bitmask = (bases & 1) + ((bases >> 1) & 1) * 2 + ((bases >> 2) & 1) * 4
    outs_clamped = np.minimum(bs.outs[active_idx], 2).astype(int)
    current_re = RE_TABLE[runners_bitmask, outs_clamped].astype(np.float32)
    prev_re = np.where(is_top, home_prev_re[active_idx], away_prev_re[active_idx]).astype(np.float32)
    re_diff = current_re - prev_re

    # Pitcher profile — depends on half
    home_avg_bf, home_rest = home_pitcher_profiles
    away_avg_bf, away_rest = away_pitcher_profiles
    home_pidx = np.clip(home_pitcher_idx[active_idx].astype(np.int32) + 1, 0, len(home_avg_bf) - 1)
    away_pidx = np.clip(away_pitcher_idx[active_idx].astype(np.int32) + 1, 0, len(away_avg_bf) - 1)
    avg_bf = np.where(is_top, home_avg_bf[home_pidx], away_avg_bf[away_pidx]).astype(np.float32)
    rest_days = np.where(is_top, home_rest[home_pidx], away_rest[away_pidx]).astype(np.float32)

    col_map = {
        "pitcher_bf_game": bf,
        "starter_flag": starter_flag,
        "outing_runs": o_runs,
        "inning": inning,
        "run_diff": run_diff,
        "outs": outs,
        "runners_on": runners_on,
        "outing_walks": o_walks,
        "outing_hits": o_hits,
        "outing_k": o_k,
        "times_through_order": tto,
        "outing_whip": outing_whip,
        "pitcher_recent_whip": recent_whip,
        "current_re": current_re,
        "re_diff": re_diff,
        "avg_bf_per_app": avg_bf,
        "pit_rest_days": rest_days,
    }

    # Stack as (n_features, n_active) — transposed for tree traversal
    return np.array([
        col_map.get(f, np.zeros(n, dtype=np.float32))
        for f in pitcher_exit_model.feature_names
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Pitcher profile lookup builders
# ---------------------------------------------------------------------------


def _build_pitcher_profile_lookup(
    starter_id: int, bp_ids: np.ndarray, player_lookup: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-build pitcher profile arrays indexed by pitcher_idx + 1.

    Index 0 = starter, 1..N = bullpen entries.
    Returns (avg_bf_per_app, rest_days) arrays.
    """
    all_ids = [int(starter_id)]
    for pid in bp_ids:
        all_ids.append(int(pid))

    n = len(all_ids)
    avg_bf = np.full(n, 20.0, dtype=np.float32)
    rest_days = np.full(n, -1.0, dtype=np.float32)

    for i, pid in enumerate(all_ids):
        p = player_lookup.get(pid)
        if p is not None:
            avg_bf[i] = float(p.get("career_avg_bf_per_app", p.get("avg_bf_per_app", 20.0)))
            rest_days[i] = float(p.get("pitcher_rest_days", p.get("pit_rest_days", -1)))

    return avg_bf, rest_days


# ---------------------------------------------------------------------------
# Defensive array builder
# ---------------------------------------------------------------------------


def _build_defensive_arrays(
    bs: BatchState, outings: BatchOutings, active_idx: np.ndarray,
    halves: np.ndarray, walk_mask: np.ndarray, hit_mask: np.ndarray,
) -> dict:
    """Build shared defensive pitcher arrays used by both exit model and PA model."""
    def_bf = np.where(halves == 0, bs.home_pitcher_bf[active_idx], bs.away_pitcher_bf[active_idx])
    def_tto = np.where(halves == 0, bs.home_pitcher_tto[active_idx], bs.away_pitcher_tto[active_idx])
    def_walks = np.where(halves == 0, outings.home_walks[active_idx], outings.away_walks[active_idx])
    def_hits = np.where(halves == 0, outings.home_hits[active_idx], outings.away_hits[active_idx])
    def_k = np.where(halves == 0, outings.home_k[active_idx], outings.away_k[active_idx])
    def_runs = np.where(halves == 0, outings.home_runs[active_idx], outings.away_runs[active_idx])

    with np.errstate(divide="ignore", invalid="ignore"):
        def_whip = _batch_outing_whip(def_walks, def_hits, def_bf)
        home_rw = _batch_recent_whip(outings.home_recent[active_idx], walk_mask, hit_mask)
        away_rw = _batch_recent_whip(outings.away_recent[active_idx], walk_mask, hit_mask)
        def_recent_whip = np.where(halves == 0, home_rw, away_rw)

    run_diff = np.where(
        halves == 1,
        bs.home_score[active_idx] - bs.away_score[active_idx],
        bs.away_score[active_idx] - bs.home_score[active_idx],
    ).astype(np.float32)

    return {
        "active_idx": active_idx,
        "def_bf": def_bf,
        "def_tto": def_tto,
        "def_walks": def_walks,
        "def_hits": def_hits,
        "def_k": def_k,
        "def_runs": def_runs,
        "def_whip": def_whip,
        "def_recent_whip": def_recent_whip,
        "run_diff": run_diff,
    }


# ---------------------------------------------------------------------------
# Batch context — collects all mutable state for the PA loop
# ---------------------------------------------------------------------------


@dataclass
class BatchContext:
    """All state needed to step through a batch simulation one PA at a time.

    Created by init_batch_context(), consumed by step_pa() and finalize_batch().
    Mutable — step_pa() modifies fields in place.
    """

    bs: BatchState
    outings: BatchOutings
    game_input: GameInput
    outcome_labels: list[str]
    rng: np.random.Generator
    n_sims: int
    max_pa: int
    manfred_runner: bool
    compiled_baserunning: object
    batch_prob_source: object
    pitcher_exit_model: object | None
    blowout_thresholds: object | None
    we_array: np.ndarray | None
    we_snapshot: np.ndarray | None
    horizon_data: HorizonData
    # Precomputed masks
    walk_mask: np.ndarray
    hit_mask: np.ndarray
    k_mask: np.ndarray
    field_out_idx: int
    # Lineup arrays
    away_lineup_ids: np.ndarray
    home_lineup_ids: np.ndarray
    home_starter_id: np.int64
    away_starter_id: np.int64
    home_bp_ids: np.ndarray
    away_bp_ids: np.ndarray
    player_lookup: dict
    # Pitcher profiles
    home_pitcher_profiles: tuple
    away_pitcher_profiles: tuple
    # Per-sim mutable tracking
    home_pitcher_idx: np.ndarray
    away_pitcher_idx: np.ndarray
    home_bullpen_pools: list
    away_bullpen_pools: list
    home_changes: np.ndarray
    away_changes: np.ndarray
    batter_pa_counts: dict
    matchup_counts: dict
    sim_outcome_counts: np.ndarray
    sim_pa_counts: np.ndarray
    # Half-inning transition tracking
    prev_inning: np.ndarray
    prev_half: np.ndarray
    home_prev_re: np.ndarray
    away_prev_re: np.ndarray
    base_re: float
    # Loop state
    pa_step: int
    last_half_changed: np.ndarray | None
    # Profiling
    timings: GameTimings | None
    _perf: object | None
    _t_game: float | None


# ---------------------------------------------------------------------------
# Init — setup + seed PA
# ---------------------------------------------------------------------------


def init_batch_context(
    game_input: GameInput,
    baserunning_table: dict,
    outcome_labels: list[str],
    batch_prob_source,
    live_prob_source,
    rng: np.random.Generator,
    n_sims: int = 1000,
    max_pa: int = 200,
    manfred_runner: bool = True,
    profile: bool = False,
    blowout_thresholds=None,
    pitcher_exit_model=None,
    compiled_baserunning=None,
    we_array: np.ndarray | None = None,
) -> BatchContext:
    """Set up batch state and run the seed PA. Returns a ready BatchContext."""
    _perf = time.perf_counter if profile else None
    _timings = GameTimings() if profile else None
    _t_game = _perf() if _perf else None

    # Compile baserunning table if not pre-compiled
    if compiled_baserunning is None:
        from sim.engine.lookups.compiled_baserunning import compile_baserunning_table
        compiled_baserunning = compile_baserunning_table(baserunning_table, outcome_labels)

    # If the seed state has outs >= 3 (e.g. live in-play adjustment detected
    # a terminal out before MLB flipped the half), flip the inning first.
    # Direct replace() instead of apply_pa to avoid phantom BF increment.
    if game_input.game_state.outs >= 3:
        gs = game_input.game_state
        if gs.half == 0:
            game_input.game_state = replace(gs, half=1, outs=0, bases=0)
        else:
            game_input.game_state = replace(gs, half=0, inning=gs.inning + 1, outs=0, bases=0)

    bs = batch_from_scalar(game_input.game_state, n_sims)
    outings = _make_batch_outings(n_sims)

    we_snapshot = np.full(n_sims, -1.0, dtype=np.float64) if we_array is not None else None
    horizon_data = HorizonData(n_sims)

    # Outcome classification masks
    walk_mask = np.array([ol in _WALK_SET for ol in outcome_labels], dtype=bool)
    hit_mask = np.array([ol in _HIT_SET for ol in outcome_labels], dtype=bool)
    k_mask = np.array([ol in _K_SET for ol in outcome_labels], dtype=bool)
    field_out_idx = outcome_labels.index("field_out")

    # Pre-extract player ID arrays
    away_lineup_ids = np.array([b["player_id"] for b in game_input.away_lineup], dtype=np.int64)
    home_lineup_ids = np.array([b["player_id"] for b in game_input.home_lineup], dtype=np.int64)
    home_starter_id = np.int64(game_input.home_pitcher["player_id"])
    away_starter_id = np.int64(game_input.away_pitcher["player_id"])
    home_bp_ids = np.array([p["player_id"] for p in game_input.home_bullpen], dtype=np.int64) if game_input.home_bullpen else np.array([], dtype=np.int64)
    away_bp_ids = np.array([p["player_id"] for p in game_input.away_bullpen], dtype=np.int64) if game_input.away_bullpen else np.array([], dtype=np.int64)

    # Player lookup
    player_lookup: dict[int, dict] = {}
    for b in game_input.away_lineup + game_input.home_lineup:
        player_lookup[b["player_id"]] = b
    player_lookup[game_input.home_pitcher["player_id"]] = game_input.home_pitcher
    player_lookup[game_input.away_pitcher["player_id"]] = game_input.away_pitcher
    for p in game_input.home_bullpen + game_input.away_bullpen:
        player_lookup[p["player_id"]] = p

    # Pitcher profiles
    home_pitcher_profiles = _build_pitcher_profile_lookup(home_starter_id, home_bp_ids, player_lookup)
    away_pitcher_profiles = _build_pitcher_profile_lookup(away_starter_id, away_bp_ids, player_lookup)

    # Per-sim tracking
    home_pitcher_idx = np.full(n_sims, -1, dtype=np.int16)
    away_pitcher_idx = np.full(n_sims, -1, dtype=np.int16)
    home_bullpen_pools = [list(range(len(game_input.home_bullpen))) for _ in range(n_sims)]
    away_bullpen_pools = [list(range(len(game_input.away_bullpen))) for _ in range(n_sims)]
    home_changes = np.zeros(n_sims, dtype=np.int16)
    away_changes = np.zeros(n_sims, dtype=np.int16)

    n_classes = len(outcome_labels)
    batter_pa_counts: dict[int, np.ndarray] = {}
    matchup_counts: dict[tuple[int, int], np.ndarray] = {}
    sim_outcome_counts = np.zeros((n_sims, n_classes), dtype=np.int32)
    sim_pa_counts = np.zeros(n_sims, dtype=np.int16)

    prev_inning = bs.inning.copy()
    prev_half = bs.half.copy()
    base_re = float(RE_TABLE[0, 0])
    home_prev_re = np.full(n_sims, base_re, dtype=np.float64)
    away_prev_re = np.full(n_sims, base_re, dtype=np.float64)

    # --- Seed PA (live model with real count data) ---
    seed_state = game_input.game_state
    side = 0 if seed_state.half == 0 else 1
    lineup = game_input.away_lineup if side == 0 else game_input.home_lineup
    batter_idx = seed_state.away_batter_idx if side == 0 else seed_state.home_batter_idx
    batter = lineup[batter_idx]
    pitcher = game_input.home_pitcher if side == 0 else game_input.away_pitcher

    batter_id = batter["player_id"]
    pitcher_id = pitcher["player_id"]

    probs_list = live_prob_source(batter, pitcher, seed_state, game_input.seed_context)
    seed_probs = np.broadcast_to(
        np.array(probs_list, dtype=np.float32).reshape(1, -1),
        (n_sims, n_classes),
    ).copy()

    all_idx = np.arange(n_sims)
    outcome_indices = _batch_sample_outcomes(seed_probs, rng)

    post_bases, runs_scored, outs_added, valid = _batch_resolve_outcomes(
        outcome_indices, bs.bases.copy(), bs.outs.copy(),
        compiled_baserunning, rng,
    )

    # Reroll invalid combos
    for _attempt in range(9):
        if valid.all():
            break
        invalid = ~valid
        new_oi = _batch_sample_outcomes(seed_probs[invalid], rng)
        outcome_indices[invalid] = new_oi
        pb, rs, oa, v = _batch_resolve_outcomes(
            new_oi, bs.bases[np.where(invalid)[0]].copy(),
            bs.outs[np.where(invalid)[0]].copy(),
            compiled_baserunning, rng,
        )
        inv_idx = np.where(invalid)[0]
        post_bases[inv_idx[v]] = pb[v]
        runs_scored[inv_idx[v]] = rs[v]
        outs_added[inv_idx[v]] = oa[v]
        valid[inv_idx[v]] = True

    still_invalid = ~valid
    if still_invalid.any():
        outcome_indices[still_invalid] = field_out_idx
        pb, rs, oa, v = _batch_resolve_outcomes(
            outcome_indices[still_invalid],
            bs.bases[np.where(still_invalid)[0]].copy(),
            bs.outs[np.where(still_invalid)[0]].copy(),
            compiled_baserunning, rng,
        )
        inv_idx = np.where(still_invalid)[0]
        post_bases[inv_idx] = pb
        runs_scored[inv_idx] = rs
        outs_added[inv_idx] = oa

    np.add.at(sim_outcome_counts, (all_idx, outcome_indices), 1)
    sim_pa_counts += 1

    _batch_update_outings(
        outings, all_idx, outcome_indices, runs_scored,
        bs.half.copy(), walk_mask, hit_mask, k_mask,
    )

    if batter_id not in batter_pa_counts:
        batter_pa_counts[batter_id] = np.zeros(n_sims, dtype=np.int16)
    batter_pa_counts[batter_id] += 1
    mk = (batter_id, pitcher_id)
    if mk not in matchup_counts:
        matchup_counts[mk] = np.zeros(n_sims, dtype=np.int16)
    matchup_counts[mk] += 1

    _batch_apply_pa(bs, all_idx, post_bases, runs_scored, outs_added)
    _batch_advance_lineup(bs, all_idx)
    bs.active &= ~batch_game_over(bs)

    return BatchContext(
        bs=bs, outings=outings, game_input=game_input,
        outcome_labels=outcome_labels, rng=rng, n_sims=n_sims,
        max_pa=max_pa, manfred_runner=manfred_runner,
        compiled_baserunning=compiled_baserunning,
        batch_prob_source=batch_prob_source,
        pitcher_exit_model=pitcher_exit_model,
        blowout_thresholds=blowout_thresholds,
        we_array=we_array, we_snapshot=we_snapshot,
        horizon_data=horizon_data,
        walk_mask=walk_mask, hit_mask=hit_mask, k_mask=k_mask,
        field_out_idx=field_out_idx,
        away_lineup_ids=away_lineup_ids, home_lineup_ids=home_lineup_ids,
        home_starter_id=home_starter_id, away_starter_id=away_starter_id,
        home_bp_ids=home_bp_ids, away_bp_ids=away_bp_ids,
        player_lookup=player_lookup,
        home_pitcher_profiles=home_pitcher_profiles,
        away_pitcher_profiles=away_pitcher_profiles,
        home_pitcher_idx=home_pitcher_idx, away_pitcher_idx=away_pitcher_idx,
        home_bullpen_pools=home_bullpen_pools, away_bullpen_pools=away_bullpen_pools,
        home_changes=home_changes, away_changes=away_changes,
        batter_pa_counts=batter_pa_counts, matchup_counts=matchup_counts,
        sim_outcome_counts=sim_outcome_counts, sim_pa_counts=sim_pa_counts,
        prev_inning=prev_inning, prev_half=prev_half,
        home_prev_re=home_prev_re, away_prev_re=away_prev_re,
        base_re=base_re, pa_step=1, last_half_changed=None,
        timings=_timings, _perf=_perf, _t_game=_t_game,
    )


# ---------------------------------------------------------------------------
# Step — one PA for all active sims
# ---------------------------------------------------------------------------


def step_pa(ctx: BatchContext) -> bool:
    """Advance all active sims by one plate appearance. Returns True if any still active.

    Sets ctx.last_half_changed to a bool mask (indexed into active_idx) indicating
    which sims crossed a half-inning boundary on this step.
    """
    bs = ctx.bs
    if not bs.active.any():
        ctx.last_half_changed = None
        return False

    active_idx = np.where(bs.active)[0]
    n_active = len(active_idx)
    _perf = ctx._perf
    _timings = ctx.timings

    # --- Manfred runner ---
    if ctx.manfred_runner:
        changed = ((bs.inning[active_idx] != ctx.prev_inning[active_idx]) |
                    (bs.half[active_idx] != ctx.prev_half[active_idx]))
        extras = bs.inning[active_idx] >= 10
        fresh = bs.outs[active_idx] == 0
        empty = bs.bases[active_idx] == 0
        apply_mr = changed & extras & fresh & empty
        if apply_mr.any():
            bs.bases[active_idx[apply_mr]] |= 0b010

    ctx.prev_inning[active_idx] = bs.inning[active_idx]
    ctx.prev_half[active_idx] = bs.half[active_idx]

    # --- Identify batters and pitchers ---
    if _perf:
        _t0 = _perf()
    halves = bs.half[active_idx]

    batter_ids = np.where(
        halves == 0,
        ctx.away_lineup_ids[bs.away_batter_idx[active_idx]],
        ctx.home_lineup_ids[bs.home_batter_idx[active_idx]],
    )

    is_top = halves == 0
    pitcher_ids_arr = np.empty(n_active, dtype=np.int64)

    if is_top.any():
        home_pi = ctx.home_pitcher_idx[active_idx[is_top]]
        home_is_starter = home_pi == -1
        pitcher_ids_arr[is_top] = np.where(
            home_is_starter,
            ctx.home_starter_id,
            np.where(
                (home_pi >= 0) & (home_pi < len(ctx.home_bp_ids)),
                ctx.home_bp_ids[np.clip(home_pi, 0, max(len(ctx.home_bp_ids) - 1, 0))],
                -1,
            ),
        )
    if (~is_top).any():
        away_pi = ctx.away_pitcher_idx[active_idx[~is_top]]
        away_is_starter = away_pi == -1
        pitcher_ids_arr[~is_top] = np.where(
            away_is_starter,
            ctx.away_starter_id,
            np.where(
                (away_pi >= 0) & (away_pi < len(ctx.away_bp_ids)),
                ctx.away_bp_ids[np.clip(away_pi, 0, max(len(ctx.away_bp_ids) - 1, 0))],
                -1,
            ),
        )

    if _perf:
        _timings.id_resolve += _perf() - _t0

    # --- Build shared defensive arrays ---
    if _perf:
        _t0 = _perf()

    shared_arrays = _build_defensive_arrays(
        bs, ctx.outings, active_idx, halves, ctx.walk_mask, ctx.hit_mask,
    )

    if _perf:
        _timings.context_build += _perf() - _t0

    # --- Pitcher exit check ---
    if _perf:
        _t0 = _perf()

    if ctx.pitcher_exit_model is not None:
        features_T = _build_exit_features_transposed(
            ctx.pitcher_exit_model, bs, shared_arrays, halves,
            ctx.home_prev_re, ctx.away_prev_re,
            ctx.home_pitcher_idx, ctx.away_pitcher_idx,
            ctx.home_pitcher_profiles, ctx.away_pitcher_profiles,
        )
        all_exit_probs = ctx.pitcher_exit_model.predict_transposed(features_T)

    any_pulled = False
    for defending_home in (True, False):
        half_val = 0 if defending_home else 1
        hmask = halves == half_val
        if not hmask.any():
            continue

        check_active = active_idx[hmask]

        if defending_home:
            sp = bs.home_starter_pulled
            bf_a = bs.home_pitcher_bf
            pidx_arr = ctx.home_pitcher_idx
            bp_pools = ctx.home_bullpen_pools
            ch_arr = ctx.home_changes
        else:
            sp = bs.away_starter_pulled
            bf_a = bs.away_pitcher_bf
            pidx_arr = ctx.away_pitcher_idx
            bp_pools = ctx.away_bullpen_pools
            ch_arr = ctx.away_changes

        if ctx.pitcher_exit_model is not None:
            exit_probs = all_exit_probs[hmask]
        else:
            exit_probs = _vectorized_placeholder_exit(bf_a, sp, check_active)

        rolls = ctx.rng.random(len(check_active))
        pulled = rolls < exit_probs
        if not pulled.any():
            continue

        any_pulled = True
        pull_idx = check_active[pulled]
        _reset_pitcher_outing(bs, ctx.outings, pull_idx, defending_home)

        for idx in pull_idx:
            si = int(idx)
            pool = bp_pools[si]
            if pool:
                choice = int(ctx.rng.integers(0, len(pool)))
                pidx_arr[si] = pool.pop(choice)
            else:
                pidx_arr[si] = -2
            ch_arr[si] += 1

        pull_set = set(pull_idx.tolist())
        for j in range(n_active):
            si = int(active_idx[j])
            if si in pull_set:
                h = int(halves[j])
                if defending_home and h == 0:
                    pi = int(ctx.home_pitcher_idx[si])
                    pitcher_ids_arr[j] = ctx.home_bp_ids[pi] if 0 <= pi < len(ctx.home_bp_ids) else -1
                elif not defending_home and h == 1:
                    pi = int(ctx.away_pitcher_idx[si])
                    pitcher_ids_arr[j] = ctx.away_bp_ids[pi] if 0 <= pi < len(ctx.away_bp_ids) else -1

    if any_pulled:
        shared_arrays = _build_defensive_arrays(
            bs, ctx.outings, active_idx, halves, ctx.walk_mask, ctx.hit_mask,
        )

    if _perf:
        _timings.pitcher_exit += _perf() - _t0

    # --- Build dynamic context arrays ---
    if _perf:
        _t0 = _perf()

    matchup_pairs = np.column_stack((batter_ids, pitcher_ids_arr))
    unique_pairs, group_idx = np.unique(matchup_pairs, axis=0, return_inverse=True)
    n_unique = len(unique_pairs)

    bpa = np.zeros(n_active, dtype=np.float32)
    mpc = np.zeros(n_active, dtype=np.float32)
    for g in range(n_unique):
        gmask = group_idx == g
        bid = int(unique_pairs[g, 0])
        pid = int(unique_pairs[g, 1])
        g_sim_idx = active_idx[gmask]
        if bid in ctx.batter_pa_counts:
            bpa[gmask] = ctx.batter_pa_counts[bid][g_sim_idx]
        if (bid, pid) in ctx.matchup_counts:
            mpc[gmask] = ctx.matchup_counts[(bid, pid)][g_sim_idx]

    def_bf = shared_arrays["def_bf"]
    def_tto = shared_arrays["def_tto"]
    def_walks = shared_arrays["def_walks"]
    def_hits = shared_arrays["def_hits"]
    def_k = shared_arrays["def_k"]
    def_runs = shared_arrays["def_runs"]
    def_whip = shared_arrays["def_whip"]
    def_recent_whip = shared_arrays["def_recent_whip"]
    run_diff = shared_arrays["run_diff"]

    dynamic_arrays = {
        "inning": bs.inning[active_idx].astype(np.float32),
        "is_bottom": halves.astype(np.float32),
        "outs": bs.outs[active_idx].astype(np.float32),
        "runner_1b": ((bs.bases[active_idx] >> 0) & 1).astype(np.float32),
        "runner_2b": ((bs.bases[active_idx] >> 1) & 1).astype(np.float32),
        "runner_3b": ((bs.bases[active_idx] >> 2) & 1).astype(np.float32),
        "run_diff": run_diff,
        "is_home": halves.astype(np.float32),
        "times_through_order": def_tto.astype(np.float32),
        "batter_prior_pa": bpa,
        "pitcher_bf_game": def_bf.astype(np.float32),
        "batter_ab_vs_pitcher": mpc,
        "pitcher_outing_walks": def_walks.astype(np.float32),
        "pitcher_outing_hits": def_hits.astype(np.float32),
        "pitcher_outing_k": def_k.astype(np.float32),
        "pitcher_outing_runs": def_runs.astype(np.float32),
        "pitcher_outing_whip": def_whip,
        "pitcher_recent_whip": def_recent_whip,
    }

    if _perf:
        _timings.context_build += _perf() - _t0

    # --- Batched ONNX inference ---
    if _perf:
        _t0 = _perf()

    unique_batter_dicts = []
    unique_pitcher_dicts = []
    for g in range(n_unique):
        bid = int(unique_pairs[g, 0])
        pid = int(unique_pairs[g, 1])
        unique_batter_dicts.append(ctx.player_lookup.get(bid, {"player_id": bid, "hand": "R"}))
        unique_pitcher_dicts.append(ctx.player_lookup.get(pid, {"player_id": pid, "hand": "R"}))

    probs = ctx.batch_prob_source(
        unique_batter_dicts, unique_pitcher_dicts, group_idx,
        dynamic_arrays, n_active,
    )

    if _perf:
        _timings.prob_source += _perf() - _t0

    # --- Outcome sampling ---
    if _perf:
        _t0 = _perf()

    outcome_indices = _batch_sample_outcomes(probs, ctx.rng)

    cur_bases = bs.bases[active_idx].copy()
    cur_outs = bs.outs[active_idx].copy()

    post_bases, runs_scored, outs_added, valid = _batch_resolve_outcomes(
        outcome_indices, cur_bases, cur_outs,
        ctx.compiled_baserunning, ctx.rng,
    )

    # Reroll invalid combos
    for _attempt in range(9):
        if valid.all():
            break
        invalid = ~valid
        new_oi = _batch_sample_outcomes(probs[invalid], ctx.rng)
        outcome_indices[invalid] = new_oi
        pb, rs, oa, v = _batch_resolve_outcomes(
            new_oi, cur_bases[invalid], cur_outs[invalid],
            ctx.compiled_baserunning, ctx.rng,
        )
        inv_idx = np.where(invalid)[0]
        post_bases[inv_idx[v]] = pb[v]
        runs_scored[inv_idx[v]] = rs[v]
        outs_added[inv_idx[v]] = oa[v]
        valid[inv_idx[v]] = True

    still_invalid = ~valid
    if still_invalid.any():
        outcome_indices[still_invalid] = ctx.field_out_idx
        pb, rs, oa, v = _batch_resolve_outcomes(
            outcome_indices[still_invalid],
            cur_bases[still_invalid],
            cur_outs[still_invalid],
            ctx.compiled_baserunning, ctx.rng,
        )
        inv_idx = np.where(still_invalid)[0]
        post_bases[inv_idx] = pb
        runs_scored[inv_idx] = rs
        outs_added[inv_idx] = oa

    np.add.at(ctx.sim_outcome_counts, (active_idx, outcome_indices), 1)

    if _perf:
        _timings.outcome_sample += _perf() - _t0

    # --- State transition ---
    if _perf:
        _t0 = _perf()

    _batch_update_outings(
        ctx.outings, active_idx, outcome_indices, runs_scored,
        halves, ctx.walk_mask, ctx.hit_mask, ctx.k_mask,
    )

    for g in range(n_unique):
        gmask = group_idx == g
        bid = int(unique_pairs[g, 0])
        pid = int(unique_pairs[g, 1])
        g_sim_idx = active_idx[gmask]
        if bid not in ctx.batter_pa_counts:
            ctx.batter_pa_counts[bid] = np.zeros(ctx.n_sims, dtype=np.int16)
        ctx.batter_pa_counts[bid][g_sim_idx] += 1
        mk = (bid, pid)
        if mk not in ctx.matchup_counts:
            ctx.matchup_counts[mk] = np.zeros(ctx.n_sims, dtype=np.int16)
        ctx.matchup_counts[mk][g_sim_idx] += 1

    # Capture current RE before state change
    _pre_bases = bs.bases[active_idx]
    _pre_rb = (_pre_bases & 1) + ((_pre_bases >> 1) & 1) * 2 + ((_pre_bases >> 2) & 1) * 4
    _pre_outs = np.minimum(bs.outs[active_idx], 2)
    _pre_re = RE_TABLE[_pre_rb.astype(int), _pre_outs.astype(int)]

    _batch_apply_pa(bs, active_idx, post_bases, runs_scored, outs_added)
    _batch_advance_lineup(bs, active_idx)

    # Detect half-inning boundary crossings
    half_changed = ((bs.inning[active_idx] != ctx.prev_inning[active_idx]) |
                    (bs.half[active_idx] != ctx.prev_half[active_idx]))
    ctx.last_half_changed = half_changed

    # Update prev_re
    if half_changed.any():
        changed_idx = active_idx[half_changed]
        ctx.home_prev_re[changed_idx] = ctx.base_re
        ctx.away_prev_re[changed_idx] = ctx.base_re
    stayed = ~half_changed
    if stayed.any():
        stayed_idx = active_idx[stayed]
        stayed_halves = halves[stayed]
        top_stayed = stayed_halves == 0
        bot_stayed = ~top_stayed
        if top_stayed.any():
            ctx.home_prev_re[stayed_idx[top_stayed]] = _pre_re[stayed][top_stayed]
        if bot_stayed.any():
            ctx.away_prev_re[stayed_idx[bot_stayed]] = _pre_re[stayed][bot_stayed]

    if _perf:
        _timings.state_transition += _perf() - _t0

    # --- Per-sim PA count ---
    ctx.sim_pa_counts[active_idx] += 1

    # --- Horizon snapshots ---
    if ctx.pa_step == 1:
        ctx.horizon_data.pa1_inning[:] = bs.inning
        ctx.horizon_data.pa1_half[:] = bs.half
        ctx.horizon_data.pa1_outs[:] = bs.outs
        ctx.horizon_data.pa1_bases[:] = bs.bases
        ctx.horizon_data.pa1_home_score[:] = bs.home_score
        ctx.horizon_data.pa1_away_score[:] = bs.away_score
        ctx.horizon_data.pa1_active[:] = bs.active
    if half_changed.any():
        ci = active_idx[half_changed]
        hc = ctx.horizon_data.hi_count[ci]
        safe = hc < ctx.horizon_data.hi_home_score.shape[1]
        if safe.any():
            ci_s = ci[safe]
            hc_s = hc[safe]
            ctx.horizon_data.hi_inning[ci_s, hc_s] = bs.inning[ci_s]
            ctx.horizon_data.hi_half[ci_s, hc_s] = bs.half[ci_s]
            ctx.horizon_data.hi_outs[ci_s, hc_s] = bs.outs[ci_s]
            ctx.horizon_data.hi_bases[ci_s, hc_s] = bs.bases[ci_s]
            ctx.horizon_data.hi_home_score[ci_s, hc_s] = bs.home_score[ci_s]
            ctx.horizon_data.hi_away_score[ci_s, hc_s] = bs.away_score[ci_s]
            if ctx.we_array is not None:
                inn_c = np.minimum(bs.inning[ci_s], 9)
                rd_c = np.clip(
                    (bs.home_score[ci_s] - bs.away_score[ci_s]).astype(np.int16),
                    -15, 15,
                )
                ctx.horizon_data.hi_we[ci_s, hc_s] = ctx.we_array[
                    inn_c, bs.half[ci_s], 0, 0, rd_c + 15
                ]
            ctx.horizon_data.hi_count[ci_s] += 1

    # --- WE snapshot (rolling — last written value is the one we keep) ---
    if ctx.we_snapshot is not None:
        _inn = np.minimum(bs.inning[active_idx], 9)
        _hlf = bs.half[active_idx]
        _rd = np.clip(
            bs.home_score[active_idx].astype(np.int16)
            - bs.away_score[active_idx].astype(np.int16),
            -15, 15,
        )
        ctx.we_snapshot[active_idx] = ctx.we_array[
            _inn, _hlf, bs.outs[active_idx], bs.bases[active_idx], _rd + 15,
        ]

    # --- Blowout pruning ---
    if ctx.blowout_thresholds is not None:
        win_thresh, loss_thresh = ctx.blowout_thresholds
        inn = bs.inning[active_idx]
        hlf = bs.half[active_idx]
        inn_clamped = np.minimum(inn, win_thresh.shape[0] - 1)
        wt = win_thresh[inn_clamped, hlf]
        lt = loss_thresh[inn_clamped, hlf]
        rd = (bs.home_score[active_idx] - bs.away_score[active_idx]).astype(np.int16)
        is_blowout = (rd >= wt) | (rd <= lt)
        if is_blowout.any():
            prune_idx = active_idx[is_blowout]
            bs.pruned[prune_idx] = True
            bs.pruned_at_pa[prune_idx] = ctx.pa_step
            bs.active[prune_idx] = False

    # --- Update active mask ---
    bs.active &= ~batch_game_over(bs)
    ctx.pa_step += 1

    return bs.active.any()


# ---------------------------------------------------------------------------
# Finalize — build results from completed batch
# ---------------------------------------------------------------------------


def finalize_batch(
    ctx: BatchContext,
) -> tuple[list[GameResult], np.ndarray | None, HorizonData]:
    """Build GameResult list from completed batch context."""
    bs = ctx.bs

    # Log if any sims hit the safety valve
    if ctx.pa_step >= ctx.max_pa and bs.active.any():
        n_capped = int(bs.active.sum())
        logger.warning(
            "Batch hit max_pa safety valve (%d PAs) — %d/%d sims still active",
            ctx.max_pa, n_capped, ctx.n_sims,
        )

    if ctx._perf and ctx._t_game is not None:
        ctx.timings.total = ctx._perf() - ctx._t_game
        ctx.timings.n_pas = ctx.pa_step

    # Correct innings
    corrected_innings = bs.inning.copy()
    away_win_reg = (bs.half == 0) & (bs.inning > 9)
    corrected_innings[away_win_reg] -= 1

    results = []
    for i in range(ctx.n_sims):
        nz_cols = np.where(ctx.sim_outcome_counts[i] > 0)[0]
        oc = {ctx.outcome_labels[k]: int(ctx.sim_outcome_counts[i, k]) for k in nz_cols}
        results.append(GameResult(
            home_score=int(bs.home_score[i]),
            away_score=int(bs.away_score[i]),
            innings=int(corrected_innings[i]),
            total_pas=int(ctx.sim_pa_counts[i]),
            home_pitcher_changes=int(ctx.home_changes[i]),
            away_pitcher_changes=int(ctx.away_changes[i]),
            timings=ctx.timings,
            outcome_counts=oc,
            pruned=bool(bs.pruned[i]),
            pruned_at_pa=int(bs.pruned_at_pa[i]),
            we_at_end=float(ctx.we_snapshot[i]) if ctx.we_snapshot is not None else -1.0,
        ))

    return results, ctx.we_snapshot, ctx.horizon_data


# ---------------------------------------------------------------------------
# Main batch simulation — thin wrapper over init/step/finalize
# ---------------------------------------------------------------------------


def simulate_game_batch(
    game_input: GameInput,
    baserunning_table: dict,
    outcome_labels: list[str],
    batch_prob_source,
    live_prob_source,
    rng: np.random.Generator,
    n_sims: int = 1000,
    max_pa: int = 200,
    manfred_runner: bool = True,
    profile: bool = False,
    blowout_thresholds=None,
    pitcher_exit_model=None,
    compiled_baserunning=None,
    we_array: np.ndarray | None = None,
) -> tuple[list[GameResult], np.ndarray | None, HorizonData]:
    """Simulate N games in parallel from the same starting state.

    Returns (results, we_snapshot, horizon_data):
      - we_snapshot: (n_sims,) WE at each sim's last PA, None if we_array not provided
      - horizon_data: state snapshots at +1PA and each half-inning boundary
    """
    ctx = init_batch_context(
        game_input, baserunning_table, outcome_labels,
        batch_prob_source, live_prob_source, rng,
        n_sims, max_pa, manfred_runner, profile,
        blowout_thresholds, pitcher_exit_model,
        compiled_baserunning, we_array,
    )
    while ctx.bs.active.any() and ctx.pa_step < ctx.max_pa:
        step_pa(ctx)
    return finalize_batch(ctx)
