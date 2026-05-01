"""Truncated Monte Carlo estimator — weighted WE average.

Sims N trajectories forward, recording the WE table value at each
half-inning boundary. The per-sim estimate is a weighted average of
those WE readings:

    P(home) = Σ w_h * WE_h

Weights come from the horizon_weights artifact — minimum-variance
weights derived from inverse error covariance (Ledoit-Wolf shrinkage)
with improvement gating. High where the sim adds value over the WE
table, zero where it doesn't.

When a sim's game ends before all weighted horizons are reached, the
known outcome (1.0 home win, 0.0 away win) substitutes for WE at
remaining horizons — the limiting case of WE when the game is decided.

Requires:
  - we_array: win expectancy lookup array
  - horizon_weights: per-state weight schedule across horizons
  - batch_prob_source: ONNX model for batched inference
Optional:
  - n_lookup: per-state N allocation (adaptive N)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Callable

import numpy as np

from sim.engine.core.engine import GameResult, ProbSource
from sim.engine.estimators.types import SimulationResult
from sim.game_inputs.game import GameInput

logger = logging.getLogger(__name__)

def _aggregate_outcome_counts(results: list[GameResult]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for r in results:
        if r.outcome_counts:
            total.update(r.outcome_counts)
    return dict(total)


def estimate(
    game_input: GameInput,
    baserunning_table: dict,
    outcome_labels: list[str],
    sim_prob_source: ProbSource,
    *,
    seed: int = 42,
    max_pa: int = 200,
    manfred_runner: bool = True,
    profile: bool = False,
    batch_prob_source: Callable | None = None,
    blowout_thresholds=None,
    live_prob_source: ProbSource,
    pitcher_exit_model=None,
    compiled_baserunning=None,
    we_array: np.ndarray | None = None,
    we_table: dict | None = None,
    n_lookup: dict | None = None,
    stopping_thresholds: dict | None = None,
    gamma_schedule: dict | None = None,
    horizon_weights: dict | None = None,
    estimator_config=None,
) -> SimulationResult:
    """Run N sims, return weighted average of WE readings at half-inning boundaries."""
    from sim.engine.core.batch_engine import init_batch_context, step_pa, finalize_batch
    from sim.engine.estimators.buckets import state_key
    from sim.engine.estimators.config import TruncatedMcConfig

    cfg = estimator_config if isinstance(estimator_config, TruncatedMcConfig) else TruncatedMcConfig()

    if we_array is None:
        raise RuntimeError(
            "Truncated MC requires a win_expectancy artifact for WE lookups."
        )
    if batch_prob_source is None:
        raise RuntimeError(
            "Truncated MC requires a batch_prob_source (ONNX model)."
        )

    # --- Resolve state and WE entry ---
    gs = game_input.game_state
    sk = state_key(gs.inning, gs.home_score - gs.away_score)
    rd = max(-15, min(15, gs.home_score - gs.away_score))
    inn = min(gs.inning, 9)
    we_entry = float(we_array[inn, gs.half, min(gs.outs, 2), gs.bases, rd + 15])

    # --- Resolve weights — early return if uncalibrated ---
    weights = None
    if horizon_weights:
        val = horizon_weights.get("lookup", {}).get(sk)
        if isinstance(val, dict):
            weights = val

    if not weights:
        logger.info(
            "No horizon_weights for state=%s, returning WE_entry=%.4f",
            sk, we_entry,
        )
        return SimulationResult(
            n_sims=0, p_home_win=we_entry, p_home_win_se=0.0,
            mean_home_score=0.0, mean_away_score=0.0, mean_total_runs=0.0,
            mean_innings=0.0, runs_per_inning=0.0, results=[],
        )

    # --- Resolve adaptive N ---
    n_sims = cfg.n_sims
    if cfg.adaptive_n and n_lookup:
        n_sims = n_lookup.get("lookup", {}).get(sk, cfg.n_sims)

    # Apply max_horizon cap — drop horizons beyond the cap, renormalize
    if cfg.max_horizon is not None:
        weights = {k: v for k, v in weights.items() if int(k[1:-2]) <= cfg.max_horizon}
        if not weights:
            logger.info(
                "All horizons beyond max_horizon=%d for state=%s, returning WE_entry=%.4f",
                cfg.max_horizon, sk, we_entry,
            )
            return SimulationResult(
                n_sims=0, p_home_win=we_entry, p_home_win_se=0.0,
                mean_home_score=0.0, mean_away_score=0.0, mean_total_runs=0.0,
                mean_innings=0.0, runs_per_inning=0.0, results=[],
            )
        w_sum = sum(weights.values())
        if w_sum > 0:
            weights = {k: v / w_sum for k, v in weights.items()}

    # Max horizon = last key with weight > 0
    max_horizon = max(
        (int(k[1:-2]) for k, w in weights.items() if w > 0),
        default=5,
    )

    logger.info(
        "Truncated MC: state=%s n_sims=%d max_horizon=%d n_weighted=%d we_entry=%.4f",
        sk, n_sims, max_horizon, len(weights), we_entry,
    )

    # --- Run sims ---
    rng = np.random.default_rng(seed)

    ctx = init_batch_context(
        game_input, baserunning_table, outcome_labels,
        batch_prob_source, live_prob_source, rng,
        n_sims=n_sims, max_pa=max_pa, manfred_runner=manfred_runner,
        profile=profile,
        blowout_thresholds=blowout_thresholds if cfg.enable_pruning else None,
        pitcher_exit_model=pitcher_exit_model,
        compiled_baserunning=compiled_baserunning, we_array=we_array,
    )

    # Track per-sim half-inning boundary count
    boundary_counts = np.zeros(n_sims, dtype=np.int16)

    while ctx.bs.active.any() and ctx.pa_step < ctx.max_pa:
        step_pa(ctx)

        if ctx.last_half_changed is not None and ctx.last_half_changed.any():
            active_idx = np.where(ctx.bs.active)[0]
            if len(active_idx) > 0 and len(ctx.last_half_changed) == len(active_idx):
                crossed = active_idx[ctx.last_half_changed]
                boundary_counts[crossed] += 1

                # Deactivate sims that passed the max weighted horizon
                reached = crossed[boundary_counts[crossed] >= max_horizon]
                if len(reached) > 0:
                    ctx.bs.active[reached] = False

        if not ctx.bs.active.any():
            break

    # --- Weighted WE average ---
    hd = ctx.horizon_data
    bs = ctx.bs

    # Build weight vector and horizon indices from artifact
    sorted_horizons = sorted(weights.items(), key=lambda x: int(x[0][1:-2]))
    h_indices = np.array([int(k[1:-2]) - 1 for k, _ in sorted_horizons], dtype=np.int32)
    w_vec = np.array([w for _, w in sorted_horizons], dtype=np.float64)

    # (n_sims, n_weighted_horizons) — WE at each weighted checkpoint
    we_at_h = hd.hi_we[:n_sims, h_indices]

    # Which horizons were reached? (hi_we is -1.0 for unreached)
    reached_mask = we_at_h >= 0.0

    # For unreached horizons: use known outcome (game ended)
    home_won = (bs.home_score[:n_sims] > bs.away_score[:n_sims]).astype(np.float64)

    # Safety-valve sims (max_pa hit, game still going) → use last recorded WE
    still_active = bs.active[:n_sims]
    if still_active.any():
        last_idx = np.clip(hd.hi_count[:n_sims].astype(np.intp) - 1, 0, hd.hi_we.shape[1] - 1)
        last_we = hd.hi_we[np.arange(n_sims), last_idx]
        has_data = hd.hi_count[:n_sims] > 0
        home_won = np.where(still_active & has_data, last_we, home_won)

    # Assemble: WE where reached, outcome where not
    values = np.where(reached_mask, we_at_h, home_won[:, np.newaxis])

    # Weighted average per sim, then aggregate
    estimate_per_sim = np.clip(values @ w_vec, 0.0, 1.0)
    p_home_win = float(np.mean(estimate_per_sim))
    p_home_win_se = float(np.std(estimate_per_sim, ddof=1) / math.sqrt(n_sims))

    # --- Build results for downstream compatibility ---
    results, we_snapshot, horizon_data = finalize_batch(ctx)

    total_home = sum(r.home_score for r in results)
    total_away = sum(r.away_score for r in results)
    total_runs = total_home + total_away
    total_innings = sum(r.innings for r in results)
    total_half_innings = sum(
        r.innings * 2 - (1 if r.home_score > r.away_score and r.innings >= 9 else 0)
        for r in results
    )
    rpi = total_runs / total_half_innings if total_half_innings > 0 else 0.0

    mean_timings = results[0].timings if results and results[0].timings else None

    sim_result = SimulationResult(
        n_sims=n_sims,
        p_home_win=p_home_win,
        p_home_win_se=p_home_win_se,
        mean_home_score=total_home / n_sims,
        mean_away_score=total_away / n_sims,
        mean_total_runs=total_runs / n_sims,
        mean_innings=total_innings / n_sims,
        runs_per_inning=rpi,
        results=results,
        mean_timings=mean_timings,
        outcome_counts=_aggregate_outcome_counts(results),
        horizon_data=horizon_data,
    )

    logger.info(
        "Weighted WE estimate: state=%s n=%d P(home)=%.3f ± %.3f "
        "max_horizon=%d mean_score=%.1f-%.1f",
        sk, n_sims, p_home_win, p_home_win_se * 1.96,
        max_horizon,
        sim_result.mean_home_score, sim_result.mean_away_score,
    )

    return sim_result
