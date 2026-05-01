"""Sequential Monte Carlo estimator.

Particle-based SMC for variance reduction over naive MC. The core loop:
  1. Resolve N, stopping threshold, γ from calibration artifacts for this game state
  2. Spawn N particles via init_batch_context (same seed PA as naive MC)
  3. Step PAs via step_pa until half-inning boundary
  4. At boundary: WE lookup → tempered reweighting → check ESS → resample if needed
  5. Repeat until ESS plateaus, spread converges, or game ends

Tempering: π_t(particle) ∝ WE(particle)^γ * sim_prior^(1-γ)
  γ near 0 → flat weights, trust sim diversity
  γ near 1 → WE-proportional, trust WE table

Importance weighting: per-particle cumulative weights track the bias
introduced by resampling. All estimates use the weighted mean, not the
unweighted mean of the (biased) resampled population. This is the
marginal likelihood correction from Del Moral (2006).

Uses the same batch engine primitives as naive MC — different control loop on top.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import Counter
from dataclasses import fields
from typing import Callable

import numpy as np

from sim.engine.core.engine import ProbSource
from sim.engine.estimators.buckets import state_key as _state_key
from sim.engine.estimators.types import SimulationResult
from sim.game_inputs.game import GameInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WE lookup for active particles
# ---------------------------------------------------------------------------

def _we_lookup_batch(
    bs, active_idx: np.ndarray, we_array: np.ndarray,
) -> np.ndarray:
    """Vectorized WE lookup at current state for active particles."""
    inn = np.minimum(bs.inning[active_idx], 9)
    hlf = bs.half[active_idx]
    outs = bs.outs[active_idx]
    bases = bs.bases[active_idx]
    rd = np.clip(
        bs.home_score[active_idx].astype(np.int16)
        - bs.away_score[active_idx].astype(np.int16),
        -15, 15,
    )
    return we_array[inn, hlf, outs, bases, rd + 15].astype(np.float64)


# ---------------------------------------------------------------------------
# Systematic resampling
# ---------------------------------------------------------------------------

def _systematic_resample(
    weights: np.ndarray, n: int, rng: np.random.Generator,
) -> np.ndarray:
    """Systematic resampling — returns index array of length n."""
    cumsum = np.cumsum(weights)
    u = (rng.random() + np.arange(n)) / n
    return np.searchsorted(cumsum, u).astype(np.intp)


def _compute_ess(weights: np.ndarray) -> float:
    """Effective Sample Size from normalized weights."""
    return 1.0 / float(np.sum(weights ** 2))


def _compute_weights(we_vals: np.ndarray, gamma: float) -> np.ndarray:
    """Compute normalized tempered likelihood weights.

    w_i ∝ WE_i^γ (per spec: π_t ∝ WE^γ * sim_prior^(1-γ))
    γ near 0 → flat weights (trust sim, preserve diversity)
    γ near 1 → concentrate on high-WE particles (trust WE table)
    """
    weights = np.power(np.clip(we_vals, 1e-10, 1.0), gamma)
    weights /= weights.sum()
    return weights


def _weighted_estimate(
    we_vals: np.ndarray, particle_weights: np.ndarray, active_idx: np.ndarray,
) -> tuple[float, float]:
    """Importance-weighted P(home) estimate and SE from particle weights."""
    w = particle_weights[active_idx]
    w_sum = w.sum()
    if w_sum < 1e-15:
        return float(np.mean(we_vals)), 0.0
    w_norm = w / w_sum
    p = float(np.sum(w_norm * we_vals))
    var = float(np.sum(w_norm * (we_vals - p) ** 2))
    ess = _compute_ess(w_norm)
    se = math.sqrt(var / ess) if ess > 1 else math.sqrt(var)
    return p, se


def _resample_particles(ctx, active_idx: np.ndarray, weights: np.ndarray,
                         rng: np.random.Generator) -> None:
    """Resample active particles using pre-computed normalized weights.

    Modifies ctx in place — reindexes all per-sim arrays for active particles.
    """
    n_active = len(active_idx)
    if n_active <= 1:
        return

    indices = _systematic_resample(weights, n_active, rng)
    resampled_full_idx = active_idx[indices]

    # Reindex BatchState arrays
    bs = ctx.bs
    for f in fields(bs):
        if f.name == "n":
            continue
        arr = getattr(bs, f.name)
        if isinstance(arr, np.ndarray) and arr.shape[0] == ctx.n_sims:
            arr[active_idx] = arr[resampled_full_idx]

    # Reindex BatchOutings arrays
    for f in fields(ctx.outings):
        arr = getattr(ctx.outings, f.name)
        if isinstance(arr, np.ndarray):
            if arr.ndim == 1 and arr.shape[0] == ctx.n_sims:
                arr[active_idx] = arr[resampled_full_idx]
            elif arr.ndim == 2 and arr.shape[0] == ctx.n_sims:
                arr[active_idx] = arr[resampled_full_idx]

    # Reindex per-sim tracking arrays on ctx
    for attr in ("home_pitcher_idx", "away_pitcher_idx",
                 "home_changes", "away_changes",
                 "sim_outcome_counts", "sim_pa_counts",
                 "prev_inning", "prev_half",
                 "home_prev_re", "away_prev_re"):
        arr = getattr(ctx, attr)
        if isinstance(arr, np.ndarray):
            if arr.ndim == 1 and arr.shape[0] == ctx.n_sims:
                arr[active_idx] = arr[resampled_full_idx]
            elif arr.ndim == 2 and arr.shape[0] == ctx.n_sims:
                arr[active_idx] = arr[resampled_full_idx]

    # Reindex WE snapshot
    if ctx.we_snapshot is not None:
        ctx.we_snapshot[active_idx] = ctx.we_snapshot[resampled_full_idx]

    # Reindex bullpen pools (list of lists — must deepcopy)
    for pool_attr in ("home_bullpen_pools", "away_bullpen_pools"):
        pools = getattr(ctx, pool_attr)
        new_pools = [copy.deepcopy(pools[int(resampled_full_idx[i])])
                     for i in range(n_active)]
        for i, idx in enumerate(active_idx):
            pools[int(idx)] = new_pools[i]

    # Reindex batter_pa_counts and matchup_counts (dict of arrays)
    for counts_dict in (ctx.batter_pa_counts, ctx.matchup_counts):
        for key, arr in counts_dict.items():
            arr[active_idx] = arr[resampled_full_idx]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_outcome_counts(results) -> dict[str, int]:
    """Sum outcome_counts across all sim results."""
    total: Counter[str] = Counter()
    for r in results:
        if r.outcome_counts:
            total.update(r.outcome_counts)
    return dict(total)


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

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
    """Run SMC particle simulation with importance-weighted estimates."""
    from sim.engine.core.batch_engine import init_batch_context, step_pa, finalize_batch
    from sim.engine.estimators.config import SmcConfig, DEFAULT_N_SIMS, DEFAULT_STOPPING_THRESHOLD

    cfg = estimator_config if isinstance(estimator_config, SmcConfig) else SmcConfig()

    # Resolve per-state params
    gs = game_input.game_state
    run_diff = gs.home_score - gs.away_score
    sk = _state_key(gs.inning, run_diff)

    if not n_lookup:
        raise RuntimeError(
            "SMC estimator requires an n_lookup artifact. "
            "Promote a calibration run to the active slot before running."
        )
    if we_array is None:
        raise RuntimeError(
            "SMC estimator requires a win_expectancy artifact for WE lookups."
        )

    n_particles = n_lookup.get("lookup", {}).get(sk, DEFAULT_N_SIMS)
    stop_thresh = (stopping_thresholds or {}).get("lookup", {}).get(sk, DEFAULT_STOPPING_THRESHOLD)
    gamma = (gamma_schedule or {}).get("lookup", {}).get(sk, cfg.gamma)

    logger.info(
        "SMC start: state=%s n_particles=%d stop_thresh=%.4f γ=%.4f",
        sk, n_particles, stop_thresh, gamma,
    )

    rng = np.random.default_rng(seed)

    try:
        ctx = init_batch_context(
            game_input, baserunning_table, outcome_labels,
            batch_prob_source, live_prob_source, rng,
            n_sims=n_particles, max_pa=max_pa, manfred_runner=manfred_runner,
            profile=profile, blowout_thresholds=blowout_thresholds,
            pitcher_exit_model=pitcher_exit_model,
            compiled_baserunning=compiled_baserunning, we_array=we_array,
        )
    except Exception:
        logger.exception("SMC init_batch_context failed for state=%s", sk)
        raise

    # Per-particle importance weights — tracks cumulative bias from resampling
    particle_weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)

    stopped_early = False
    p_estimate = None
    p_se = None
    n_boundaries = 0
    n_resamples = 0

    try:
        while ctx.bs.active.any() and ctx.pa_step < ctx.max_pa:
            step_pa(ctx)

            if (ctx.last_half_changed is not None
                    and ctx.last_half_changed.any()
                    and we_array is not None):
                active_idx = np.where(ctx.bs.active)[0]
                if len(active_idx) == 0:
                    break

                we_vals = _we_lookup_batch(ctx.bs, active_idx, we_array)
                n_boundaries += 1

                # Update cumulative importance weights
                incremental = np.power(np.clip(we_vals, 1e-10, 1.0), gamma)
                particle_weights[active_idx] *= incremental
                pw_sum = particle_weights[active_idx].sum()
                if pw_sum > 0:
                    particle_weights[active_idx] /= pw_sum

                # Compute ESS from current weights
                active_w = particle_weights[active_idx]
                active_w_norm = active_w / active_w.sum()
                ess = _compute_ess(active_w_norm)
                ess_ratio = ess / len(active_idx)

                # Importance-weighted estimate
                p_est, p_est_se = _weighted_estimate(
                    we_vals, particle_weights, active_idx,
                )

                # Stopping: SE below threshold
                if p_est_se < stop_thresh and len(active_idx) >= 10:
                    p_estimate = p_est
                    p_se = p_est_se
                    stopped_early = True
                    logger.info(
                        "SMC stopped: boundary=%d P(home)=%.4f "
                        "SE=%.4f thresh=%.4f n_active=%d ESS=%.1f",
                        n_boundaries, p_estimate, p_se,
                        stop_thresh, len(active_idx), ess,
                    )
                    break

                # Resample only if ESS < N/2 (Del Moral / Chopin)
                did_resample = False
                if ess_ratio < 0.5:
                    _resample_particles(ctx, active_idx, active_w_norm, rng)
                    particle_weights[active_idx] = 1.0 / len(active_idx)
                    did_resample = True
                    n_resamples += 1

                logger.debug(
                    "SMC boundary=%d: P=%.4f SE=%.4f ESS=%.1f ratio=%.3f "
                    "resample=%s n_active=%d",
                    n_boundaries, p_est, p_est_se, ess, ess_ratio,
                    did_resample, len(active_idx),
                )

    except Exception:
        logger.exception(
            "SMC loop failed at pa_step=%d boundary=%d state=%s n_active=%d",
            ctx.pa_step, n_boundaries, sk,
            int(ctx.bs.active.sum()),
        )
        raise

    # Build results
    results, we_snapshot, horizon_data = finalize_batch(ctx)

    if stopped_early and p_estimate is not None:
        p_home_win = p_estimate
        p_home_win_se = p_se if p_se is not None else 0.0
    else:
        # Ran to completion — importance-weighted outcome aggregation
        outcomes = np.array(
            [float(r.home_score > r.away_score) for r in results],
            dtype=np.float64,
        )
        w = particle_weights / particle_weights.sum()
        p_home_win = float(np.sum(w * outcomes))
        ess_final = _compute_ess(w)
        p_home_win_se = math.sqrt(
            p_home_win * (1 - p_home_win) / ess_final
        ) if ess_final > 1 else 0.0

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
        n_sims=n_particles,
        p_home_win=p_home_win,
        p_home_win_se=p_home_win_se,
        mean_home_score=total_home / n_particles,
        mean_away_score=total_away / n_particles,
        mean_total_runs=total_runs / n_particles,
        mean_innings=total_innings / n_particles,
        runs_per_inning=rpi,
        results=results,
        mean_timings=mean_timings,
        outcome_counts=_aggregate_outcome_counts(results),
        horizon_data=horizon_data,
    )

    logger.info(
        "SMC complete: n=%d (state=%s), P(home)=%.3f ± %.3f, "
        "boundaries=%d, resamples=%d, stopped_early=%s, γ=%.2f",
        n_particles, sk, p_home_win, p_home_win_se * 1.96,
        n_boundaries, n_resamples, stopped_early, gamma,
    )

    return sim_result
