"""Naive Monte Carlo estimator.

Runs N simulations of a game from a given state and aggregates the results
into win probabilities, score distributions, and confidence intervals.

Two paths:
  - Batch (default): vectorized engine runs all N sims in parallel arrays.
  - Scalar (fallback): sequential loop, one simulate_game() per sim.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from typing import Callable

import numpy as np

from sim.engine.core.engine import GameResult, GameTimings, ProbSource, simulate_game
from sim.engine.estimators.types import SimulationResult
from sim.game_inputs.game import GameInput

logger = logging.getLogger(__name__)


def _aggregate_outcome_counts(results: list[GameResult]) -> dict[str, int]:
    """Sum outcome_counts across all sim results."""
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
    """Run N simulations of a game and aggregate results.

    If batch_prob_source is provided, uses the vectorized batch engine
    (all sims in parallel). Otherwise falls back to the scalar loop.
    """
    from sim.engine.estimators.config import NaiveMcConfig
    cfg = estimator_config if isinstance(estimator_config, NaiveMcConfig) else NaiveMcConfig()
    n_sims = cfg.n_sims

    # Adaptive N: read per-state sim count from n_lookup artifact
    if cfg.adaptive_n and n_lookup:
        from sim.engine.estimators.buckets import state_key
        gs = game_input.game_state
        sk = state_key(gs.inning, gs.home_score - gs.away_score)
        n_sims = n_lookup.get("lookup", {}).get(sk, cfg.n_sims)
        logger.info("Adaptive N: state=%s → n_sims=%d (fallback=%d)", sk, n_sims, cfg.n_sims)

    if batch_prob_source is not None:
        return _run_batch(
            game_input=game_input,
            baserunning_table=baserunning_table,
            outcome_labels=outcome_labels,
            batch_prob_source=batch_prob_source,
            n_sims=n_sims,
            seed=seed,
            max_pa=max_pa,
            manfred_runner=manfred_runner,
            profile=profile,
            blowout_thresholds=blowout_thresholds,
            live_prob_source=live_prob_source,
            pitcher_exit_model=pitcher_exit_model,
            compiled_baserunning=compiled_baserunning,
            we_array=we_array,
        )
    return _run_scalar(
        game_input=game_input,
        baserunning_table=baserunning_table,
        outcome_labels=outcome_labels,
        sim_prob_source=sim_prob_source,
        n_sims=n_sims,
        seed=seed,
        max_pa=max_pa,
        manfred_runner=manfred_runner,
        profile=profile,
        live_prob_source=live_prob_source,
        pitcher_exit_model=pitcher_exit_model,
    )


def _run_batch(
    game_input: GameInput,
    baserunning_table: dict,
    outcome_labels: list[str],
    batch_prob_source: Callable,
    n_sims: int,
    seed: int,
    max_pa: int,
    manfred_runner: bool,
    profile: bool,
    blowout_thresholds=None,
    live_prob_source=None,
    pitcher_exit_model=None,
    compiled_baserunning=None,
    we_array: np.ndarray | None = None,
) -> SimulationResult:
    """Vectorized path: all sims run in parallel arrays."""
    from sim.engine.core.batch_engine import simulate_game_batch

    rng = np.random.default_rng(seed)

    results, we_snapshot, horizon_data = simulate_game_batch(
        game_input=game_input,
        baserunning_table=baserunning_table,
        outcome_labels=outcome_labels,
        batch_prob_source=batch_prob_source,
        rng=rng,
        n_sims=n_sims,
        max_pa=max_pa,
        manfred_runner=manfred_runner,
        profile=profile,
        blowout_thresholds=blowout_thresholds,
        live_prob_source=live_prob_source,
        pitcher_exit_model=pitcher_exit_model,
        compiled_baserunning=compiled_baserunning,
        we_array=we_array,
    )

    # Aggregate from results list
    home_wins = sum(1 for r in results if r.home_score > r.away_score)
    total_home = sum(r.home_score for r in results)
    total_away = sum(r.away_score for r in results)
    total_runs = total_home + total_away
    total_innings = sum(r.innings for r in results)

    p_home_win = home_wins / n_sims
    p_home_win_se = math.sqrt(p_home_win * (1 - p_home_win) / n_sims)

    # Timings: batch engine shares one GameTimings across all results
    mean_timings = results[0].timings if results and results[0].timings else None

    # Runs per inning: total runs / total half-innings simmed (each team's
    # contribution). A pruned sim that played 6 innings contributes 12
    # half-innings of run data, not 18.
    total_half_innings = sum(
        r.innings * 2 - (1 if r.home_score > r.away_score and r.innings >= 9 else 0)
        for r in results
    )
    rpi = total_runs / total_half_innings if total_half_innings > 0 else 0.0

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

    # Pruning stats
    n_pruned = sum(1 for r in results if r.pruned)
    prune_msg = ""
    if n_pruned:
        pruned_pas = [r.pruned_at_pa for r in results if r.pruned]
        mean_prune_pa = sum(pruned_pas) / len(pruned_pas)
        prune_msg = (
            f", pruned {n_pruned}/{n_sims} ({100 * n_pruned / n_sims:.1f}%)"
            f" mean PA@prune {mean_prune_pa:.0f}"
        )

    # Max PA safety valve stats
    n_capped = sum(1 for r in results if r.total_pas >= max_pa)
    cap_msg = ""
    if n_capped:
        cap_msg = f", max_pa capped {n_capped}/{n_sims} ({100 * n_capped / n_sims:.1f}%)"
        logger.warning(
            "%d/%d sims hit max_pa safety valve (%d) — possible engine bug",
            n_capped, n_sims, max_pa,
        )

    logger.info(
        "Naive MC complete (batch): n=%d, P(home)=%.3f ± %.3f, "
        "mean score %.1f-%.1f, mean innings %.1f, runs/inn %.3f%s%s",
        n_sims,
        sim_result.p_home_win,
        sim_result.p_home_win_se * 1.96,
        sim_result.mean_home_score,
        sim_result.mean_away_score,
        sim_result.mean_innings,
        sim_result.runs_per_inning,
        prune_msg,
        cap_msg,
    )

    return sim_result


def _run_scalar(
    game_input: GameInput,
    baserunning_table: dict,
    outcome_labels: list[str],
    sim_prob_source: ProbSource,
    n_sims: int,
    seed: int,
    max_pa: int,
    manfred_runner: bool,
    profile: bool,
    live_prob_source=None,
    pitcher_exit_model=None,
) -> SimulationResult:
    """Original scalar path: one simulate_game() per sim."""
    results: list[GameResult] = []
    home_wins = 0
    total_home_score = 0
    total_away_score = 0
    total_runs = 0
    total_innings = 0

    milestones = {int(n_sims * p): p for p in (0.20, 0.50, 0.75)}

    for i in range(n_sims):
        result = simulate_game(
            game_input=game_input,
            baserunning_table=baserunning_table,

            outcome_labels=outcome_labels,
            sim_prob_source=sim_prob_source,
            rng=random.Random(seed + i),
            max_pa=max_pa,
            manfred_runner=manfred_runner,
            profile=profile,
            live_prob_source=live_prob_source,
            pitcher_exit_model=pitcher_exit_model,
        )
        results.append(result)

        if result.home_score > result.away_score:
            home_wins += 1
        total_home_score += result.home_score
        total_away_score += result.away_score
        total_runs += result.home_score + result.away_score
        total_innings += result.innings

        if i + 1 in milestones:
            p_so_far = home_wins / (i + 1)
            logger.info(
                "  %d%% (%d/%d) — P(home)=%.3f so far",
                int(milestones[i + 1] * 100), i + 1, n_sims, p_so_far,
            )

    p_home_win = home_wins / n_sims
    p_home_win_se = math.sqrt(p_home_win * (1 - p_home_win) / n_sims)

    # Aggregate profiling timings
    mean_timings = None
    if profile and results and results[0].timings is not None:
        _fields = ["prob_source", "context_build", "pitcher_exit",
                    "outcome_sample", "state_transition", "total"]
        avg = {f: sum(getattr(r.timings, f) for r in results) / n_sims
               for f in _fields}
        avg["n_pas"] = round(sum(r.timings.n_pas for r in results) / n_sims)
        mean_timings = GameTimings(**avg)

    total_half_innings = sum(
        r.innings * 2 - (1 if r.home_score > r.away_score and r.innings >= 9 else 0)
        for r in results
    )
    rpi = total_runs / total_half_innings if total_half_innings > 0 else 0.0

    sim_result = SimulationResult(
        n_sims=n_sims,
        p_home_win=p_home_win,
        p_home_win_se=p_home_win_se,
        mean_home_score=total_home_score / n_sims,
        mean_away_score=total_away_score / n_sims,
        mean_total_runs=total_runs / n_sims,
        mean_innings=total_innings / n_sims,
        runs_per_inning=rpi,
        results=results,
        mean_timings=mean_timings,
        outcome_counts=_aggregate_outcome_counts(results),
    )

    # Max PA safety valve stats
    n_capped = sum(1 for r in results if r.total_pas >= max_pa)
    cap_msg = ""
    if n_capped:
        cap_msg = f", max_pa capped {n_capped}/{n_sims} ({100 * n_capped / n_sims:.1f}%)"
        logger.warning(
            "%d/%d sims hit max_pa safety valve (%d) — possible engine bug",
            n_capped, n_sims, max_pa,
        )

    logger.info(
        "Naive MC complete (scalar): n=%d, P(home)=%.3f ± %.3f, "
        "mean score %.1f-%.1f, mean innings %.1f, runs/inn %.3f%s",
        n_sims,
        sim_result.p_home_win,
        sim_result.p_home_win_se * 1.96,
        sim_result.mean_home_score,
        sim_result.mean_away_score,
        sim_result.mean_innings,
        sim_result.runs_per_inning,
        cap_msg,
    )

    return sim_result
