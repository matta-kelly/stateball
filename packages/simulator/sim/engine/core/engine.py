"""Game simulation engine for the Markov baseball simulator.

Drives the PA loop: samples outcomes via a probability source, resolves
baserunning transitions, manages pitcher exits and bullpen, tracks outing
counters. Pure Python — no I/O, no ML imports. The probability source is
a callable injected by the caller (uniform for testing, XGBoost for prod).

Usage:
    prob_source = make_uniform_prob_source(len(labels))
    result = simulate_game(
        game_input=gi,
        baserunning_table=brt,
        outcome_labels=labels,
        sim_prob_source=prob_source,
        live_prob_source=prob_source,
    )
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Callable

from sim.game_inputs.game import GameInput
from sim.engine.core.state import (
    GameState,
    advance_lineup,
    apply_pa,
    batting_team,
    completed_innings,
    current_batter_idx,
    defensive_pitcher_bf,
    defensive_pitcher_tto,
    game_over,
    mark_pitcher_pulled,
    resolve_outcome,
)
from sim.engine.lookups.re_table import RE_TABLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ProbSource = Callable[[dict, dict, GameState, dict], list[float]]


@dataclass
class GameTimings:
    """Accumulated timing breakdown for one simulated game."""

    prob_source: float = 0.0
    context_build: float = 0.0
    pitcher_exit: float = 0.0
    outcome_sample: float = 0.0
    state_transition: float = 0.0
    id_resolve: float = 0.0
    total: float = 0.0
    n_pas: int = 0


@dataclass(frozen=True)
class GameResult:
    """Output of a single simulated game."""

    home_score: int
    away_score: int
    innings: int
    total_pas: int
    home_pitcher_changes: int
    away_pitcher_changes: int
    timings: GameTimings | None = None
    outcome_counts: dict[str, int] | None = None
    pruned: bool = False
    pruned_at_pa: int = 0
    we_at_end: float = -1.0  # WE at last PA state (-1 = not available)


# ---------------------------------------------------------------------------
# Outcome classification sets (for outing counter updates)
# ---------------------------------------------------------------------------

_WALK_OUTCOMES = frozenset({"walk", "intent_walk", "hit_by_pitch"})
_HIT_OUTCOMES = frozenset({"single", "double", "triple", "home_run"})
_K_OUTCOMES = frozenset({"strikeout", "strikeout_double_play"})
_WALK_OR_HIT = _WALK_OUTCOMES | _HIT_OUTCOMES


# ---------------------------------------------------------------------------
# Pitcher outing tracker (mutable, outside frozen GameState)
# ---------------------------------------------------------------------------


@dataclass
class _PitcherOuting:
    walks: int = 0
    hits: int = 0
    k: int = 0
    runs: int = 0
    recent: deque = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.recent is None:
            self.recent = deque(maxlen=9)

    def whip(self, bf: int) -> float:
        return (self.walks + self.hits) / bf if bf > 0 else 0.0

    def recent_whip(self) -> float:
        if not self.recent:
            return 0.0
        wh = sum(1 for o in self.recent if o in _WALK_OR_HIT)
        return wh / len(self.recent)

    def reset(self):
        self.walks = 0
        self.hits = 0
        self.k = 0
        self.runs = 0
        self.recent.clear()


# ---------------------------------------------------------------------------
# Pitcher exit
# ---------------------------------------------------------------------------
# When pitcher_exit_model is provided, uses a trained XGBoost binary
# classifier (PitcherExitModel) with 17 game-state features.
# Falls back to a flat per-PA pull probability heuristic when no model
# is available.

_STARTER_BF_THRESHOLD = 15   # don't even consider pulling before this
_RELIEVER_BF_THRESHOLD = 3
_BASE_PULL_PROB = 0.04       # per-PA probability once past threshold


def _placeholder_pull_prob(bf: int, is_starter: bool) -> float:
    """Flat pull probability based on BF and role. Placeholder for v2 model."""
    threshold = _STARTER_BF_THRESHOLD if is_starter else _RELIEVER_BF_THRESHOLD
    if bf < threshold:
        return 0.0
    return _BASE_PULL_PROB


def _model_pull_prob(
    pitcher_exit_model,
    state: GameState,
    outing: "_PitcherOuting",
    pitcher: dict,
    is_starter: bool,
    bf: int,
    prev_re: float,
) -> float:
    """Compute P(pulled) using the trained pitcher exit model.

    Builds a feature dict from engine state and calls model.predict().
    """
    runners_bitmask = int(
        (state.bases & 1)
        + ((state.bases >> 1) & 1) * 2
        + ((state.bases >> 2) & 1) * 4
    )
    outs_clamped = min(state.outs, 2)
    current_re = float(RE_TABLE[runners_bitmask, outs_clamped])
    runners_on = bin(state.bases).count("1")

    # run_diff from batting team perspective (same as game_state.run_diff)
    side = batting_team(state)
    if side == 0:  # away batting, home pitching
        run_diff = state.away_score - state.home_score
    else:  # home batting, away pitching
        run_diff = state.home_score - state.away_score

    features = {
        "pitcher_bf_game": float(bf),
        "starter_flag": 1.0 if is_starter else 0.0,
        "outing_runs": float(outing.runs),
        "inning": float(state.inning),
        "run_diff": float(run_diff),
        "outs": float(state.outs),
        "runners_on": float(runners_on),
        "outing_walks": float(outing.walks),
        "outing_hits": float(outing.hits),
        "outing_k": float(outing.k),
        "times_through_order": float(defensive_pitcher_tto(state)),
        "outing_whip": outing.whip(bf),
        "pitcher_recent_whip": outing.recent_whip(),
        "current_re": current_re,
        "re_diff": current_re - prev_re,
        "avg_bf_per_app": float(pitcher.get("career_avg_bf_per_app", pitcher.get("avg_bf_per_app", 20.0))),
        "pit_rest_days": float(pitcher.get("pitcher_rest_days", pitcher.get("pit_rest_days", -1))),
    }

    return pitcher_exit_model.predict(features)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_outcome(
    probs: list[float], labels: list[str], rng: random.Random
) -> str:
    r = rng.random()
    cumulative = 0.0
    for p, label in zip(probs, labels):
        cumulative += p
        if r < cumulative:
            return label
    return labels[-1]


def _apply_manfred_runner(state: GameState) -> GameState:
    """Place ghost runner on 2B at start of extra-inning half."""
    if state.inning >= 10 and state.outs == 0 and state.bases == 0:
        return replace(state, bases=state.bases | 0b010)
    return state


def _update_outing(outing: _PitcherOuting, outcome: str, runs_scored: int):
    """Update pitcher outing counters after a PA."""
    if outcome in _WALK_OUTCOMES:
        outing.walks += 1
    if outcome in _HIT_OUTCOMES:
        outing.hits += 1
    if outcome in _K_OUTCOMES:
        outing.k += 1
    outing.runs += runs_scored
    outing.recent.append(outcome)


def _build_context(
    state: GameState,
    outing: _PitcherOuting,
    batter_pa_counts: dict[int, int],
    matchup_counts: dict[tuple[int, int], int],
    batter_id: int,
    pitcher_id: int,
) -> dict:
    """Assemble the context dict passed to prob_source."""
    bf = defensive_pitcher_bf(state)
    return {
        "times_through_order": defensive_pitcher_tto(state),
        "batter_prior_pa": batter_pa_counts.get(batter_id, 0),
        "pitcher_bf_game": bf,
        "batter_ab_vs_pitcher": matchup_counts.get((batter_id, pitcher_id), 0),
        "pitcher_outing_walks": outing.walks,
        "pitcher_outing_hits": outing.hits,
        "pitcher_outing_k": outing.k,
        "pitcher_outing_runs": outing.runs,
        "pitcher_outing_whip": outing.whip(bf),
        "pitcher_recent_whip": outing.recent_whip(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_uniform_prob_source(n_classes: int) -> ProbSource:
    """Create a uniform probability source over n outcome classes. For testing."""
    uniform = [1.0 / n_classes] * n_classes

    def prob_source(
        batter: dict,
        pitcher: dict,
        state: GameState,
        context: dict,
    ) -> list[float]:
        return uniform

    return prob_source


def simulate_game(
    game_input: GameInput,
    baserunning_table: dict,
    outcome_labels: list[str],
    sim_prob_source: ProbSource,
    live_prob_source: ProbSource,
    rng: random.Random | None = None,
    max_pa: int = 200,
    manfred_runner: bool = True,
    profile: bool = False,
    pitcher_exit_model=None,
) -> GameResult:
    """Simulate a single baseball game from the given starting state.

    Args:
        game_input: Lineups, pitchers, bullpens, initial state.
        baserunning_table: JSON artifact with runner advancement distributions.
        outcome_labels: Ordered list of PA outcome class names.
        sim_prob_source: Callable returning probability distributions (sim model, no count features).
        rng: Random source (seeded for reproducibility).
        max_pa: Safety valve — stop after this many PAs.
        manfred_runner: Place ghost runner on 2B in extras (inning >= 10).
        live_prob_source: Callable for seed PA (live model with count features).

    Returns:
        GameResult with final scores and game metadata.
    """
    if rng is None:
        rng = random.Random()

    state = game_input.game_state

    # If the seed state has outs >= 3 (e.g. live in-play adjustment detected
    # a terminal out before MLB flipped the half), flip the inning first.
    # Direct replace() instead of apply_pa to avoid phantom BF increment.
    if state.outs >= 3:
        if state.half == 0:
            state = replace(state, half=1, outs=0, bases=0)
        else:
            state = replace(state, half=0, inning=state.inning + 1, outs=0, bases=0)
        if game_over(state):
            return GameResult(
                home_score=state.home_score,
                away_score=state.away_score,
                innings=completed_innings(state),
                pitcher_changes={"home": 0, "away": 0},
                total_pas=0,
                outcome_counts={},
                timings=None,
            )

    # Mutable copies of bullpen lists — we pop from these
    home_bullpen = list(game_input.home_bullpen)
    away_bullpen = list(game_input.away_bullpen)

    # Current pitchers (may change during the game)
    home_pitcher = game_input.home_pitcher
    away_pitcher = game_input.away_pitcher

    # Outing counters per side
    home_outing = _PitcherOuting()
    away_outing = _PitcherOuting()

    # Track whether current pitcher on each side is the starter
    home_is_starter = True
    away_is_starter = True

    # Engine-tracked counters
    batter_pa_counts: dict[int, int] = defaultdict(int)
    matchup_counts: dict[tuple[int, int], int] = defaultdict(int)
    outcome_counts: dict[str, int] = defaultdict(int)
    home_pitcher_changes = 0
    away_pitcher_changes = 0
    total_pas = 0

    # Track half-inning transitions for Manfred runner
    prev_inning = state.inning
    prev_half = state.half

    # Track previous RE for re_diff (pitcher exit model feature).
    # Reset to bases-empty/0-outs RE at start of each half-inning.
    _base_re = float(RE_TABLE[0, 0])  # bases empty, 0 outs
    home_prev_re = _base_re
    away_prev_re = _base_re

    # Profiling: gate all timing on a local variable (zero-cost when off)
    _perf = time.perf_counter if profile else None
    _timings = GameTimings() if profile else None
    if _perf:
        _t_game = _perf()

    # --- Seed PA (live model with real count data) ---
    # Use real dynamic features from the vector instead of fabricating zeros.
    # seed_context contains dynamic game_state fields + balls/strikes.
    side = batting_team(state)
    lineup = game_input.away_lineup if side == 0 else game_input.home_lineup
    batter = lineup[current_batter_idx(state)]
    pitcher = home_pitcher if side == 0 else away_pitcher
    outing = home_outing if side == 0 else away_outing

    batter_id = batter["player_id"]
    pitcher_id = pitcher["player_id"]

    ctx = game_input.seed_context

    probs = live_prob_source(batter, pitcher, state, ctx)

    for _attempt in range(10):
        outcome = _sample_outcome(probs, outcome_labels, rng)
        resolved = resolve_outcome(outcome, state, baserunning_table, rng)
        if resolved is not None:
            post_bases, runs_scored, outs_added = resolved
            break
    else:
        outcome = "field_out"
        post_bases, runs_scored, outs_added = resolve_outcome(
            outcome, state, baserunning_table, rng,
        )

    outcome_counts[outcome] += 1
    _update_outing(outing, outcome, runs_scored)
    batter_pa_counts[batter_id] += 1
    matchup_counts[(batter_id, pitcher_id)] += 1
    state = apply_pa(state, post_bases, runs_scored, outs_added)
    state = advance_lineup(state)
    total_pas += 1

    while not game_over(state) and total_pas < max_pa:
        # --- Manfred runner on half-inning change ---
        if manfred_runner and (
            state.inning != prev_inning or state.half != prev_half
        ):
            state = _apply_manfred_runner(state)
        prev_inning = state.inning
        prev_half = state.half

        # --- Identify participants ---
        side = batting_team(state)  # 0=away batting, 1=home batting
        lineup = game_input.away_lineup if side == 0 else game_input.home_lineup
        batter = lineup[current_batter_idx(state)]

        # Defensive pitcher (facing the batter)
        if side == 0:
            pitcher = home_pitcher
            outing = home_outing
            bullpen = home_bullpen
        else:
            pitcher = away_pitcher
            outing = away_outing
            bullpen = away_bullpen

        # --- Pitcher exit check (all pitchers) ---
        if _perf:
            _t0 = _perf()
        is_starter = home_is_starter if side == 0 else away_is_starter
        bf = defensive_pitcher_bf(state)
        if pitcher_exit_model is not None:
            prev_re = home_prev_re if side == 0 else away_prev_re
            p_pull = _model_pull_prob(
                pitcher_exit_model, state, outing, pitcher,
                is_starter, bf, prev_re,
            )
        else:
            p_pull = _placeholder_pull_prob(bf, is_starter)
        if rng.random() < p_pull:
            if bullpen:
                new_pitcher = rng.choice(bullpen)
                bullpen.remove(new_pitcher)
            else:
                new_pitcher = {"player_id": -1, "hand": "R"}
                logger.warning(
                    "Bullpen exhausted for %s side — using generic reliever",
                    "home" if side == 0 else "away",
                )

            if side == 0:
                home_pitcher = new_pitcher
                home_outing.reset()
                home_pitcher_changes += 1
                home_is_starter = False
            else:
                away_pitcher = new_pitcher
                away_outing.reset()
                away_pitcher_changes += 1
                away_is_starter = False

            # Re-bind after change
            pitcher = new_pitcher
            outing = home_outing if side == 0 else away_outing
            state = mark_pitcher_pulled(state)
        if _perf:
            _timings.pitcher_exit += _perf() - _t0

        # --- Get probability distribution ---
        if _perf:
            _t0 = _perf()
        batter_id = batter["player_id"]
        pitcher_id = pitcher["player_id"]
        ctx = _build_context(
            state, outing, batter_pa_counts, matchup_counts,
            batter_id, pitcher_id,
        )
        if _perf:
            _timings.context_build += _perf() - _t0

        if _perf:
            _t0 = _perf()
        probs = sim_prob_source(batter, pitcher, state, ctx)
        if _perf:
            _timings.prob_source += _perf() - _t0

        # --- Sample outcome (reroll if baserunning combo unobserved) ---
        if _perf:
            _t0 = _perf()
        for _attempt in range(10):
            outcome = _sample_outcome(probs, outcome_labels, rng)
            resolved = resolve_outcome(outcome, state, baserunning_table, rng)
            if resolved is not None:
                post_bases, runs_scored, outs_added = resolved
                break
        else:
            outcome = "field_out"
            post_bases, runs_scored, outs_added = resolve_outcome(
                outcome, state, baserunning_table, rng,
            )
        if _perf:
            _timings.outcome_sample += _perf() - _t0

        # --- Update outing counters + apply PA + advance lineup ---
        if _perf:
            _t0 = _perf()
        outcome_counts[outcome] += 1
        _update_outing(outing, outcome, runs_scored)
        batter_pa_counts[batter_id] += 1
        matchup_counts[(batter_id, pitcher_id)] += 1

        # Capture current RE before state change (for next PA's re_diff)
        _cur_re_bases = int(
            (state.bases & 1)
            + ((state.bases >> 1) & 1) * 2
            + ((state.bases >> 2) & 1) * 4
        )
        _cur_re = float(RE_TABLE[_cur_re_bases, min(state.outs, 2)])

        state = apply_pa(state, post_bases, runs_scored, outs_added)
        state = advance_lineup(state)

        # Update prev_re for the defending side; reset on half-inning flip
        if state.inning != prev_inning or state.half != prev_half:
            # Half-inning changed — reset both sides to base RE
            home_prev_re = _base_re
            away_prev_re = _base_re
        else:
            if side == 0:
                home_prev_re = _cur_re
            else:
                away_prev_re = _cur_re

        if _perf:
            _timings.state_transition += _perf() - _t0

        total_pas += 1

    if total_pas >= max_pa:
        logger.warning(
            "Game hit max_pa safety valve (%d PAs) — inning=%d, score=%d-%d",
            max_pa, state.inning, state.home_score, state.away_score,
        )

    if _perf:
        _timings.total = _perf() - _t_game
        _timings.n_pas = total_pas

    return GameResult(
        home_score=state.home_score,
        away_score=state.away_score,
        innings=completed_innings(state),
        total_pas=total_pas,
        home_pitcher_changes=home_pitcher_changes,
        away_pitcher_changes=away_pitcher_changes,
        timings=_timings,
        outcome_counts=dict(outcome_counts),
    )
