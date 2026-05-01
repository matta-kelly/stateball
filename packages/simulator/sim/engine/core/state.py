"""Game state machine for the Markov baseball simulator.

Defines the 24 base-out states (8 runner configs × 3 outs), inning/half
progression, game-over conditions, and transition rules. Pure logic — no
I/O, no sampling, no ML. The simulator (engine.py) drives this by sampling
outcomes and calling apply_pa() with the resolved physics.

Based on Smith (2016) / Bukiet (1997) state space, extended with bookkeeping
fields for score, lineup position, and pitcher tracking.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of a baseball game at a point between PAs.

    The Markov state is (inning, half, outs, bases). Everything else is
    bookkeeping that the simulator needs to build feature vectors and
    query lookup tables.
    """

    inning: int = 1            # 1-indexed
    half: int = 0              # 0=top (away batting), 1=bot (home batting)
    outs: int = 0              # 0, 1, 2
    bases: int = 0             # bitmask 0-7: bit0=1B, bit1=2B, bit2=3B
    away_score: int = 0
    home_score: int = 0
    # Lineup cursors (0-8, index into 9-man batting order)
    away_batter_idx: int = 0
    home_batter_idx: int = 0
    # Pitcher tracking — the sim needs these to query the exit table.
    # Tracks the DEFENSIVE pitcher (the one facing the current batter).
    away_pitcher_bf: int = 0   # batters faced this game
    home_pitcher_bf: int = 0
    away_pitcher_runs: int = 0
    home_pitcher_runs: int = 0
    away_pitcher_tto: int = 1  # times through order (starts at 1)
    home_pitcher_tto: int = 1
    away_starter_pulled: bool = False
    home_starter_pulled: bool = False


# ---------------------------------------------------------------------------
# Outcome resolution
# ---------------------------------------------------------------------------


def resolve_outcome(
    outcome: str,
    state: GameState,
    baserunning_table: dict,
    rng: random.Random,
) -> tuple[int, int, int] | None:
    """Resolve a PA outcome into (post_bases, runs_scored, outs_added).

    Looks up the outcome in the baserunning table and samples a transition
    weighted by empirical probability. The table covers all 22 outcome types
    including deterministic ones (strikeout, walk, home_run) — their
    distributions have dominant entries at p≈1.0 with real edge cases
    (dropped third strikes, mid-PA steals) captured in the tail.

    Returns None if the (outcome, bases, outs) combo was never observed
    in historical data — the caller should resample a different outcome.

    Args:
        outcome: PA result string (e.g. "single", "strikeout", "walk").
        state: Current game state (bases and outs used for key).
        baserunning_table: Loaded JSON artifact with "transitions" dict.
        rng: Random source for sampling.

    Returns:
        (post_bases, runs_scored, outs_added) tuple, or None if unobserved.
    """
    key = f"{outcome}|{state.bases}|{state.outs}"
    transitions = baserunning_table["transitions"].get(key)

    if transitions is None:
        return None

    r = rng.random()
    cumulative = 0.0
    for t in transitions:
        cumulative += t["p"]
        if r < cumulative:
            return (t["post_bases"], t["runs_scored"], t["outs_added"])

    # Floating point guard — return last entry
    last = transitions[-1]
    return (last["post_bases"], last["runs_scored"], last["outs_added"])


# ---------------------------------------------------------------------------
# Accessors — avoid half-conditional boilerplate in the simulator
# ---------------------------------------------------------------------------

def batting_team(state: GameState) -> int:
    """0=away, 1=home."""
    return state.half


def current_batter_idx(state: GameState) -> int:
    """Batting team's current lineup cursor."""
    return state.away_batter_idx if state.half == 0 else state.home_batter_idx


def runners_on(state: GameState) -> int:
    """Count of runners currently on base."""
    return bin(state.bases).count("1")


def defensive_pitcher_bf(state: GameState) -> int:
    """Batters faced by the defensive pitcher."""
    return state.home_pitcher_bf if state.half == 0 else state.away_pitcher_bf


def defensive_pitcher_tto(state: GameState) -> int:
    """Times through order for the defensive pitcher."""
    return state.home_pitcher_tto if state.half == 0 else state.away_pitcher_tto


def defensive_pitcher_runs(state: GameState) -> int:
    """Runs allowed by the defensive pitcher."""
    return state.home_pitcher_runs if state.half == 0 else state.away_pitcher_runs


def defensive_starter_pulled(state: GameState) -> bool:
    """Whether the defensive team's starter has been pulled."""
    return state.home_starter_pulled if state.half == 0 else state.away_starter_pulled


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def apply_pa(
    state: GameState,
    post_bases: int,
    runs_scored: int,
    outs_added: int,
) -> GameState:
    """Apply a resolved PA result to the game state.

    The caller (simulator) is responsible for resolving the PA outcome into
    (post_bases, runs_scored, outs_added) — either via the baserunning table
    lookup or deterministic rules (strikeout, HR, walk). This function just
    applies the physics.

    Args:
        state: Current game state.
        post_bases: Bitmask (0-7) of runner positions after the PA.
        runs_scored: Runs scored on this PA.
        outs_added: Outs recorded on this PA (0, 1, or 2).

    Returns:
        New GameState reflecting the PA result.
    """
    new_outs = state.outs + outs_added

    if new_outs >= 3:
        return _flip_inning(state, runs_scored)

    return _update_mid_inning(state, post_bases, runs_scored, new_outs)


def game_over(state: GameState) -> bool:
    """Check if the game has ended.

    Two terminal conditions in baseball (9th inning or later):
    1. Walk-off — bottom half, home leads at any point.
    2. Extras resolved — a full extra inning just completed (we're now
       at top of the next inning with 0 outs), and scores differ.

    Note: "top of 9 ends, home already leads" is covered by (1) — once
    _flip_inning puts us in bot 9 with outs=0, the walk-off check fires
    immediately on the next game_over() call.
    """
    if state.inning < 9:
        return False

    # Walk-off: home leads in bottom of 9+
    if state.half == 1 and state.home_score > state.away_score:
        return True

    # Extras resolved: full extra inning complete, scores differ
    if (state.half == 0 and state.outs == 0
            and state.inning > 9 and state.home_score != state.away_score):
        return True

    return False


def completed_innings(state: GameState) -> int:
    """Return the actual last inning played for result reporting.

    When game_over() fires via the 'extras resolved' path, _flip_inning
    has already incremented the inning counter past the last real inning
    (e.g. inning=10 for a regulation away win). Subtract 1 in that case.
    Walk-off games (half=1) report correctly as-is.
    """
    if state.half == 0 and state.outs == 0 and state.inning > 9:
        return state.inning - 1
    return state.inning


def advance_lineup(state: GameState) -> GameState:
    """Advance the batting team's lineup cursor. Increment TTO on wrap.

    Called by the simulator after apply_pa(). Separate so the sim can inspect
    the post-PA state before advancing.
    """
    if state.half == 0:
        new_idx = (state.away_batter_idx + 1) % 9
        updates: dict = {"away_batter_idx": new_idx}
        # TTO increments when the lineup wraps — defensive pitcher faces
        # the top of the order again.
        if new_idx == 0:
            updates["home_pitcher_tto"] = state.home_pitcher_tto + 1
    else:
        new_idx = (state.home_batter_idx + 1) % 9
        updates = {"home_batter_idx": new_idx}
        if new_idx == 0:
            updates["away_pitcher_tto"] = state.away_pitcher_tto + 1

    return replace(state, **updates)


def mark_pitcher_pulled(state: GameState) -> GameState:
    """Mark the defensive team's starter as pulled.

    The sim calls this when the pitcher exit roll says 'pulled'. Resets
    bf/runs/tto counters for the reliever.
    """
    if state.half == 0:
        return replace(
            state,
            home_starter_pulled=True,
            home_pitcher_bf=0,
            home_pitcher_runs=0,
            home_pitcher_tto=1,
        )
    return replace(
        state,
        away_starter_pulled=True,
        away_pitcher_bf=0,
        away_pitcher_runs=0,
        away_pitcher_tto=1,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flip_inning(state: GameState, runs_scored: int) -> GameState:
    """Handle 3-out transition: clear bases, flip half/inning, add runs."""
    updates: dict = {"outs": 0, "bases": 0}

    # Add runs to batting team
    if state.half == 0:
        updates["away_score"] = state.away_score + runs_scored
        updates["home_pitcher_bf"] = state.home_pitcher_bf + 1
        updates["home_pitcher_runs"] = state.home_pitcher_runs + runs_scored
    else:
        updates["home_score"] = state.home_score + runs_scored
        updates["away_pitcher_bf"] = state.away_pitcher_bf + 1
        updates["away_pitcher_runs"] = state.away_pitcher_runs + runs_scored

    # Flip half-inning
    if state.half == 0:
        updates["half"] = 1  # top → bottom, same inning
    else:
        updates["half"] = 0  # bottom → top, next inning
        updates["inning"] = state.inning + 1

    return replace(state, **updates)


def _update_mid_inning(
    state: GameState,
    post_bases: int,
    runs_scored: int,
    new_outs: int,
) -> GameState:
    """Mid-inning update: set bases, add runs, update outs + pitcher counters."""
    updates: dict = {"bases": post_bases, "outs": new_outs}

    if state.half == 0:
        updates["away_score"] = state.away_score + runs_scored
        updates["home_pitcher_bf"] = state.home_pitcher_bf + 1
        updates["home_pitcher_runs"] = state.home_pitcher_runs + runs_scored
    else:
        updates["home_score"] = state.home_score + runs_scored
        updates["away_pitcher_bf"] = state.away_pitcher_bf + 1
        updates["away_pitcher_runs"] = state.away_pitcher_runs + runs_scored

    return replace(state, **updates)
