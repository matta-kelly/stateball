"""Shared types for estimators — SimulationResult."""

from __future__ import annotations

from dataclasses import dataclass

from sim.engine.core.engine import GameResult, GameTimings


@dataclass(frozen=True)
class SimulationResult:
    """Aggregate output of N simulated games from the same starting state."""

    n_sims: int
    p_home_win: float
    p_home_win_se: float
    mean_home_score: float
    mean_away_score: float
    mean_total_runs: float
    mean_innings: float
    runs_per_inning: float         # total runs / total half-innings simmed
    results: list[GameResult]
    mean_timings: GameTimings | None = None
    outcome_counts: dict[str, int] | None = None
    horizon_data: object | None = None  # HorizonData from batch engine
