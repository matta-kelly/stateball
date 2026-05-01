"""Estimator layer — pluggable simulation strategies.

Available estimators:
  - naive_mc: brute force N sims to completion (supports adaptive N)
  - smc: sequential Monte Carlo with particle management
  - truncated_mc: sim to horizon H, return mean(WE_H) — default for production
"""

from sim.engine.estimators.config import (
    DEFAULT_N_SIMS,
    DEFAULT_STOPPING_THRESHOLD,
    NaiveMcConfig,
    SmcConfig,
    TruncatedMcConfig,
    get_default_config,
)
from sim.engine.estimators.types import SimulationResult

__all__ = [
    "DEFAULT_N_SIMS", "DEFAULT_STOPPING_THRESHOLD",
    "NaiveMcConfig", "SimulationResult", "SmcConfig", "TruncatedMcConfig",
    "get_default_config", "get_estimator",
]


def get_estimator(name: str = "naive_mc"):
    """Return the estimate() callable for the named strategy."""
    if name == "naive_mc":
        from sim.engine.estimators.naive_mc import estimate
        return estimate
    if name == "smc":
        from sim.engine.estimators.smc import estimate
        return estimate
    if name == "truncated_mc":
        from sim.engine.estimators.truncated_mc import estimate
        return estimate
    raise ValueError(f"Unknown estimator: {name!r}")
