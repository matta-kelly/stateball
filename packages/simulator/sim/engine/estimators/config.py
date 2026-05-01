"""Estimator configuration dataclasses.

Each estimator has its own frozen dataclass with sensible defaults.
Complex derived data (lookup tables, threshold tables) are artifacts
loaded through the standard artifact pipeline — not config.

Shared defaults used when calibration artifacts are missing a state key.
"""

from __future__ import annotations

from dataclasses import dataclass

# Shared fallback defaults — used when calibration artifacts have no
# entry for a given game state.
DEFAULT_N_SIMS = 1000
DEFAULT_STOPPING_THRESHOLD = 0.01


@dataclass(frozen=True)
class NaiveMcConfig:
    """Config for naive Monte Carlo estimator."""

    n_sims: int = DEFAULT_N_SIMS
    adaptive_n: bool = False
    enable_pruning: bool = False


@dataclass(frozen=True)
class SmcConfig:
    """Config for Sequential Monte Carlo estimator.

    SMC-specific tuning only. N and stopping thresholds come from
    calibration artifacts; gamma defaults here are used when the
    gamma_schedule artifact has no entry for a state.
    """

    gamma: float = 0.3


@dataclass(frozen=True)
class TruncatedMcConfig:
    """Config for truncated Monte Carlo estimator.

    Sims to horizon H, returns mean(WE_H). Consumes n_lookup for
    adaptive N and horizon_weights for per-state H.
    """

    n_sims: int = DEFAULT_N_SIMS
    adaptive_n: bool = True
    enable_pruning: bool = True
    max_horizon: int | None = 1  # cap on half-inning horizons (None = use artifact as-is)


def get_default_config(estimator: str) -> NaiveMcConfig | SmcConfig | TruncatedMcConfig:
    """Return the default config for the named estimator."""
    if estimator == "naive_mc":
        return NaiveMcConfig()
    if estimator == "smc":
        return SmcConfig()
    if estimator == "truncated_mc":
        return TruncatedMcConfig()
    raise ValueError(f"Unknown estimator: {estimator!r}")
