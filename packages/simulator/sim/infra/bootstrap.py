"""Shared Simulator construction with per-artifact slot resolution.

Encapsulates the setup steps that every runner needs: resolve artifact
paths from the registry's per-artifact slots (prod/test), load from S3,
build a ready Simulator.

Required artifact types per slot:
  - baserunning
  - xgboost_sim (sim model, no count features)
  - xgboost_live (live model, with count features)
  - win_expectancy
  - pitcher_exit

Callers still own:
  - DB connection lifecycle
  - Profile table caching (runner-specific)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sim.simulator import Simulator

logger = logging.getLogger(__name__)

_REQUIRED_ARTIFACT_TYPES = {"baserunning", "xgboost_sim", "xgboost_live", "win_expectancy", "pitcher_exit"}


def bootstrap(
    conn,
    *,
    slot: str = "prod",
    seed: int = 42,
    estimator: str = "truncated_mc",
    estimator_config=None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[Simulator, str, dict[str, str]]:
    """Resolve per-artifact slot config and build a ready Simulator.

    Args:
        conn: DuckDB connection (already connected to DuckLake).
        slot: "prod" or "test" — which slot to resolve artifacts from.
        n_sims: Monte Carlo iterations.
        seed: RNG seed.
        log_fn: Logging callback.

    Returns:
        (simulator, config_id, artifact_paths) — ready Simulator, config ID,
        and dict of {artifact_type: s3_path} for provenance tracking.

    Raises:
        RuntimeError: If any required artifact type is missing from the slot.
    """
    if log_fn is None:
        log_fn = logger.info

    from sim.infra.artifact_catalog import get_slot_artifacts
    from sim.simulator import Simulator

    slot_artifacts = get_slot_artifacts(conn, slot=slot)

    # Check all required types are present
    present = set(slot_artifacts.keys())
    missing = _REQUIRED_ARTIFACT_TYPES - present
    if missing:
        raise RuntimeError(
            f"Missing artifact types for slot '{slot}': {sorted(missing)}. "
            f"Present: {sorted(present)}. "
            "Register and promote artifacts before running simulations."
        )

    baserunning_path = slot_artifacts["baserunning"]["s3_path"]
    model_path = slot_artifacts["xgboost_sim"]["s3_path"]
    live_model_path = slot_artifacts["xgboost_live"]["s3_path"]
    win_expectancy_path = slot_artifacts["win_expectancy"]["s3_path"]
    pitcher_exit_path = slot_artifacts["pitcher_exit"]["s3_path"]
    config_id = f"slot:{slot}"

    # Optional calibration artifacts
    calibration_paths: dict[str, str | None] = {}
    for atype in ("n_lookup", "stopping_thresholds", "gamma_schedule", "horizon_weights"):
        calibration_paths[atype] = (
            slot_artifacts[atype]["s3_path"] if atype in slot_artifacts else None
        )

    log_fn(f"Resolved {len(slot_artifacts)} artifacts from {slot} slot")
    log_fn(f"  baserunning: {baserunning_path}")
    log_fn(f"  model (sim): {model_path}")
    log_fn(f"  model (live): {live_model_path}")
    log_fn(f"  win_expectancy: {win_expectancy_path}")
    log_fn(f"  pitcher_exit: {pitcher_exit_path}")
    for atype, path in calibration_paths.items():
        if path:
            log_fn(f"  {atype}: {path}")

    t0 = time.perf_counter()
    sim = Simulator.from_s3(
        baserunning_path=baserunning_path,
        model_path=model_path,
        live_model_path=live_model_path,
        win_expectancy_path=win_expectancy_path,
        pitcher_exit_path=pitcher_exit_path,
        n_lookup_path=calibration_paths["n_lookup"],
        stopping_thresholds_path=calibration_paths["stopping_thresholds"],
        gamma_schedule_path=calibration_paths["gamma_schedule"],
        horizon_weights_path=calibration_paths["horizon_weights"],
        seed=seed,
        estimator=estimator,
        estimator_config=estimator_config,
    )
    log_fn(f"Simulator loaded ({time.perf_counter() - t0:.1f}s)")

    artifact_paths = {
        "baserunning": baserunning_path,
        "xgboost_sim": model_path,
        "xgboost_live": live_model_path,
        "win_expectancy": win_expectancy_path,
        "pitcher_exit": pitcher_exit_path,
    }
    for atype, path in calibration_paths.items():
        if path:
            artifact_paths[atype] = path

    return sim, config_id, artifact_paths
