"""Simulator shell — loaded once, simulates many games.

The Simulator holds all pre-loaded artifacts (baserunning table, XGBoost
sim/live prob sources) and Monte Carlo config. Call simulate() with a GameInput to
get a SimulationResult. The Simulator doesn't know about the warehouse
or where GameInputs come from.
"""

from __future__ import annotations

import logging

from sim.engine.core.engine import ProbSource
from sim.game_inputs.game import GameInput
from sim.engine.estimators import SimulationResult, get_default_config, get_estimator
from sim.engine.lookups.compiled_baserunning import CompiledBaserunning, compile_baserunning_table

logger = logging.getLogger(__name__)


class Simulator:
    """Loaded simulation shell. Set up once, simulate many games."""

    def __init__(
        self,
        baserunning_table: dict,
        sim_prob_source: ProbSource,
        outcome_labels: list[str],
        *,
        seed: int = 42,
        max_pa: int = 200,
        manfred_runner: bool = True,
        batch_prob_source=None,
        blowout_thresholds=None,
        live_prob_source: ProbSource,
        pitcher_exit_model=None,
        compiled_baserunning: CompiledBaserunning | None = None,
        we_table: dict | None = None,
        n_lookup: dict | None = None,
        stopping_thresholds: dict | None = None,
        gamma_schedule: dict | None = None,
        horizon_weights: dict | None = None,
        estimator: str = "naive_mc",
        estimator_config=None,
    ):
        self.baserunning_table = baserunning_table
        self.compiled_baserunning = compiled_baserunning

        self.sim_prob_source = sim_prob_source
        self.outcome_labels = outcome_labels
        self.seed = seed
        self.max_pa = max_pa
        self.manfred_runner = manfred_runner
        self.batch_prob_source = batch_prob_source
        self.blowout_thresholds = blowout_thresholds
        self.live_prob_source = live_prob_source
        self.pitcher_exit_model = pitcher_exit_model
        self.we_table = we_table
        self.n_lookup = n_lookup
        self.stopping_thresholds = stopping_thresholds
        self.gamma_schedule = gamma_schedule
        self.horizon_weights = horizon_weights
        self.estimator_config = estimator_config or get_default_config(estimator)
        self.we_array = None
        self.sensitivity_array = None

        # Build dense WE lookup + sensitivity arrays (once at startup)
        if we_table is not None:
            from sim.engine.lookups.win_expectancy import build_lookup_array, build_sensitivity_array
            self.we_array = build_lookup_array(we_table)
            self.sensitivity_array = build_sensitivity_array(we_table)

        # Auto-compile if not provided and batch mode is active
        if self.compiled_baserunning is None and batch_prob_source is not None:
            self.compiled_baserunning = compile_baserunning_table(
                baserunning_table, outcome_labels,
            )

        self._estimator = get_estimator(estimator)

        mode = "batch" if batch_prob_source is not None else "scalar"
        logger.info(
            "Simulator ready: %d outcomes, seed=%d, mode=%s, estimator=%s, config=%s",
            len(outcome_labels), seed, mode, estimator, self.estimator_config,
        )

    def simulate(self, game_input: GameInput, *, profile: bool = False) -> SimulationResult:
        """Run the configured estimator on a single GameInput."""
        cfg = self.estimator_config
        enable_pruning = getattr(cfg, "enable_pruning", False)
        return self._estimator(
            game_input=game_input,
            baserunning_table=self.baserunning_table,
            outcome_labels=self.outcome_labels,
            sim_prob_source=self.sim_prob_source,
            seed=self.seed,
            max_pa=self.max_pa,
            manfred_runner=self.manfred_runner,
            profile=profile,
            batch_prob_source=self.batch_prob_source,
            blowout_thresholds=self.blowout_thresholds if enable_pruning else None,
            live_prob_source=self.live_prob_source,
            pitcher_exit_model=self.pitcher_exit_model,
            compiled_baserunning=self.compiled_baserunning,
            we_array=self.we_array,
            we_table=self.we_table,
            n_lookup=self.n_lookup,
            stopping_thresholds=self.stopping_thresholds,
            gamma_schedule=self.gamma_schedule,
            horizon_weights=self.horizon_weights,
            estimator_config=cfg,
        )

    @classmethod
    def from_s3(
        cls,
        baserunning_path: str,
        model_path: str,
        live_model_path: str,
        win_expectancy_path: str | None = None,
        pitcher_exit_path: str | None = None,
        n_lookup_path: str | None = None,
        stopping_thresholds_path: str | None = None,
        gamma_schedule_path: str | None = None,
        horizon_weights_path: str | None = None,
        **kwargs,
    ) -> Simulator:
        """Load S3 artifacts and build a ready Simulator."""
        from sim.infra.artifact_loaders import (
            build_blowout_thresholds,
            load_baserunning_table,
            load_model,
            load_pitcher_exit_model,
            load_win_expectancy_table,
        )
        from sim.engine.core.prob import make_batch_prob_source, make_prob_source

        baserunning_table = load_baserunning_table(baserunning_path)
        bundle = load_model(model_path)
        prob_source = make_prob_source(bundle)

        batch_prob_source = None
        if bundle.onnx_session is not None:
            batch_prob_source = make_batch_prob_source(bundle)

        blowout_thresholds = None
        we_table = None
        if win_expectancy_path is not None:
            we_table = load_win_expectancy_table(win_expectancy_path)
            blowout_thresholds = build_blowout_thresholds(we_table)

        pitcher_exit_model = None
        if pitcher_exit_path is not None:
            pitcher_exit_model = load_pitcher_exit_model(pitcher_exit_path)
            logger.info("Pitcher exit model loaded from %s", pitcher_exit_path)

        live_bundle = load_model(live_model_path)
        assert live_bundle.outcome_labels == bundle.outcome_labels, (
            f"Live/sim model outcome labels mismatch: "
            f"{live_bundle.outcome_labels} != {bundle.outcome_labels}"
        )
        live_prob_source = make_prob_source(live_bundle)
        logger.info("Live model loaded from %s", live_model_path)

        # Load calibration artifacts
        from sim.infra.artifact_loaders import (
            load_n_lookup, load_stopping_thresholds,
            load_gamma_schedule, load_horizon_weights,
        )
        n_lookup = load_n_lookup(n_lookup_path) if n_lookup_path else None
        stopping_thresholds = load_stopping_thresholds(stopping_thresholds_path) if stopping_thresholds_path else None
        gamma_schedule = load_gamma_schedule(gamma_schedule_path) if gamma_schedule_path else None
        horizon_weights = load_horizon_weights(horizon_weights_path) if horizon_weights_path else None

        return cls(
            baserunning_table=baserunning_table,
            sim_prob_source=prob_source,
            outcome_labels=bundle.outcome_labels,
            batch_prob_source=batch_prob_source,
            blowout_thresholds=blowout_thresholds,
            live_prob_source=live_prob_source,
            pitcher_exit_model=pitcher_exit_model,
            we_table=we_table,
            n_lookup=n_lookup,
            stopping_thresholds=stopping_thresholds,
            gamma_schedule=gamma_schedule,
            horizon_weights=horizon_weights,
            **kwargs,
        )
