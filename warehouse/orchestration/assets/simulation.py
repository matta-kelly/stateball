"""Simulation eval assets: calibration and production evaluation."""
from dagster import asset, AssetExecutionContext

from ._shared import load_secrets
from ._configs import CalibrationConfig, ProductionConfig


@asset(group_name="simulation")
def calibration_eval(context: AssetExecutionContext, config: CalibrationConfig):
    """Full-completion naive MC eval — populates DuckLake tables for artifact builders.

    Runs every sim to game end with fixed N (no adaptive N, no pruning).
    Data lands in sim_eval_convergence + sim_eval_horizons tables.
    Run the 4 build assets afterward to produce artifacts from this data.
    """
    load_secrets()
    from .eval_runner import run

    result, _sim = run(
        n_per_inning=config.n_per_inning,
        n_sims=config.n_sims,
        seed=config.seed,
        log_fn=context.log.info,
        profile=True,
        slot=config.slot,
        estimator="naive_mc",
        adaptive_n=False,
        enable_pruning=False,
    )
    context.log.info(
        f"Calibration complete: {result['correct']}/{result['n_games']} correct "
        f"({result['accuracy']:.0%}), mean MC time {result['mean_mc_time']:.1f}s"
    )
    context.add_output_metadata({
        "accuracy": result["accuracy"],
        "n_games": result["n_games"],
        "n_per_inning": config.n_per_inning,
        "n_sims": config.n_sims,
        "mean_mc_time_s": result["mean_mc_time"],
        "total_wall_time_s": result["total_time"],
    })


@asset(
    group_name="simulation",
    deps=["build_n_lookup_artifact", "build_stopping_thresholds_artifact",
          "build_gamma_schedule_artifact", "build_horizon_weights_artifact"],
)
def production_eval(context: AssetExecutionContext, config: ProductionConfig):
    """Run production estimator eval.

    Consumes calibration artifacts (n_lookup, horizon_weights, etc.).
    Fast path — adaptive N, pruning, truncated horizons.
    """
    load_secrets()
    from .eval_runner import run

    # Build estimator config with max_horizon if set
    estimator_config = None
    if config.estimator == "truncated_mc" and config.max_horizon > 0:
        from sim.engine.estimators.config import TruncatedMcConfig
        estimator_config = TruncatedMcConfig(
            n_sims=config.n_sims,
            adaptive_n=config.adaptive_n,
            enable_pruning=config.enable_pruning,
            max_horizon=config.max_horizon,
        )

    result, _sim = run(
        n_per_inning=config.n_per_inning,
        n_sims=config.n_sims,
        seed=config.seed,
        log_fn=context.log.info,
        profile=True,
        slot=config.slot,
        estimator=config.estimator,
        adaptive_n=config.adaptive_n,
        enable_pruning=config.enable_pruning,
        estimator_config=estimator_config,
    )
    context.log.info(
        f"Complete: {result['correct']}/{result['n_games']} correct "
        f"({result['accuracy']:.0%}), mean MC time {result['mean_mc_time']:.1f}s"
    )
    context.add_output_metadata({
        "accuracy": result["accuracy"],
        "n_games": result["n_games"],
        "n_per_inning": config.n_per_inning,
        "n_sims": config.n_sims,
        "seed": config.seed,
        "estimator": config.estimator,
        "mean_mc_time_s": result["mean_mc_time"],
        "total_wall_time_s": result["total_time"],
    })

