"""CLI entrypoint for the eval runner.

Usage:
    python -m orchestration.assets.eval_runner.cli --n-per-inning 100 --n-sims 1000
"""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="Run stratified eval simulations")
    parser.add_argument("--n-per-inning", type=int, default=20,
                        help="Games per inning (1-9). Total = 9 * n_per_inning")
    parser.add_argument("--n-sims", type=int, default=1000, help="Monte Carlo iterations per game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot", default="prod", help="Artifact slot (prod or test)")
    parser.add_argument("--profile", action="store_true", help="Enable engine profiling")
    parser.add_argument("--estimator", default="naive_mc",
                        choices=["naive_mc", "smc", "truncated_mc"], help="Estimator strategy")
    parser.add_argument("--adaptive-n", action="store_true", help="Enable adaptive N")
    parser.add_argument("--enable-pruning", action="store_true", help="Enable blowout pruning")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    from orchestration.secrets import load_secrets
    load_secrets()

    from .run import run
    run(
        n_per_inning=args.n_per_inning,
        n_sims=args.n_sims,
        seed=args.seed,
        slot=args.slot,
        profile=args.profile,
        estimator=args.estimator,
        adaptive_n=args.adaptive_n,
        enable_pruning=args.enable_pruning,
    )


if __name__ == "__main__":
    main()
