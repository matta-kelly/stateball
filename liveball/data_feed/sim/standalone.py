"""Standalone sim worker — consumes Redis queue, runs sims, publishes results.

Runs as its own deployment, independent of the data-feed pod. Bootstraps
Simulator from DuckLake artifact registry at startup, then blocks on
Redis queue for GameInput jobs.

Usage: .venv/bin/python -m data_feed.sim.standalone [--workers N] [--estimator smc]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Standalone sim worker")
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of sim worker processes (default: 2)",
    )
    parser.add_argument(
        "--estimator", default="truncated_mc", choices=["naive_mc", "smc", "truncated_mc"],
        help="Simulation estimator (default: truncated_mc)",
    )
    parser.add_argument(
        "--slot", default="prod",
        help="Artifact registry slot (default: prod)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Bootstrap: resolve artifact S3 paths from DuckLake registry
    from sim.infra.bootstrap import bootstrap

    from data_feed.sim.connection import get_ducklake_connection

    logger.info("Bootstrapping artifacts from DuckLake (slot=%s, estimator=%s)", args.slot, args.estimator)
    conn = get_ducklake_connection()
    _, _, artifact_paths = bootstrap(conn, slot=args.slot, seed=42, estimator=args.estimator)
    conn.close()
    logger.info("Bootstrap complete — DuckLake connection closed")

    from_s3_kwargs = {
        "baserunning_path": artifact_paths["baserunning"],
        "model_path": artifact_paths["xgboost_sim"],
        "live_model_path": artifact_paths["xgboost_live"],
        "win_expectancy_path": artifact_paths["win_expectancy"],
        "pitcher_exit_path": artifact_paths["pitcher_exit"],
    }
    for atype in ("n_lookup", "stopping_thresholds", "gamma_schedule", "horizon_weights"):
        if atype in artifact_paths:
            from_s3_kwargs[f"{atype}_path"] = artifact_paths[atype]

    # ProcessPoolExecutor with spawn context (avoids DuckDB fork hazard)
    from data_feed.sim.worker import init_worker

    mp_ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp_ctx,
        initializer=init_worker,
        initargs=(from_s3_kwargs, 42, args.estimator),
    )
    logger.info("ProcessPool ready (%d workers, spawn context)", args.workers)

    # Run dispatcher loop
    try:
        asyncio.run(_run(pool))
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        pool.shutdown(wait=False)


async def _health_handler(reader, writer):
    """Minimal HTTP health endpoint — responds 200 to any request."""
    await reader.read(1024)  # consume request
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
    await writer.drain()
    writer.close()


async def _run(pool: ProcessPoolExecutor) -> None:
    from data_feed.sim.dispatcher import run_dispatcher

    server = await asyncio.start_server(_health_handler, "0.0.0.0", 8002)
    logger.info("Health server listening on :8002")

    try:
        await run_dispatcher(pool)
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    main()
