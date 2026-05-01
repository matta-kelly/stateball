"""Sim dispatcher — consumes jobs from Redis, submits to ProcessPoolExecutor, publishes results."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor

from data_feed.redis_client import get_client as get_redis
from data_feed.sim.redis_queue import pop_sim_job, pop_all_sim_jobs, publish_sim_result
from data_feed.sim.worker import run_sim

logger = logging.getLogger(__name__)


async def _submit_one(
    game_pk: int,
    game_input_bytes: bytes,
    loop: asyncio.AbstractEventLoop,
    pool: ProcessPoolExecutor,
) -> None:
    """Submit one sim job to the pool and publish result via Redis."""
    t0 = time.monotonic()
    try:
        result = await loop.run_in_executor(pool, run_sim, game_input_bytes)
        duration_ms = (time.monotonic() - t0) * 1000
        result["duration_ms"] = duration_ms

        await publish_sim_result(get_redis(), game_pk, result)

        logger.info(
            "[%d] Sim complete: p_home=%.3f\u00b1%.3f (%dms)",
            game_pk,
            result["p_home_win"],
            result["p_home_win_se"],
            int(duration_ms),
        )
    except Exception:
        logger.exception("[%d] Sim failed", game_pk)


async def run_dispatcher(
    pool: ProcessPoolExecutor,
) -> None:
    """Consume sim jobs from Redis, submit to pool, publish results."""
    loop = asyncio.get_running_loop()
    redis = get_redis()

    logger.info("Sim dispatcher started (Redis queue)")

    while True:
        # Block until at least one job arrives
        job = await pop_sim_job(redis, timeout=2.0)
        if job is None:
            continue

        game_pk, game_input_bytes = job

        # Drain + dedup — latest per game_pk wins
        pending: dict[int, bytes] = {game_pk: game_input_bytes}
        extra = await pop_all_sim_jobs(redis)
        pending.update(extra)

        # Submit all pending games as fire-and-forget tasks
        # (fast games aren't blocked by slow SMC runs)
        for pk, input_bytes in pending.items():
            asyncio.create_task(_submit_one(pk, input_bytes, loop, pool))
