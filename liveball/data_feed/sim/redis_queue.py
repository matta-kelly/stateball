"""Redis-backed sim job queue and result channels.

Job queue: Redis List (RPUSH/BLPOP) — serialized GameInput as base64 pickle.
Results: Redis Pub/Sub — JSON with scalar sim results.
Snapshot changes: Redis Pub/Sub — lightweight notification for SSE wake-up.
"""

from __future__ import annotations

import base64
import json
import logging

logger = logging.getLogger(__name__)

SIM_JOBS_KEY = "sim:jobs"
SIM_RESULTS_CHANNEL = "sim:results"
SNAPSHOT_CHANGES_CHANNEL = "snapshot:changes"


async def push_sim_job(redis, game_pk: int, game_input_bytes: bytes) -> None:
    """Enqueue a sim job. GameInput is pickled and base64-encoded for Redis string safety."""
    payload = json.dumps({
        "game_pk": game_pk,
        "input": base64.b64encode(game_input_bytes).decode(),
    })
    await redis.rpush(SIM_JOBS_KEY, payload)


async def pop_sim_job(redis, timeout: float = 1.0) -> tuple[int, bytes] | None:
    """Blocking pop from sim job queue. Returns (game_pk, pickled_input) or None on timeout."""
    result = await redis.blpop(SIM_JOBS_KEY, timeout=timeout)
    if result is None:
        return None
    payload = json.loads(result[1])
    return payload["game_pk"], base64.b64decode(payload["input"])


async def pop_all_sim_jobs(redis) -> dict[int, bytes]:
    """Non-blocking drain of all pending jobs, deduped by game_pk (latest wins)."""
    pending: dict[int, bytes] = {}
    while True:
        result = await redis.lpop(SIM_JOBS_KEY)
        if result is None:
            break
        payload = json.loads(result)
        pending[payload["game_pk"]] = base64.b64decode(payload["input"])
    return pending


async def publish_sim_result(redis, game_pk: int, result: dict) -> None:
    """Publish sim result to Redis channel."""
    await redis.publish(SIM_RESULTS_CHANNEL, json.dumps({"game_pk": game_pk, **result}))
