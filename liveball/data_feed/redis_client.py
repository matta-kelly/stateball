"""Shared async Redis connection pool for the data feed."""

from __future__ import annotations

import logging
import os

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_client: aioredis.Redis | None = None


def get_client() -> aioredis.Redis:
    """Return a shared async Redis client. Lazy-init from REDIS_URL env var."""
    global _client
    if _client is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _client = aioredis.from_url(url, decode_responses=True)
        logger.info("Redis client initialized: %s", url.split("@")[-1])  # log host, not creds
    return _client


async def close() -> None:
    """Close the Redis connection pool. Called on shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
