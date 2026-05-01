"""Sync Redis client for the liveball backend."""

import logging
import os

import redis

logger = logging.getLogger(__name__)

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """Return a shared sync Redis client. Lazy-init from REDIS_URL env var."""
    global _client
    if _client is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _client = redis.from_url(url, decode_responses=True)
        logger.info("Backend Redis client initialized")
    return _client
