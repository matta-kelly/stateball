"""Schedule monitor runner — polls warehouse on interval."""

import logging
import os
import time

import httpx

from data_feed.db import init_db
from data_feed.mlb.schedule.monitor import fetch_trackable, sync_games

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://liveball")


def run(
    poll_interval: int = 300,
    db_path=None,
    backend_url: str = DEFAULT_BACKEND_URL,
) -> None:
    """Run the schedule monitor loop.

    Args:
        poll_interval: Seconds between polls (default 300 = 5 min).
        db_path: Optional SQLite DB path.
        backend_url: Base URL of the liveball backend.
    """
    init_db(db_path)
    logger.info(
        f"Schedule monitor starting (poll every {poll_interval}s, "
        f"backend={backend_url})"
    )

    kwargs = {"db_path": db_path} if db_path else {}

    consecutive_errors = 0

    with httpx.Client(timeout=30.0) as client:
        while True:
            try:
                games = fetch_trackable(backend_url, client)
                count = sync_games(games, **kwargs)
                logger.info(f"Synced {count} trackable games from warehouse")
                consecutive_errors = 0
            except httpx.HTTPError as e:
                consecutive_errors += 1
                log_fn = logger.error if consecutive_errors >= 5 else logger.warning
                log_fn(f"HTTP error polling warehouse (attempt {consecutive_errors}): {e}")
            except Exception:
                consecutive_errors += 1
                log_fn = logger.error if consecutive_errors >= 5 else logger.warning
                log_fn(f"Unexpected error during poll (attempt {consecutive_errors})", exc_info=True)

            time.sleep(poll_interval)
