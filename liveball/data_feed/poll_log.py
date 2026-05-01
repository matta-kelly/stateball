"""Flush per-game poll ring buffer to S3 on game final.

Called from tracker.py when abstract_game_state transitions to Final.
Runs in a thread pool executor so it doesn't block the event loop.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def flush_game_log(game_pk: int, game_date: str) -> None:
    """Write the ring buffer for game_pk to S3 as JSONL.

    Path: s3://{S3_BUCKET}/stateball/liveball/poll_logs/{game_date}/{game_pk}.jsonl
    Each line is one PollEntry. Errors are logged but never raised — a failed
    flush doesn't affect the tracker or the feed.
    """
    from data_feed.mlb.live.metrics import _store

    import s3fs

    gm = _store.get(game_pk)
    if not gm or not gm.history:
        logger.debug("[%d] No poll history to flush", game_pk)
        return

    bucket = os.environ.get("S3_BUCKET", "dazoo")
    path = f"s3://{bucket}/stateball/liveball/poll_logs/{game_date}/{game_pk}.jsonl"

    try:
        fs = s3fs.S3FileSystem(
            key=os.environ.get("S3_ACCESS_KEY_ID", ""),
            secret=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
            endpoint_url=os.environ.get("S3_ENDPOINT", ""),
        )
        with fs.open(path, "w") as f:
            for entry in gm.history:
                f.write(
                    json.dumps({
                        "ts": entry.ts,
                        "response_ms": round(entry.response_ms, 1),
                        "changed": entry.changed,
                        "error": entry.error,
                        "event_type": entry.event_type,
                        "mlb_event_ts": entry.mlb_event_ts,
                        "interval_s": entry.interval_s,
                        "inning": entry.inning,
                        "inning_half": entry.inning_half,
                        "inning_transition": entry.inning_transition,
                    })
                    + "\n"
                )
        logger.info("[%d] Poll log flushed → %s (%d entries)", game_pk, path, len(gm.history))
    except Exception:
        logger.exception("[%d] Failed to flush poll log to S3", game_pk)
