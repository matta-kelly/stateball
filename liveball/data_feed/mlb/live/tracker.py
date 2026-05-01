"""Per-game async tracker — consumes a GameSource until game is Final."""

from __future__ import annotations

import asyncio
import logging
import pickle
import random
import time
from typing import TYPE_CHECKING

from data_feed.mlb.live.metrics import PollEntry, get_or_create, remove as remove_metrics
from data_feed.mlb.live.source import GameSource, SourceError
from data_feed.poll_log import flush_game_log
from data_feed.redis_client import get_client as get_redis
from data_feed.sim.redis_queue import push_sim_job

if TYPE_CHECKING:
    from data_feed.snapshot import GameSnapshot

logger = logging.getLogger(__name__)

MAX_ERROR_INTERVAL = 300.0  # 5 min backoff cap

LIVE_INTERVAL = 2.0   # standard poll interval during live play (seconds)

# Events with known dead zones — back off to _BACKOFF_INTERVAL for a fixed window,
# then resume LIVE_INTERVAL.
# Derived from poll log analysis (see analysis/analyze_poll_intervals.py).
_BACKOFF_EVENTS: dict[str, float] = {
    "Mound Visit":            60.0,  # p05=21s, p50=82s — back off 60s
    "Pitching Substitution":  45.0,  # p05=15s, p50=39s — back off 45s
}
_BACKOFF_INTERVAL = 15.0   # poll every 15s during known dead zones
_INNING_BREAK_BACKOFF = 105.0  # between-inning break backoff window (seconds)


def _poll_interval(abstract_state: str) -> float:
    """Return seconds to sleep based on game state."""
    if abstract_state == "Live":
        return LIVE_INTERVAL
    if abstract_state == "Preview":
        return 60.0
    # Final, error, or unknown
    return 30.0


def _error_interval(consecutive_errors: int) -> float:
    """Exponential backoff: 3 → 6 → 12 → ... → 300s cap, with ±12.5% jitter."""
    base = min(3.0 * (2 ** (consecutive_errors - 1)), MAX_ERROR_INTERVAL)
    jitter = base * 0.25 * (random.random() - 0.5)
    return base + jitter


async def track_game(
    game_pk: int,
    source: GameSource,
    snapshot: GameSnapshot,
    *,
    game_date: str | None = None,
    hydrator=None,
) -> None:
    """Consume a GameSource for a single game until Final.

    Writes live state to the in-memory snapshot (not SQLite).
    When hydrator is provided and Redis is available, pushes sim jobs on every state change.
    """
    logger.info(f"[{game_pk}] Tracker started")
    gm = get_or_create(game_pk)
    prev_state: dict | None = None
    consecutive_errors = 0
    backoff_until: float = 0.0  # monotonic time after which we resume LIVE_INTERVAL polling
    prev_inning: int | None = None
    prev_inning_half: str | None = None

    await source.open()
    try:
        while True:
            try:
                update = await source.next_update()
            except SourceError as e:
                consecutive_errors += 1
                interval = _error_interval(consecutive_errors)
                error_detail = str(e)
                if e.status_code:
                    error_detail = f"{e.status_code} {e}"
                logger.warning(f"[{game_pk}] Fetch error (attempt {consecutive_errors}): {error_detail}")
                gm.record_poll(
                    PollEntry(ts=time.time(), response_ms=e.response_ms, changed=False, error=error_detail,
                              interval_s=interval),
                    interval=interval,
                    state="",
                )
                source.set_interval(interval)
                continue
            except Exception:
                consecutive_errors += 1
                interval = _error_interval(consecutive_errors)
                logger.exception(f"[{game_pk}] Unexpected error (attempt {consecutive_errors})")
                gm.record_poll(
                    PollEntry(ts=time.time(), response_ms=0.0, changed=False, error="unexpected_error",
                              interval_s=interval),
                    interval=interval,
                    state="",
                )
                source.set_interval(interval)
                continue

            consecutive_errors = 0
            state = update.flat_state
            changed = state != prev_state

            # Write to in-memory snapshot
            snapshot.update_live(game_pk, state, update.gumbo or {})

            # Game state transitions — record before we lose prev_state.
            # Only records inning/half/outs/PA boundaries; no-op otherwise.
            if changed:
                from data_feed import game_state_recorder
                game_state_recorder.record_transition(snapshot, game_pk, state, prev_state)

            prev_state = state

            abstract = state.get("abstract_game_state", "")

            # Detect inning transition
            inning = state.get("inning")
            inning_half = state.get("inning_half")
            inning_transition = (
                prev_inning is not None
                and (inning != prev_inning or inning_half != prev_inning_half)
            )
            prev_inning = inning
            prev_inning_half = inning_half

            # Adaptive interval — back off during known dead zones, else state-based default
            if abstract == "Live":
                event_type = state.get("event_type")
                if inning_transition:
                    backoff_until = time.monotonic() + _INNING_BREAK_BACKOFF
                    logger.debug("[%d] Backoff %ds after inning transition", game_pk, _INNING_BREAK_BACKOFF)
                elif changed and event_type in _BACKOFF_EVENTS:
                    backoff_until = time.monotonic() + _BACKOFF_EVENTS[event_type]
                    logger.debug("[%d] Backoff %ds after %s", game_pk, _BACKOFF_EVENTS[event_type], event_type)
                interval = _BACKOFF_INTERVAL if time.monotonic() < backoff_until else _poll_interval(abstract)
            else:
                interval = _poll_interval(abstract)

            # --- Market snapshots on every state change ---
            if abstract == "Live" and changed:
                from data_feed.market_recorder import schedule_event_snapshots
                asyncio.create_task(schedule_event_snapshots(snapshot, game_pk))

            # --- Sim dispatch on every state change ---
            if hydrator is not None and abstract == "Live" and changed and update.gumbo is not None:
                try:
                    ctx = await hydrator.ensure_hydrated(
                        game_pk, game_date, update.gumbo
                    )
                    game_input = ctx.to_game_input(update.gumbo)
                    game_input_bytes = pickle.dumps(game_input)
                    await push_sim_job(get_redis(), game_pk, game_input_bytes)
                    logger.debug("[%d] Sim request queued to Redis", game_pk)
                except Exception:
                    logger.exception("[%d] Failed to queue sim request", game_pk)

            event_type = state.get("event_type")
            mlb_event_ts = state.get("mlb_event_ts")

            if abstract == "Final":
                logger.info(f"[{game_pk}] Game final")
                snapshot.on_game_final(game_pk)
                if hydrator is not None:
                    hydrator.remove(game_pk)
                gm.record_poll(
                    PollEntry(ts=time.time(), response_ms=update.response_ms, changed=changed, error=None,
                              event_type=event_type, mlb_event_ts=mlb_event_ts, interval_s=interval,
                              inning=inning, inning_half=inning_half, inning_transition=inning_transition),
                    interval=interval,
                    state="Final",
                )
                if game_date:
                    loop = asyncio.get_running_loop()
                    # run_in_executor returns an already-scheduled Future —
                    # fire-and-forget, do not wrap in create_task (which expects a coroutine).
                    loop.run_in_executor(None, flush_game_log, game_pk, game_date)
                remove_metrics(game_pk)
                return

            gm.record_poll(
                PollEntry(ts=time.time(), response_ms=update.response_ms, changed=changed, error=None,
                          event_type=event_type, mlb_event_ts=mlb_event_ts, interval_s=interval,
                          inning=inning, inning_half=inning_half, inning_transition=inning_transition),
                interval=interval,
                state=abstract,
            )
            source.set_interval(interval)
    finally:
        await source.close()
