"""Live game manager — scans snapshot, spawns/reaps per-game trackers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from data_feed.mlb.live.tracker import track_game
from data_feed.mlb.live.websocket import WebSocketSource

if TYPE_CHECKING:
    from data_feed.snapshot import GameSnapshot

logger = logging.getLogger(__name__)

MANAGER_SCAN_INTERVAL = 60  # seconds between scans


def _get_trackable_games(snapshot: GameSnapshot) -> list[dict]:
    """Return games from snapshot that need a live tracker.

    Criteria: Live games, or Preview games within 60 min of start.
    """
    now = datetime.now(timezone.utc)
    trackable = []

    for game in snapshot.get_all_games():
        state = game.get("abstract_game_state", "")
        if state == "Live":
            trackable.append(game)
        elif state == "Preview":
            dt_str = game.get("game_datetime")
            if dt_str:
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    if (dt - now).total_seconds() <= 3600:
                        trackable.append(game)
                except (ValueError, TypeError):
                    pass

    return trackable


async def run_manager(
    snapshot: GameSnapshot,
    *,
    hydrator=None,
) -> None:
    """Main manager loop — scan snapshot, spawn/reap tracker tasks."""
    active_tasks: dict[int, asyncio.Task] = {}

    logger.info("Live game manager started")

    # Wait briefly for schedule sync to populate the snapshot on startup
    for _ in range(10):
        if snapshot.game_count() > 0:
            break
        await asyncio.sleep(1)

    while True:
        try:
            games = _get_trackable_games(snapshot)

            # Reap finished tasks first so respawns happen in the same scan
            reaped_pks: set[int] = set()
            for pk in list(active_tasks):
                task = active_tasks[pk]
                if task.done():
                    exc = task.exception()
                    if exc:
                        logger.error(
                            f"[{pk}] Tracker died with error",
                            exc_info=exc,
                        )
                    del active_tasks[pk]
                    reaped_pks.add(pk)

            # Spawn trackers for new/respawned games
            for game in games:
                pk = game["game_pk"]
                if pk not in active_tasks:
                    label = "Respawning" if pk in reaped_pks else "Spawning"
                    logger.info(
                        f"[{pk}] {label} tracker "
                        f"(state={game['abstract_game_state']})"
                    )
                    source = WebSocketSource(pk)
                    task = asyncio.create_task(
                        track_game(
                            pk,
                            source,
                            snapshot,
                            game_date=game.get("game_date"),
                            hydrator=hydrator,
                        )
                    )
                    active_tasks[pk] = task

            if active_tasks:
                logger.debug(
                    f"Active trackers: {len(active_tasks)} "
                    f"({list(active_tasks.keys())})"
                )

        except Exception:
            logger.exception("Manager scan error")

        await asyncio.sleep(MANAGER_SCAN_INTERVAL)
