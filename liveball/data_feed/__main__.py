"""CLI entrypoint for data_feed — starts HTTP server + schedule poll + live manager."""

import argparse
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from data_feed.db import init_db, read_schedule
from data_feed.snapshot import GameSnapshot

logger = logging.getLogger(__name__)

# Globals so lifespan can access CLI args / state
_db_path: Path = Path("data_feed.db")
_snapshot: GameSnapshot | None = None
_sim_enabled: bool = False
_hydrator = None


async def _schedule_sync_loop(snapshot: GameSnapshot, db_path: Path) -> None:
    """Periodically merge schedule data from SQLite into snapshot."""
    while True:
        await asyncio.sleep(60)
        try:
            loop = asyncio.get_running_loop()
            games = await loop.run_in_executor(None, read_schedule, db_path)
            snapshot.merge_schedule(games)
        except Exception:
            logger.exception("Schedule sync failed")


@asynccontextmanager
async def _lifespan(app):
    """Start the live game manager, schedule sync, and sim dispatcher."""
    from data_feed.mlb.live.manager import run_manager

    # Hydrate snapshot from Redis (recovers state from previous pod lifecycle)
    await _snapshot.hydrate_from_redis()

    tasks = []

    # Schedule sync (reads SQLite → snapshot every 60s)
    tasks.append(asyncio.create_task(_schedule_sync_loop(_snapshot, _db_path)))

    # Sim result listener (sim workers run in separate deployment)
    if _sim_enabled:
        tasks.append(asyncio.create_task(_snapshot.run_result_listener()))

    # Live game manager
    tasks.append(
        asyncio.create_task(
            run_manager(_snapshot, hydrator=_hydrator)
        )
    )

    yield

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    from data_feed.redis_client import close as close_redis
    await close_redis()


def main():
    global _db_path, _snapshot, _sim_enabled, _hydrator

    parser = argparse.ArgumentParser(
        description="liveball data feed — schedule mirror + live game tracking"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data_feed.db"),
        help="Path to SQLite database (default: data_feed.db)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="HTTP server port (default: 8001)",
    )
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Liveball backend URL (default: BACKEND_URL env or http://liveball)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Seconds between warehouse polls (default: 300)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    # Sim integration
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Enable sim job dispatch + result listener (workers run separately)",
    )
    args = parser.parse_args()
    _db_path = args.db

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Init Redis client
    from data_feed.redis_client import get_client as get_redis_client

    redis_client = None
    try:
        redis_client = get_redis_client()
    except Exception:
        logger.warning("Redis unavailable — running without persistence", exc_info=True)

    _snapshot = GameSnapshot(redis_client=redis_client)

    # Init DB + seed snapshot from SQLite
    init_db(args.db)
    schedule = read_schedule(args.db)
    _snapshot.merge_schedule(schedule)
    logger.info("Snapshot seeded with %d games from SQLite", len(schedule))

    # Wire snapshot to server
    from data_feed.server import set_snapshot

    set_snapshot(_snapshot)

    # Sim dispatch: hydrator pushes jobs to Redis, workers run separately
    if args.sim:
        from data_feed.sim.connection import get_ducklake_connection
        from data_feed.sim.hydrator import GameHydrator

        conn = get_ducklake_connection()
        _sim_enabled = True
        _hydrator = GameHydrator(conn)
        logger.info("Sim dispatch enabled (hydrator + result listener)")

    # Start schedule poll loop in background thread
    from data_feed.mlb.schedule.runner import run as run_poller

    poll_kwargs = {
        "poll_interval": args.poll_interval,
        "db_path": args.db,
    }
    if args.backend_url:
        poll_kwargs["backend_url"] = args.backend_url

    poll_thread = threading.Thread(target=run_poller, kwargs=poll_kwargs, daemon=True)
    poll_thread.start()

    # Apply lifespan to server app (starts manager + sync + dispatcher)
    from data_feed.server import app

    app.router.lifespan_context = _lifespan

    # Run HTTP server in main thread (uvicorn event loop hosts everything)
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
