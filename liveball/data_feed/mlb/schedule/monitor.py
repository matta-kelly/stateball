"""Schedule monitor — polls the warehouse for trackable games."""

import logging
from datetime import datetime, timezone

import httpx

from data_feed.db import get_connection

logger = logging.getLogger(__name__)


def fetch_trackable(backend_url: str, client: httpx.Client) -> list[dict]:
    """Fetch trackable games from the warehouse via the backend API.

    Args:
        backend_url: Base URL of the liveball backend (e.g. http://liveball).
        client: httpx client for connection reuse.

    Returns:
        List of game dicts from /api/v1/games/trackable.
    """
    url = f"{backend_url}/api/v1/games/trackable"
    resp = client.get(url)
    resp.raise_for_status()
    return resp.json()


def sync_games(games: list[dict], db_path=None) -> int:
    """Upsert schedule data into SQLite.

    Simple overwrite — no conditional preservation needed since
    live state lives in the in-memory snapshot, not SQLite.

    Args:
        games: List of game dicts from the trackable endpoint.
        db_path: Optional path to SQLite DB.

    Returns:
        Number of games synced.
    """
    kwargs = {"db_path": db_path} if db_path else {}
    conn = get_connection(**kwargs)
    now = datetime.now(timezone.utc).isoformat()

    for game in games:
        conn.execute(
            """INSERT INTO games
               (game_pk, game_date, game_datetime, game_type, status,
                abstract_game_state, away_team_id, away_team_name,
                home_team_id, home_team_name, venue_name, last_synced_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(game_pk) DO UPDATE SET
                game_date = excluded.game_date,
                game_datetime = excluded.game_datetime,
                game_type = excluded.game_type,
                status = excluded.status,
                abstract_game_state = excluded.abstract_game_state,
                away_team_id = excluded.away_team_id,
                away_team_name = excluded.away_team_name,
                home_team_id = excluded.home_team_id,
                home_team_name = excluded.home_team_name,
                venue_name = excluded.venue_name,
                last_synced_at = excluded.last_synced_at""",
            [
                game["game_pk"],
                game.get("game_date"),
                game.get("game_datetime"),
                game.get("game_type"),
                game.get("status"),
                game.get("abstract_game_state"),
                game.get("away_team_id"),
                game.get("away_team_name"),
                game.get("home_team_id"),
                game.get("home_team_name"),
                game.get("venue_name"),
                now,
            ],
        )

    conn.commit()
    conn.close()
    return len(games)
