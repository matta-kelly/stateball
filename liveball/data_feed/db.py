"""SQLite database for data feed — schedule persistence only.

Live game state and sim results live in the in-memory GameSnapshot.
SQLite stores schedule fields for restart recovery.
"""

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("data_feed.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_pk              INTEGER PRIMARY KEY,
    game_date            TEXT,
    game_datetime        TEXT,
    game_type            TEXT,
    status               TEXT,
    abstract_game_state  TEXT,
    away_team_id         INTEGER,
    away_team_name       TEXT,
    home_team_id         INTEGER,
    home_team_name       TEXT,
    venue_name           TEXT,
    last_synced_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_abstract_state ON games(abstract_game_state);
"""


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection with WAL mode and return it."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create tables if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def read_schedule(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """Read all schedule rows for snapshot seeding."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM games ORDER BY game_datetime ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
