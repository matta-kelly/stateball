"""Shared warehouse data access for player profiles and handedness.

Used by both the training/eval path (sim/init.py → init_game) and the
live path (sim/context.py → SimContext). These functions are the only
sim code that touches the warehouse for player-level data.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema / table helpers
# ---------------------------------------------------------------------------


def detect_schema(conn) -> str:
    """Return table prefix: 'lakehouse.main' for DuckLake, 'main' for local."""
    try:
        conn.execute("SELECT 1 FROM lakehouse.main.proc_mlb__events LIMIT 0")
        return "lakehouse.main"
    except Exception:
        return "main"


def profile_table(conn, remote: str, local: str) -> str:
    """Use local temp table if it exists, otherwise fall back to remote."""
    try:
        conn.execute(f"SELECT 1 FROM {local} LIMIT 0")
        return local
    except Exception:
        return remote


# ---------------------------------------------------------------------------
# Player data fetchers
# ---------------------------------------------------------------------------


def fetch_handedness(
    conn, schema: str, player_ids: list[int]
) -> dict[int, dict]:
    """Fetch bats/throws for all players from ref_mlb__players."""
    if not player_ids:
        return {}
    placeholders = ",".join(str(pid) for pid in player_ids)
    rows = conn.execute(
        f"""
        SELECT player_id, bats, throws
        FROM {schema}.ref_mlb__players
        WHERE player_id IN ({placeholders})
        """
    ).fetchall()
    return {
        row[0]: {"bats": row[1], "throws": row[2]}
        for row in rows
    }


def fetch_batter_profiles(
    conn, schema: str, batter_ids: list[int], game_date: str
) -> dict[int, dict]:
    """Fetch wide batter profiles from int_mlb__batter_profile (ASOF by date).

    One query against the wide profile table that joins batters + arsenal +
    discipline. Both platoon splits included.
    """
    if not batter_ids:
        return {}
    placeholders = ",".join(str(pid) for pid in batter_ids)

    table = profile_table(conn, f"{schema}.int_mlb__batter_profile",
                          "_local_batter_profiles")
    rows = conn.execute(
        f"""
        SELECT *
        FROM {table}
        WHERE batter_id IN ({placeholders})
          AND game_date < ?
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY batter_id ORDER BY game_date DESC
        ) = 1
        """,
        [game_date],
    ).fetchdf()

    profiles: dict[int, dict] = {}
    for _, row in rows.iterrows():
        pid = int(row["batter_id"])
        profiles[pid] = row.to_dict()

    return profiles


def fetch_pitcher_profiles(
    conn, schema: str, pitcher_ids: list[int], game_date: str
) -> dict[int, dict]:
    """Fetch wide pitcher profiles from int_mlb__pitcher_profile (ASOF by date).

    One query against the wide profile table that joins pitchers + arsenal.
    Both platoon splits included.
    """
    if not pitcher_ids:
        return {}
    placeholders = ",".join(str(pid) for pid in pitcher_ids)

    table = profile_table(conn, f"{schema}.int_mlb__pitcher_profile",
                          "_local_pitcher_profiles")
    rows = conn.execute(
        f"""
        SELECT *
        FROM {table}
        WHERE pitcher_id IN ({placeholders})
          AND game_date < ?
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY pitcher_id ORDER BY game_date DESC
        ) = 1
        """,
        [game_date],
    ).fetchdf()

    profiles: dict[int, dict] = {}
    for _, row in rows.iterrows():
        pid = int(row["pitcher_id"])
        profiles[pid] = row.to_dict()

    return profiles


# ---------------------------------------------------------------------------
# Player dict assembly
# ---------------------------------------------------------------------------


def build_player(pid: int, profiles: dict, handedness: dict, hand_key: str) -> dict:
    """Merge a player's profile with their ID and handedness.

    Args:
        pid: Player ID.
        profiles: {player_id: profile_dict} from fetch_batter/pitcher_profiles.
        handedness: {player_id: {"bats": ..., "throws": ...}} from fetch_handedness.
        hand_key: "bats" for batters, "throws" for pitchers.

    Returns:
        Profile dict augmented with player_id and hand keys.
    """
    profile = profiles.get(pid, {})
    profile["player_id"] = pid
    profile["hand"] = handedness.get(pid, {}).get(hand_key)
    return profile
