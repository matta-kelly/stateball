"""Game input assembly for the simulator (training/eval path).

Queries the warehouse to assemble everything the engine needs to simulate
a game forward from any point: batting orders, pitcher profiles, bullpen
pools, and an initial GameState.

Input: a feature vector (one PA row from feat_mlb__vectors) + DuckDB conn.
Output: a GameInput dataclass ready for the engine.

Player profile and handedness fetching lives in sim.data.profiles (shared
with the live path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sim.engine.core.state import GameState
from sim.game_inputs.profiles import (
    build_player,
    detect_schema,
    fetch_batter_profiles,
    fetch_handedness,
    fetch_pitcher_profiles,
)

logger = logging.getLogger(__name__)


@dataclass
class GameInput:
    """Everything the engine needs to simulate a game."""

    home_lineup: list[dict]
    away_lineup: list[dict]
    home_pitcher: dict
    away_pitcher: dict
    home_bullpen: list[dict]
    away_bullpen: list[dict]
    game_state: GameState
    game_pk: int
    game_date: str
    seed_context: dict


def init_game(vector: dict, conn) -> GameInput:
    """Assemble game inputs from a feature vector and warehouse connection.

    Args:
        vector: A single row from feat_mlb__vectors as a dict. Must include
                metadata columns (game_pk, game_date, at_bat_number) and
                game state columns (inning, is_bottom, outs, etc.).
        conn: DuckDB connection (local or DuckLake).

    Returns:
        GameInput with both sides' lineups, pitchers, bullpens, and initial
        game state — ready to feed to the engine.
    """
    schema = detect_schema(conn)
    game_pk = int(vector["game_pk"])
    game_date = str(vector["game_date"])
    at_bat_number = int(vector["at_bat_number"])

    game_state = _build_game_state(vector)

    # Roster data for this game
    roster = _fetch_roster(conn, schema, game_pk)
    starting_lineup = {
        "home": _starting_lineup(roster, "home"),
        "away": _starting_lineup(roster, "away"),
    }
    starting_pitchers = _starting_pitchers(roster)

    # Determine current lineups and pitchers
    if at_bat_number <= 1:
        home_lineup_ids = starting_lineup["home"]
        away_lineup_ids = starting_lineup["away"]
        home_pitcher_id = starting_pitchers["home"]
        away_pitcher_id = starting_pitchers["away"]
        used_pitcher_ids = set()
    else:
        home_lineup_ids, away_lineup_ids, current_pitchers, used_pitcher_ids = (
            _reconstruct_mid_game(
                conn, schema, game_pk, at_bat_number, starting_lineup
            )
        )
        home_pitcher_id = current_pitchers["home"]
        away_pitcher_id = current_pitchers["away"]

    # Bullpen: roster pitchers minus lineup minus already used
    lineup_ids = set(home_lineup_ids + away_lineup_ids)
    home_bullpen_ids = _available_bullpen(
        roster, "home", lineup_ids, used_pitcher_ids, home_pitcher_id
    )
    away_bullpen_ids = _available_bullpen(
        roster, "away", lineup_ids, used_pitcher_ids, away_pitcher_id
    )

    # Collect all player IDs and fetch profiles
    all_batter_ids = list(set(home_lineup_ids + away_lineup_ids))
    all_pitcher_ids = list(
        set(
            [home_pitcher_id, away_pitcher_id]
            + home_bullpen_ids
            + away_bullpen_ids
        )
    )
    all_player_ids = list(set(all_batter_ids + all_pitcher_ids))

    handedness = fetch_handedness(conn, schema, all_player_ids)
    batter_profiles = fetch_batter_profiles(
        conn, schema, all_batter_ids, game_date
    )
    pitcher_profiles = fetch_pitcher_profiles(
        conn, schema, all_pitcher_ids, game_date
    )

    # Extract real dynamic features from the vector for the seed PA.
    # Dynamic game_state fields the live model needs for the seed PA.
    # Using real values instead of fabricating zeros.
    _SEED_CONTEXT_KEYS = [
        "times_through_order", "batter_prior_pa",
        "pitcher_pitch_count", "pitcher_bf_game", "batter_ab_vs_pitcher",
        "pitcher_outing_walks", "pitcher_outing_hits",
        "pitcher_outing_k", "pitcher_outing_runs",
        "pitcher_outing_whip", "pitcher_recent_whip",
        "balls", "strikes",
    ]
    seed_context = {
        k: float(vector.get(k, 0)) for k in _SEED_CONTEXT_KEYS
    }

    return GameInput(
        home_lineup=[
            build_player(pid, batter_profiles, handedness, "bats")
            for pid in home_lineup_ids
        ],
        away_lineup=[
            build_player(pid, batter_profiles, handedness, "bats")
            for pid in away_lineup_ids
        ],
        home_pitcher=build_player(
            home_pitcher_id, pitcher_profiles, handedness, "throws"
        ),
        away_pitcher=build_player(
            away_pitcher_id, pitcher_profiles, handedness, "throws"
        ),
        home_bullpen=[
            build_player(pid, pitcher_profiles, handedness, "throws")
            for pid in home_bullpen_ids
        ],
        away_bullpen=[
            build_player(pid, pitcher_profiles, handedness, "throws")
            for pid in away_bullpen_ids
        ],
        game_state=game_state,
        game_pk=game_pk,
        game_date=game_date,
        seed_context=seed_context,
    )


# ---------------------------------------------------------------------------
# Internal helpers (training/eval path only)
# ---------------------------------------------------------------------------


def _build_game_state(vector: dict) -> GameState:
    """Extract initial GameState from a feature vector's dynamic fields."""
    is_bottom = int(vector.get("is_bottom", 0))
    outs = int(vector.get("outs", 0))
    bases = (
        (int(vector.get("runner_1b", 0)) << 0)
        | (int(vector.get("runner_2b", 0)) << 1)
        | (int(vector.get("runner_3b", 0)) << 2)
    )
    run_diff = int(vector.get("run_diff", 0))
    inning = int(vector.get("inning", 1))

    # run_diff is bat_score - fld_score. Convert to absolute scores.
    # We don't know absolute scores from the vector — use run_diff
    # relative to 0. The sim only needs the differential anyway.
    if is_bottom:
        home_score = max(run_diff, 0)
        away_score = max(-run_diff, 0)
    else:
        away_score = max(run_diff, 0)
        home_score = max(-run_diff, 0)

    return GameState(
        inning=inning,
        half=is_bottom,
        outs=outs,
        bases=bases,
        home_score=home_score,
        away_score=away_score,
    )


def _fetch_roster(conn, schema: str, game_pk: int) -> list[dict]:
    """Fetch full roster for a game from proc_mlb__rosters."""
    rows = conn.execute(
        f"""
        SELECT player_id, team_id, side, position, batting_order,
               is_starting_pitcher
        FROM {schema}.proc_mlb__rosters
        WHERE game_pk = ?
        """,
        [game_pk],
    ).fetchall()
    cols = ["player_id", "team_id", "side", "position", "batting_order",
            "is_starting_pitcher"]
    return [dict(zip(cols, row)) for row in rows]


def _starting_lineup(roster: list[dict], side: str) -> list[int]:
    """Extract the starting 9 in batting order for one side."""
    batters = [
        r for r in roster
        if r["side"] == side and r["batting_order"] is not None
    ]
    batters.sort(key=lambda r: r["batting_order"])
    return [r["player_id"] for r in batters]


def _starting_pitchers(roster: list[dict]) -> dict[str, int]:
    """Extract starting pitcher IDs for both sides."""
    result = {}
    for r in roster:
        if r["is_starting_pitcher"]:
            result[r["side"]] = r["player_id"]
    return result


def _reconstruct_mid_game(
    conn,
    schema: str,
    game_pk: int,
    at_bat_number: int,
    starting_lineup: dict[str, list[int]],
) -> tuple[list[int], list[int], dict[str, int], set[int]]:
    """Reconstruct current lineups and pitchers from pitch data.

    Returns:
        (home_lineup_ids, away_lineup_ids, current_pitchers, used_pitcher_ids)
    """
    # Get all PAs up to this point — one row per PA with batter and pitcher
    rows = conn.execute(
        f"""
        SELECT DISTINCT ON (at_bat_number)
            at_bat_number, batter_id, pitcher_id, inning_topbot
        FROM {schema}.proc_mlb__events
        WHERE game_pk = ? AND at_bat_number < ?
        ORDER BY at_bat_number, pitch_number
        """,
        [game_pk, at_bat_number],
    ).fetchall()

    # Separate by side
    # Top = away batting, Bot = home batting
    home_batters_seen = []
    away_batters_seen = []
    home_pitchers_seen = []  # pitchers facing home batters (away team pitching)
    away_pitchers_seen = []  # pitchers facing away batters (home team pitching)

    for row in rows:
        ab_num, batter_id, pitcher_id, topbot = row
        if topbot == "Bot":
            if batter_id not in home_batters_seen:
                home_batters_seen.append(batter_id)
            if pitcher_id not in away_pitchers_seen:
                away_pitchers_seen.append(pitcher_id)
        else:
            if batter_id not in away_batters_seen:
                away_batters_seen.append(batter_id)
            if pitcher_id not in home_pitchers_seen:
                home_pitchers_seen.append(pitcher_id)

    # Reconstruct lineups: last 9 distinct batters, fill from lineup card
    home_lineup = _merge_lineup(home_batters_seen, starting_lineup["home"])
    away_lineup = _merge_lineup(away_batters_seen, starting_lineup["away"])

    # Current pitchers: last seen on each side
    current_pitchers = {
        "home": home_pitchers_seen[-1] if home_pitchers_seen else starting_lineup["home"][0],
        "away": away_pitchers_seen[-1] if away_pitchers_seen else starting_lineup["away"][0],
    }

    # All pitchers who've appeared (for bullpen exclusion)
    used_pitcher_ids = set(home_pitchers_seen + away_pitchers_seen)

    return home_lineup, away_lineup, current_pitchers, used_pitcher_ids


def _merge_lineup(
    batters_seen: list[int], starting_lineup: list[int]
) -> list[int]:
    """Build current 9-man lineup from observed batters + lineup card.

    If fewer than 9 distinct batters have appeared, fill remaining slots
    from the starting lineup card (players who haven't batted yet).
    The order is: observed batters in batting order, then unfilled starters.
    """
    if len(batters_seen) >= 9:
        # Take the last 9 distinct batters in the order they appeared
        return batters_seen[-9:]

    # Fill from lineup card — starters who haven't batted yet
    seen_set = set(batters_seen)
    remaining = [pid for pid in starting_lineup if pid not in seen_set]

    # Rebuild in lineup card order, substituting observed batters
    # into their predecessors' slots
    lineup = []
    remaining_idx = 0
    for pid in starting_lineup:
        if pid in seen_set:
            lineup.append(pid)
        elif remaining_idx < len(remaining):
            lineup.append(remaining[remaining_idx])
            remaining_idx += 1

    return lineup[:9]


def _available_bullpen(
    roster: list[dict],
    side: str,
    lineup_ids: set[int],
    used_pitcher_ids: set[int],
    current_pitcher_id: int,
) -> list[int]:
    """Determine available bullpen arms for one side."""
    return [
        r["player_id"]
        for r in roster
        if r["side"] == side
        and r["position"] == "P"
        and r["player_id"] not in lineup_ids
        and r["player_id"] not in used_pitcher_ids
        and r["player_id"] != current_pitcher_id
    ]
