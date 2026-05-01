"""HTTP + SSE server for data feed — reads from in-memory GameSnapshot."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterable

from fastapi import FastAPI, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

if TYPE_CHECKING:
    from data_feed.snapshot import GameSnapshot

logger = logging.getLogger(__name__)

app = FastAPI(title="liveball-data-feed")

# Module-level ref — set by __main__.py at startup
_snapshot: GameSnapshot | None = None


def set_snapshot(snapshot: GameSnapshot) -> None:
    global _snapshot
    _snapshot = snapshot


# --- SSE stream ---

# Fields used to detect changes between polls
_FINGERPRINT_KEYS = (
    "live_updated_at",
    "last_synced_at",
    "abstract_game_state",
    "away_score",
    "home_score",
    "inning",
    "outs",
    "balls",
    "strikes",
    "current_batter_id",
    "status",
    "sim_updated_at",
)


def _game_fingerprint(game: dict) -> tuple:
    return tuple(game.get(k) for k in _FINGERPRINT_KEYS)


async def _stream_games() -> AsyncIterable[ServerSentEvent]:
    """Async generator that yields SSE events for game changes."""
    event_id = 0
    fingerprints: dict[int, tuple] = {}

    # Initial snapshot — send all games
    games = _snapshot.get_all_games()
    for game in games:
        event_id += 1
        fingerprints[game["game_pk"]] = _game_fingerprint(game)
        yield ServerSentEvent(
            data=game,
            event="game_update",
            id=str(event_id),
        )

    prev_pks = {g["game_pk"] for g in games}

    # Wait for snapshot change notification, falling back to 0.5s poll
    while True:
        try:
            await asyncio.wait_for(_snapshot.wait_for_change(), timeout=0.5)
        except asyncio.TimeoutError:
            pass

        try:
            games = _snapshot.get_all_games()
        except Exception:
            logger.exception("SSE: error reading snapshot")
            continue

        current_pks = {g["game_pk"] for g in games}

        # Check for added/removed games (schedule sync)
        if current_pks != prev_pks:
            event_id += 1
            yield ServerSentEvent(
                data=games,
                event="schedule_sync",
                id=str(event_id),
            )
            fingerprints = {g["game_pk"]: _game_fingerprint(g) for g in games}
            prev_pks = current_pks
            continue

        # Check for changed games
        for game in games:
            pk = game["game_pk"]
            fp = _game_fingerprint(game)
            if fingerprints.get(pk) != fp:
                event_id += 1
                fingerprints[pk] = fp
                yield ServerSentEvent(
                    data=game,
                    event="game_update",
                    id=str(event_id),
                )


# --- Endpoints ---


@app.get("/stream", response_class=EventSourceResponse)
async def stream_games():
    """SSE stream of real-time game updates."""
    async for event in _stream_games():
        yield event


@app.get("/games")
def list_games():
    """Return all games from snapshot."""
    return _snapshot.get_all_games()


@app.get("/games/{game_pk}")
def get_game(game_pk: int):
    """Return a single game by game_pk."""
    game = _snapshot.get_game(game_pk)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


@app.get("/games/{game_pk}/detail")
def game_detail(game_pk: int):
    """Return lineups, pitching, bullpen, and play-by-play from cached gumbo."""
    gumbo = _snapshot.get_gumbo(game_pk)
    if not gumbo:
        raise HTTPException(status_code=404, detail="No live data cached")
    return _extract_detail(gumbo)


@app.get("/games/{game_pk}/game-state-events")
def game_state_events(game_pk: int):
    """Return game state transition events (inning/half/out boundaries) for markers."""
    return _snapshot.get_game_state_events(game_pk)


@app.get("/games/{game_pk}/sim-results")
def sim_results(game_pk: int):
    """Return sim result time-series for win probability chart."""
    return _snapshot.get_sim_results(game_pk)


@app.get("/metrics")
def get_metrics():
    """Return per-game tracker polling metrics."""
    from data_feed.mlb.live.metrics import snapshot

    return snapshot()


@app.get("/health")
def health():
    """Health check with snapshot metadata."""
    return {
        "status": "ok",
        "games_count": _snapshot.game_count(),
        "live_tracking_count": _snapshot.live_count(),
    }


# --- Gumbo extraction for detail endpoint ---


def _extract_detail(gumbo: dict) -> dict:
    """Extract display-friendly game detail from full gumbo response."""
    boxscore = gumbo.get("liveData", {}).get("boxscore", {})
    all_plays = gumbo.get("liveData", {}).get("plays", {}).get("allPlays", [])
    game_data = gumbo.get("gameData", {})

    linescore = gumbo.get("liveData", {}).get("linescore", {})

    return {
        "home": _extract_team(boxscore, "home", game_data),
        "away": _extract_team(boxscore, "away", game_data),
        "plays": _extract_plays(all_plays),
        "linescore": _extract_linescore(linescore),
    }


def _extract_team(boxscore: dict, side: str, game_data: dict) -> dict:
    """Extract lineup, pitcher, and bullpen for one team."""
    team = boxscore.get("teams", {}).get(side, {})
    players = team.get("players", {})
    batting_order = team.get("battingOrder", [])
    pitchers_used = team.get("pitchers", [])
    bullpen_ids = team.get("bullpen", [])
    bench_ids = team.get("bench", [])

    team_info = game_data.get("teams", {}).get(side, {})
    team_name = team_info.get("teamName", "")
    abbreviation = team_info.get("abbreviation", "")

    # Lineup
    lineup = []
    for i, pid in enumerate(batting_order, 1):
        p = players.get(f"ID{pid}", {})
        person = p.get("person", {})
        batting = p.get("stats", {}).get("batting", {})
        lineup.append({
            "id": pid,
            "name": person.get("fullName", f"#{pid}"),
            "position": p.get("position", {}).get("abbreviation", ""),
            "jersey": p.get("jerseyNumber", ""),
            "batting_order": i,
            "stats": {
                "ab": batting.get("atBats", 0) or 0,
                "h": batting.get("hits", 0) or 0,
                "r": batting.get("runs", 0) or 0,
                "rbi": batting.get("rbi", 0) or 0,
                "bb": batting.get("baseOnBalls", 0) or 0,
                "so": batting.get("strikeOuts", 0) or 0,
                "hr": batting.get("homeRuns", 0) or 0,
                "avg": batting.get("avg", ".000"),
            },
        })

    def _pitcher_entry(pid: int) -> dict:
        p = players.get(f"ID{pid}", {})
        person = p.get("person", {})
        pitching = p.get("stats", {}).get("pitching", {})
        return {
            "id": pid,
            "name": person.get("fullName", f"#{pid}"),
            "jersey": p.get("jerseyNumber", ""),
            "stats": {
                "ip": pitching.get("inningsPitched", "0.0"),
                "h": pitching.get("hits", 0) or 0,
                "r": pitching.get("runs", 0) or 0,
                "er": pitching.get("earnedRuns", 0) or 0,
                "bb": pitching.get("baseOnBalls", 0) or 0,
                "so": pitching.get("strikeOuts", 0) or 0,
                "pitches": pitching.get("numberOfPitches", 0) or 0,
                "era": pitching.get("era", "0.00"),
            },
        }

    # Current pitcher (last in pitchers_used list)
    empty_pitcher = {
        "id": 0, "name": "", "jersey": "",
        "stats": {"ip": "0.0", "h": 0, "r": 0, "er": 0, "bb": 0, "so": 0, "pitches": 0, "era": "0.00"},
    }
    pitcher_info = _pitcher_entry(pitchers_used[-1]) if pitchers_used else empty_pitcher

    # Pitchers used — all except current, in appearance order, with stats
    used_set = set(pitchers_used)
    prev_pitchers = [_pitcher_entry(pid) for pid in pitchers_used[:-1]] if pitchers_used else []

    # Bullpen — available only (exclude anyone who has appeared)
    bullpen = []
    for pid in bullpen_ids:
        if pid in used_set:
            continue
        p = players.get(f"ID{pid}", {})
        person = p.get("person", {})
        bullpen.append({
            "id": pid,
            "name": person.get("fullName", f"#{pid}"),
            "jersey": p.get("jerseyNumber", ""),
        })

    # Bench — position players not in batting order
    bench = []
    for pid in bench_ids:
        p = players.get(f"ID{pid}", {})
        person = p.get("person", {})
        batting = p.get("stats", {}).get("batting", {})
        bench.append({
            "id": pid,
            "name": person.get("fullName", f"#{pid}"),
            "jersey": p.get("jerseyNumber", ""),
            "position": p.get("position", {}).get("abbreviation", ""),
            "stats": {
                "ab": batting.get("atBats", 0) or 0,
                "h": batting.get("hits", 0) or 0,
                "rbi": batting.get("rbi", 0) or 0,
            },
        })

    return {
        "team_name": team_name,
        "abbreviation": abbreviation,
        "lineup": lineup,
        "pitcher": pitcher_info,
        "pitchers_used": prev_pitchers,
        "bullpen": bullpen,
        "bench": bench,
    }


def _extract_linescore(linescore: dict) -> dict:
    """Extract inning-by-inning runs + R/H/E totals."""
    innings = []
    for inn in linescore.get("innings", []):
        innings.append({
            "num": inn.get("num", 0),
            "away": inn.get("away", {}).get("runs"),
            "home": inn.get("home", {}).get("runs"),
        })

    def _team_totals(side: str) -> dict:
        t = linescore.get("teams", {}).get(side, {})
        return {
            "runs": t.get("runs", 0) or 0,
            "hits": t.get("hits", 0) or 0,
            "errors": t.get("errors", 0) or 0,
        }

    return {
        "innings": innings,
        "away": _team_totals("away"),
        "home": _team_totals("home"),
    }


def _extract_plays(all_plays: list) -> list[dict]:
    """Extract completed plays as display-friendly dicts."""
    plays = []
    for play in all_plays:
        about = play.get("about", {})
        result = play.get("result", {})
        if not about.get("isComplete", False):
            continue
        plays.append({
            "inning": about.get("inning", 0),
            "half": about.get("halfInning", "top"),
            "event": result.get("event", ""),
            "description": result.get("description", ""),
            "is_scoring": bool(about.get("isScoringPlay", False)),
        })
    return plays
