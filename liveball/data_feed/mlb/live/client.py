"""MLB live game feed API client."""

import json
import logging

import httpx

logger = logging.getLogger(__name__)

MLB_LIVE_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


class FetchError(Exception):
    """Raised when MLB API returns an error — carries detail for metrics."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _extract_current_ab(current_play: dict) -> dict | None:
    """Extract pitch-by-pitch data for the current at-bat."""
    matchup = current_play.get("matchup", {})
    events = current_play.get("playEvents", [])

    pitches = []
    sz_top = sz_bottom = None

    for ev in events:
        if ev.get("type") != "pitch":
            continue
        pd = ev.get("pitchData", {})
        coords = pd.get("coordinates", {})
        details = ev.get("details", {})

        if sz_top is None:
            sz_top = pd.get("strikeZoneTop")
            sz_bottom = pd.get("strikeZoneBottom")

        pitches.append({
            "num": ev.get("pitchNumber"),
            "type_code": details.get("type", {}).get("code"),
            "type_name": details.get("type", {}).get("description"),
            "speed": pd.get("startSpeed"),
            "px": coords.get("pX"),
            "pz": coords.get("pZ"),
            "result": details.get("description"),
            "is_strike": details.get("isStrike", False),
            "is_ball": details.get("isBall", False),
            "is_in_play": details.get("isInPlay", False),
            "balls_after": ev.get("count", {}).get("balls"),
            "strikes_after": ev.get("count", {}).get("strikes"),
        })

    return {
        "batter_id": matchup.get("batter", {}).get("id"),
        "batter_name": matchup.get("batter", {}).get("fullName"),
        "pitcher_id": matchup.get("pitcher", {}).get("id"),
        "pitcher_name": matchup.get("pitcher", {}).get("fullName"),
        "bat_side": matchup.get("batSide", {}).get("code"),
        "sz_top": sz_top,
        "sz_bottom": sz_bottom,
        "pitches": pitches,
    }


def extract_flat_state(data: dict) -> dict:
    """Extract flat game state dict from full gumbo response."""
    game_data = data.get("gameData", {})
    live_data = data.get("liveData", {})
    linescore = live_data.get("linescore", {})
    current_play = live_data.get("plays", {}).get("currentPlay", {})
    matchup = current_play.get("matchup", {})

    status = game_data.get("status", {})

    # Runner positions from offense
    offense = linescore.get("offense", {})
    runners = {}
    for base in ("first", "second", "third"):
        runner = offense.get(base)
        if runner:
            runners[base] = runner.get("fullName", True)

    # Last play description
    result = current_play.get("result", {})
    last_play = result.get("description")

    return {
        "abstract_game_state": status.get("abstractGameState"),
        "status": status.get("detailedState"),
        "away_score": linescore.get("teams", {}).get("away", {}).get("runs"),
        "home_score": linescore.get("teams", {}).get("home", {}).get("runs"),
        "inning": linescore.get("currentInning"),
        "inning_half": linescore.get("inningHalf"),
        "outs": linescore.get("outs"),
        "balls": linescore.get("balls"),
        "strikes": linescore.get("strikes"),
        "runners": json.dumps(runners) if runners else None,
        "current_batter_id": matchup.get("batter", {}).get("id"),
        "current_batter_name": matchup.get("batter", {}).get("fullName"),
        "current_pitcher_id": matchup.get("pitcher", {}).get("id"),
        "current_pitcher_name": matchup.get("pitcher", {}).get("fullName"),
        "last_play": last_play,
        "event_type": result.get("event"),  # e.g. "Strikeout", "Mound Visit", "Single"
        "current_ab": _extract_current_ab(current_play),
        # When the play completed on MLB's side — used to measure detection latency
        "mlb_event_ts": (
            current_play.get("about", {}).get("endTime")
            if current_play.get("about", {}).get("isComplete")
            else None
        ),
    }


async def fetch_game_data(
    game_pk: int, client: httpx.AsyncClient
) -> tuple[dict, dict]:
    """Fetch live game data from the MLB Stats API.

    Returns (flat_state, full_gumbo). The flat_state matches our SQLite
    schema; the full gumbo is passed to SimContext.to_game_input().
    Raises FetchError on failure with status code and detail.
    """
    url = MLB_LIVE_URL.format(game_pk=game_pk)
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except httpx.TimeoutException:
        raise FetchError("timeout")
    except httpx.ConnectError:
        raise FetchError("connection_refused")
    except httpx.HTTPStatusError as e:
        raise FetchError(f"{e.response.status_code} {e.response.reason_phrase}", status_code=e.response.status_code)
    except httpx.HTTPError as e:
        raise FetchError(str(e))

    data = resp.json()
    return extract_flat_state(data), data
