"""MLB Gumbo dlt resources — per-pitch timestamps from the live feed endpoint."""
from datetime import datetime

import dlt

from ..statsapi.client import MLBClient


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _runner_id(matchup: dict, key: str) -> int | None:
    node = matchup.get(key) or {}
    val = node.get("id")
    return int(val) if val is not None else None


def _flatten_game(game_pk: int, gumbo: dict) -> list[dict]:
    """Flatten a gumbo payload into one row per playEvents[j]."""
    plays = (gumbo.get("liveData") or {}).get("plays", {}).get("allPlays") or []
    rows: list[dict] = []

    for play in plays:
        about = play.get("about") or {}
        result = play.get("result") or {}
        matchup = play.get("matchup") or {}
        batter = matchup.get("batter") or {}
        pitcher = matchup.get("pitcher") or {}

        at_bat_index = about.get("atBatIndex")
        if at_bat_index is None:
            continue
        at_bat_number = at_bat_index + 1

        pa_ctx = {
            "inning": about.get("inning"),
            "half_inning": about.get("halfInning"),
            "is_top_inning": about.get("isTopInning"),
            "batter_id": batter.get("id"),
            "pitcher_id": pitcher.get("id"),
            "pa_event": result.get("event"),
            "pa_event_type": result.get("type"),
            "pa_description": result.get("description"),
            "pa_is_scoring_play": about.get("isScoringPlay"),
            "pa_has_out": about.get("hasOut"),
            "pa_rbi": result.get("rbi"),
            "pa_away_score": result.get("awayScore"),
            "pa_home_score": result.get("homeScore"),
            "pa_post_on_first_id": _runner_id(matchup, "postOnFirst"),
            "pa_post_on_second_id": _runner_id(matchup, "postOnSecond"),
            "pa_post_on_third_id": _runner_id(matchup, "postOnThird"),
            "pa_start_ts": _parse_ts(about.get("startTime")),
            "pa_end_ts": _parse_ts(about.get("endTime")),
        }

        for ev in play.get("playEvents") or []:
            details = ev.get("details") or {}
            count = ev.get("count") or {}
            pdata = ev.get("pitchData") or {}
            coords = pdata.get("coordinates") or {}

            rows.append({
                "game_pk": game_pk,
                "at_bat_number": at_bat_number,
                "play_event_index": ev.get("index"),
                "pitch_number": ev.get("pitchNumber"),
                "play_id": ev.get("playId"),
                "event_type": ev.get("type"),
                "event_start_ts": _parse_ts(ev.get("startTime")),
                "event_end_ts": _parse_ts(ev.get("endTime")),
                "event_description": details.get("description"),
                "is_pitch": ev.get("isPitch"),
                "is_in_play": details.get("isInPlay"),
                "is_ball": details.get("isBall"),
                "is_strike": details.get("isStrike"),
                "post_balls": count.get("balls"),
                "post_strikes": count.get("strikes"),
                "post_outs": count.get("outs"),
                "pitch_start_speed": pdata.get("startSpeed"),
                "pitch_plate_x": coords.get("pX"),
                "pitch_plate_z": coords.get("pZ"),
                **pa_ctx,
            })

    return rows


@dlt.resource(write_disposition="append")
def events(game_pks: list[int], log=None, extracted_pks: set | None = None):
    """Fetch MLB Gumbo play-by-play events for a batch of games.

    One API call per game_pk. Yields one record per playEvents[] entry
    (pitches AND actions like pickoffs / stolen bases), flattened with
    parent play-level context.

    Args:
        game_pks: List of MLB game PKs to fetch
        log: Optional logger
        extracted_pks: Optional set — only game_pks that returned data are added.
            Games where Gumbo returns no plays are skipped and retried next run.
    """
    client = MLBClient()

    if log:
        log.info(f"[gumbo] Fetching {len(game_pks)} games")

    total = 0
    errors = 0
    for i, game_pk in enumerate(game_pks):
        try:
            gumbo = client.get(f"/api/v1.1/game/{game_pk}/feed/live", log=log)
            rows = _flatten_game(game_pk, gumbo)

            if not rows:
                if log:
                    log.info(f"[gumbo] Game {game_pk}: no plays, will retry next run")
                continue

            total += len(rows)
            yield rows

            if extracted_pks is not None:
                extracted_pks.add(game_pk)

            if log and (i + 1) % 50 == 0:
                log.info(f"[gumbo] Progress: {i + 1}/{len(game_pks)} games")

        except Exception as e:
            errors += 1
            if log:
                log.warning(f"[gumbo] Game {game_pk} failed, skipping: {e}")

    if log:
        log.info(f"[gumbo] Done. {total} rows, {errors} errors")
