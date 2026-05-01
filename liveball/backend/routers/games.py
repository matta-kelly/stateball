import asyncio
import json
import logging
import os
import statistics
import time
from datetime import date, datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from backend.db import get_conn

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 15  # seconds

DATA_FEED_URL = os.environ.get("DATA_FEED_URL", "http://liveball-data-feed:8001")

# Reusable client for sync data-feed calls — avoids socket churn
_feed_client = httpx.Client(timeout=5.0, base_url=DATA_FEED_URL)

router = APIRouter()


@router.get("/games")
def list_games(
    date: date = Query(default_factory=lambda: datetime.now(timezone.utc).date()),
    game_type: str | None = Query(None),
):
    # Try data feed first — it has live state (inning, outs, runners, etc.)
    # that DuckLake doesn't. Only fall through to DuckLake if data feed has
    # no games for the requested date (i.e. historical).
    try:
        resp = _feed_client.get("/games")
        resp.raise_for_status()
        feed_games = resp.json()
        date_str = str(date)
        matched = [g for g in feed_games if g.get("game_date", "").startswith(date_str)]
        if matched:
            if game_type:
                matched = [g for g in matched if g.get("game_type") == game_type]
            return matched
    except httpx.HTTPError:
        pass  # Fall through to DuckLake

    conn = get_conn()

    sql = (
        "SELECT game_pk, game_date, game_datetime, game_type, status, "
        "abstract_game_state, away_team_id, away_team_name, "
        "home_team_id, home_team_name, away_score, home_score, "
        "venue_name "
        "FROM lh.main.proc_mlb__games "
        "WHERE game_date = ?"
    )
    params: list = [str(date)]

    if game_type:
        sql += " AND game_type = ?"
        params.append(game_type)

    sql += " ORDER BY game_datetime ASC"

    return _query_games(conn, sql, params)


@router.get("/games/trackable")
def trackable_games(hours_back: int = Query(default=8)):
    """Return games the data feed should be tracking.

    Includes all non-Final games plus recently finished games.
    """
    conn = get_conn()

    sql = (
        "SELECT g.game_pk, g.game_date, g.game_datetime, g.game_type, g.status, "
        "g.abstract_game_state, g.away_team_id, g.away_team_name, "
        "g.home_team_id, g.home_team_name, g.away_score, g.home_score, "
        "g.venue_name "
        "FROM lh.main.proc_mlb__games g "
        "WHERE CAST(g.game_date AS DATE) <= CURRENT_DATE + INTERVAL 1 DAY "
        "  AND (g.abstract_game_state != 'Final' "
        f"      OR g.game_datetime > now() - INTERVAL {int(hours_back)} HOUR) "
        "ORDER BY g.game_datetime ASC"
    )
    return _query_games(conn, sql, [])


@router.get("/games/live")
def live_games():
    """Proxy to data feed for current trackable games."""
    resp = _feed_client.get("/games")
    resp.raise_for_status()
    return resp.json()


@router.get("/feed/metrics")
def feed_metrics():
    """Proxy to data feed for tracker polling metrics."""
    resp = _feed_client.get("/metrics")
    resp.raise_for_status()
    return resp.json()


@router.get("/feed/health")
def feed_health():
    """Live system health summary + per-game alerts derived from in-memory metrics."""
    try:
        resp = _feed_client.get("/metrics")
        resp.raise_for_status()
        metrics = resp.json()
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Data feed unavailable")

    now = time.time()
    alerts = []
    total_polls = 0
    total_errors = 0
    active_trackers = 0

    for gm in metrics:
        state = gm.get("abstract_game_state", "")
        poll_count = gm.get("poll_count", 0)
        error_count = gm.get("error_count", 0)
        total_polls += poll_count
        total_errors += error_count
        if state == "Live":
            active_trackers += 1

        game_pk = gm["game_pk"]

        if gm.get("consecutive_errors", 0) >= 3:
            alerts.append({
                "game_pk": game_pk,
                "level": "error",
                "message": f"{gm['consecutive_errors']} consecutive errors — last: {gm.get('last_error', 'unknown')}",
            })

        if poll_count > 10 and error_count / poll_count > 0.10:
            alerts.append({
                "game_pk": game_pk,
                "level": "warn",
                "message": f"High error rate ({error_count / poll_count * 100:.0f}% of {poll_count} polls)",
            })

        if gm.get("avg_response_ms", 0) > 2000:
            alerts.append({
                "game_pk": game_pk,
                "level": "warn",
                "message": f"Slow responses (avg {gm['avg_response_ms']:.0f}ms)",
            })

        if state == "Live":
            last_change = gm.get("last_change_at")
            if last_change and (now - last_change) > 120:
                alerts.append({
                    "game_pk": game_pk,
                    "level": "warn",
                    "message": f"No change in {int(now - last_change)}s for Live game",
                })

    return {
        "summary": {
            "active_trackers": active_trackers,
            "total_trackers": len(metrics),
            "total_polls": total_polls,
            "error_rate_pct": round(total_errors / total_polls * 100, 1) if total_polls else 0.0,
        },
        "alerts": alerts,
    }


@router.get("/feed/history")
def feed_history():
    """List completed game poll logs available in S3."""
    import s3fs

    bucket = os.environ.get("S3_BUCKET", "dazoo")
    prefix = f"{bucket}/stateball/liveball/poll_logs"

    try:
        fs = s3fs.S3FileSystem(
            key=os.environ.get("S3_ACCESS_KEY_ID", ""),
            secret=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
            endpoint_url=os.environ.get("S3_ENDPOINT", ""),
        )
        files = fs.glob(f"{prefix}/*/*.jsonl")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"S3 unavailable: {e}")

    games = []
    for path in sorted(files, reverse=True):
        # path: dazoo/stateball/liveball/poll_logs/2026-03-17/12345.jsonl
        parts = path.split("/")
        if len(parts) < 2:
            continue
        game_date = parts[-2]
        game_pk_str = parts[-1].replace(".jsonl", "")
        try:
            games.append({"game_pk": int(game_pk_str), "game_date": game_date})
        except ValueError:
            continue

    return games


@router.get("/feed/history/{game_pk}")
def feed_game_history(game_pk: int, game_date: str = Query(...)):
    """Load S3 poll log for a completed game and compute aggregates."""
    import s3fs

    bucket = os.environ.get("S3_BUCKET", "dazoo")
    path = f"{bucket}/stateball/liveball/poll_logs/{game_date}/{game_pk}.jsonl"

    try:
        fs = s3fs.S3FileSystem(
            key=os.environ.get("S3_ACCESS_KEY_ID", ""),
            secret=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
            endpoint_url=os.environ.get("S3_ENDPOINT", ""),
        )
        with fs.open(f"s3://{path}", "r") as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Poll log not found")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"S3 error: {e}")

    if not entries:
        raise HTTPException(status_code=404, detail="Poll log is empty")

    total = len(entries)
    errors = [e for e in entries if e.get("error")]
    changed = [e for e in entries if e.get("changed")]
    response_times = [e["response_ms"] for e in entries if not e.get("error")]

    # Polls-per-change by event_type:
    # After each change (event_type=X), count subsequent no-change polls until next change.
    # This tells us how many polls were "wasted" after each event type.
    idle_gaps: dict[str, list[int]] = {}
    idle_count = 0
    last_event_type: str | None = None

    for entry in entries:
        if entry.get("changed"):
            if last_event_type is not None:
                idle_gaps.setdefault(last_event_type, []).append(idle_count)
            idle_count = 0
            last_event_type = entry.get("event_type")
        else:
            idle_count += 1

    polls_per_change_by_event = {
        et: round(statistics.median(gaps), 1)
        for et, gaps in idle_gaps.items()
        if gaps
    }

    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "total_polls": total,
        "change_rate": round(len(changed) / total, 3) if total else 0,
        "wasted_poll_pct": round((1 - len(changed) / total) * 100, 1) if total else 0,
        "error_rate": round(len(errors) / total, 3) if total else 0,
        "avg_response_ms": round(statistics.mean(response_times), 1) if response_times else 0,
        "polls_per_change_by_event": polls_per_change_by_event,
        "entries": entries,
    }


@router.get("/games/stream", response_class=EventSourceResponse)
async def stream_games(request: Request):
    """Proxy SSE stream from data feed to frontend.

    Injects heartbeat comments every HEARTBEAT_INTERVAL seconds so the
    client can distinguish "quiet game" from "dead stream".
    """
    headers = {}
    last_id = request.headers.get("last-event-id")
    if last_id:
        headers["Last-Event-ID"] = last_id

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET", f"{DATA_FEED_URL}/stream", headers=headers
            ) as resp:
                line_iter = resp.aiter_lines().__aiter__()
                event_type = None
                event_id = None
                data_lines: list[str] = []

                while True:
                    try:
                        line = await asyncio.wait_for(
                            line_iter.__anext__(),
                            timeout=HEARTBEAT_INTERVAL,
                        )
                    except asyncio.TimeoutError:
                        yield ServerSentEvent(comment="heartbeat")
                        continue
                    except StopAsyncIteration:
                        logger.info("SSE: upstream closed")
                        break

                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("id:"):
                        event_id = line[3:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif line == "" and data_lines:
                        yield ServerSentEvent(
                            raw_data="\n".join(data_lines),
                            event=event_type,
                            id=event_id,
                        )
                        event_type = None
                        event_id = None
                        data_lines = []
    except httpx.HTTPError as exc:
        logger.warning("SSE: upstream error: %s", exc)
    except Exception as exc:
        logger.warning("SSE: unexpected error: %s", exc)


@router.get("/games/{game_pk}")
def get_game(game_pk: int):
    """Look up a single game by game_pk. Data feed first, then DuckLake."""
    try:
        resp = _feed_client.get("/games")
        resp.raise_for_status()
        for g in resp.json():
            if g.get("game_pk") == game_pk:
                return g
    except httpx.HTTPError:
        pass

    conn = get_conn()
    sql = (
        "SELECT game_pk, game_date, game_datetime, game_type, status, "
        "abstract_game_state, away_team_id, away_team_name, "
        "home_team_id, home_team_name, away_score, home_score, "
        "venue_name "
        "FROM lh.main.proc_mlb__games "
        "WHERE game_pk = ?"
    )
    rows = _query_games(conn, sql, [game_pk])
    if not rows:
        raise HTTPException(status_code=404, detail="Game not found")
    return rows[0]


@router.get("/games/{game_pk}/detail")
def game_detail(game_pk: int):
    """Proxy to data feed for lineups, pitching, bullpen, play-by-play."""
    try:
        resp = _feed_client.get(f"/games/{game_pk}/detail")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="No live data")
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Data feed unavailable")


@router.get("/games/{game_pk}/market-history")
def market_history(game_pk: int):
    """Proxy to data feed for market snapshot time-series."""
    try:
        resp = _feed_client.get(f"/games/{game_pk}/market-history")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return []


@router.get("/games/{game_pk}/game-state-events")
def game_state_events(game_pk: int):
    """Proxy to data feed for game state transition events."""
    try:
        resp = _feed_client.get(f"/games/{game_pk}/game-state-events")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return []


@router.get("/games/{game_pk}/sim-results")
def sim_results(game_pk: int):
    """Proxy to data feed for sim result time-series."""
    try:
        resp = _feed_client.get(f"/games/{game_pk}/sim-results")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return []


@router.get("/games/{game_pk}/fills")
def game_fills(game_pk: int):
    """Proxy to data feed for trade fill history."""
    try:
        resp = _feed_client.get(f"/games/{game_pk}/fills")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return []


def _query_games(conn, sql: str, params: list) -> list[dict]:
    try:
        rows = conn.execute(sql, params).fetchall()
        cols = [d[0] for d in conn.description]
        results = []
        for row in rows:
            item = dict(zip(cols, row))
            for key in ("game_date", "game_datetime"):
                val = item.get(key)
                if val is not None:
                    # Use isoformat() for proper ISO 8601 (T separator)
                    item[key] = val.isoformat() if hasattr(val, "isoformat") else str(val)
            results.append(item)
        return results
    finally:
        conn.close()
