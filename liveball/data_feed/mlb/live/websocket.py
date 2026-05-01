"""MLB GameDay WebSocket source — push-triggered with diffPatch + polling fallback.

Connects to MLB's GameDay WebSocket to receive real-time push notifications.
On normal pushes, fetches a lightweight diffPatch and applies it to a cached
gumbo. On roster-change events or errors, falls back to a full gumbo fetch.
If the WebSocket dies, falls back to polling and periodically reconnects.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time

import httpx
import websockets

from data_feed.mlb.live.client import extract_flat_state
from data_feed.mlb.live.source import GameUpdate, SourceError

logger = logging.getLogger(__name__)

WS_URL = "wss://ws.statsapi.mlb.com/api/v1/game/push/subscribe/gameday/{game_pk}"
WS_FEED_URL = "https://ws.statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
DIFF_PATCH_URL = "https://ws.statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live/diffPatch"
POLL_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

WS_HEADERS = {
    "Origin": "https://www.mlb.com",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
}

HEARTBEAT_MSG = "Gameday5"
HEARTBEAT_INTERVAL = 10.0  # seconds between heartbeats
STALE_THRESHOLD = 120.0  # force reconnect if no data message for this long (> inning break)
RECONNECT_INTERVAL = 60.0  # seconds between reconnect attempts while polling

# Events that trigger a full refresh instead of diffPatch — roster/lineup changes
# that touch deep gumbo structures where accumulated patches are most likely to drift.
FULL_REFRESH_EVENTS = {
    "game_finished",
    "pitching_substitution",
    "offensive_substitution",
    "defensive_switch",
}


# ---------------------------------------------------------------------------
# JSON Patch (RFC 6902) — inline implementation, no external dependency
# ---------------------------------------------------------------------------

def _resolve_path(obj: dict | list, segments: list[str]) -> tuple:
    """Walk to the parent of the target, return (parent, final_key)."""
    for i, seg in enumerate(segments[:-1]):
        if isinstance(obj, list):
            obj = obj[int(seg)]
        else:
            if seg not in obj:
                obj[seg] = {} if not segments[i + 1].isdigit() else []
            obj = obj[seg]
    return obj, segments[-1]


def _get_value(obj: dict | list, path: str):
    """Read a value at a JSON Pointer path (e.g. /liveData/plays/allPlays/0)."""
    segments = path.strip("/").split("/")
    for seg in segments:
        if isinstance(obj, list):
            obj = obj[int(seg)]
        else:
            obj = obj[seg]
    return obj


def apply_patches(gumbo: dict, diffs: list[dict]) -> None:
    """Apply a list of RFC 6902 JSON Patch operations to gumbo in-place."""
    for diff in diffs:
        op = diff["op"]
        segments = diff["path"].strip("/").split("/")
        parent, key = _resolve_path(gumbo, segments)

        if op in ("add", "replace"):
            if isinstance(parent, list):
                idx = int(key)
                if op == "add":
                    parent.insert(idx, diff["value"])
                else:
                    parent[idx] = diff["value"]
            else:
                parent[key] = diff["value"]

        elif op == "remove":
            if isinstance(parent, list):
                parent.pop(int(key))
            else:
                del parent[key]

        elif op == "copy":
            val = copy.deepcopy(_get_value(gumbo, diff["from"]))
            if isinstance(parent, list):
                parent.insert(int(key), val)
            else:
                parent[key] = val

        elif op == "move":
            val = _get_value(gumbo, diff["from"])
            # Remove from source
            from_segments = diff["from"].strip("/").split("/")
            from_parent, from_key = _resolve_path(gumbo, from_segments)
            if isinstance(from_parent, list):
                from_parent.pop(int(from_key))
            else:
                del from_parent[from_key]
            # Add to destination
            if isinstance(parent, list):
                parent.insert(int(key), val)
            else:
                parent[key] = val


# ---------------------------------------------------------------------------
# WebSocketSource
# ---------------------------------------------------------------------------


class WebSocketSource:
    """MLB GameDay WebSocket source with diffPatch and automatic polling fallback."""

    def __init__(self, game_pk: int, *, ws_timeout: float = 10.0, fetch_timeout: float = 10.0):
        self._game_pk = game_pk
        self._ws_timeout = ws_timeout
        self._fetch_timeout = fetch_timeout

        self._client: httpx.AsyncClient | None = None
        self._ws: websockets.ClientConnection | None = None
        self._connected = False

        # Cached gumbo — maintained incrementally via diffPatch
        self._cached_gumbo: dict | None = None

        # Polling fallback state
        self._poll_interval: float = 2.0
        self._last_reconnect_attempt: float = 0.0

        # Duplicate detection (from gameday bot pattern)
        self._last_ws_timestamp: str | None = None
        self._last_ws_data_len: int = 0
        self._last_data_at: float = 0.0  # monotonic time of last real data message

    async def open(self) -> None:
        self._client = httpx.AsyncClient(timeout=self._fetch_timeout)
        # Seed the gumbo cache with a full fetch
        try:
            url = WS_FEED_URL.format(game_pk=self._game_pk)
            resp = await self._client.get(url)
            resp.raise_for_status()
            self._cached_gumbo = resp.json()
            logger.info("[%d] Initial gumbo cached", self._game_pk)
        except Exception:
            logger.warning("[%d] Failed to seed gumbo cache", self._game_pk, exc_info=True)
        await self._connect_ws()

    async def _connect_ws(self) -> None:
        """Attempt WebSocket connection. On failure, log and set _connected=False."""
        url = WS_URL.format(game_pk=self._game_pk)
        try:
            self._ws = await websockets.connect(
                url,
                additional_headers=WS_HEADERS,
                open_timeout=self._ws_timeout,
                close_timeout=5,
            )
            await self._ws.send(HEARTBEAT_MSG)
            self._connected = True
            self._last_data_at = time.monotonic()
            logger.info("[%d] WebSocket connected", self._game_pk)
        except Exception:
            self._connected = False
            self._ws = None
            logger.warning("[%d] WebSocket connect failed — falling back to polling", self._game_pk, exc_info=True)

    async def _maybe_reconnect(self) -> None:
        """Attempt WebSocket reconnect if enough time has passed."""
        now = time.monotonic()
        if now - self._last_reconnect_attempt < RECONNECT_INTERVAL:
            return
        self._last_reconnect_attempt = now
        logger.info("[%d] Attempting WebSocket reconnect...", self._game_pk)
        await self._connect_ws()

    async def next_update(self) -> GameUpdate:
        if self._connected:
            return await self._next_ws_update()
        return await self._next_poll_update()

    async def _next_ws_update(self) -> GameUpdate:
        """Wait for a WebSocket push, then fetch via diffPatch or full gumbo."""
        while True:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                # Check for silently dead connection
                if self._last_data_at and (time.monotonic() - self._last_data_at) > STALE_THRESHOLD:
                    self._connected = False
                    logger.warning("[%d] WebSocket stale (no data for %.0fs) — falling back to polling",
                                   self._game_pk, time.monotonic() - self._last_data_at)
                    return await self._next_poll_update()
                try:
                    await self._ws.send(HEARTBEAT_MSG)
                except Exception:
                    self._connected = False
                    logger.warning("[%d] WebSocket heartbeat failed — falling back to polling", self._game_pk)
                    return await self._next_poll_update()
                continue
            except websockets.ConnectionClosed as e:
                self._connected = False
                logger.warning("[%d] WebSocket closed (code=%s) — falling back to polling", self._game_pk, e.code)
                return await self._next_poll_update()
            except Exception:
                self._connected = False
                logger.warning("[%d] WebSocket error — falling back to polling", self._game_pk, exc_info=True)
                return await self._next_poll_update()

            # Duplicate detection
            msg = json.loads(raw)
            ts = msg.get("timeStamp")
            data_len = len(raw)
            if ts == self._last_ws_timestamp and data_len == self._last_ws_data_len:
                logger.debug("[%d] Duplicate WebSocket message — skipping", self._game_pk)
                continue
            self._last_ws_timestamp = ts
            self._last_ws_data_len = data_len
            self._last_data_at = time.monotonic()

            # Decide: full fetch or diffPatch
            update_id = msg.get("updateId", "")
            game_events = set(msg.get("gameEvents", []))
            change_type = msg.get("changeEvent", {}).get("type")
            needs_full = (
                change_type == "full_refresh"
                or FULL_REFRESH_EVENTS & game_events
                or self._cached_gumbo is None
            )

            if needs_full:
                logger.debug("[%d] Full fetch (reason: %s)", self._game_pk,
                             change_type if change_type == "full_refresh"
                             else (FULL_REFRESH_EVENTS & game_events) or "no cache")
                return await self._full_fetch(update_id)

            # diffPatch path
            timecode = self._cached_gumbo.get("metaData", {}).get("timeStamp", "")
            diff_url = (
                f"{DIFF_PATCH_URL.format(game_pk=self._game_pk)}"
                f"?language=en&startTimecode={timecode}&pushUpdateId={update_id}"
            )
            try:
                patches = await self._fetch_json(diff_url)
                if not isinstance(patches, list):
                    logger.debug("[%d] diffPatch returned %s — full fetch", self._game_pk, type(patches).__name__)
                    return await self._full_fetch(update_id)
                for patch_obj in patches:
                    if not isinstance(patch_obj, dict):
                        logger.debug("[%d] diffPatch item is %s — full fetch", self._game_pk, type(patch_obj).__name__)
                        return await self._full_fetch(update_id)
                    apply_patches(self._cached_gumbo, patch_obj.get("diff", []))
                state = extract_flat_state(self._cached_gumbo)
                return GameUpdate(
                    flat_state=state,
                    gumbo=self._cached_gumbo,
                    response_ms=(time.monotonic() - self._last_data_at) * 1000,
                )
            except Exception:
                logger.warning("[%d] diffPatch failed — falling back to full fetch", self._game_pk, exc_info=True)
                return await self._full_fetch(update_id)

    async def _full_fetch(self, update_id: str) -> GameUpdate:
        """Fetch full gumbo, replace cache, return GameUpdate."""
        url = f"{WS_FEED_URL.format(game_pk=self._game_pk)}?language=en&pushUpdateId={update_id}"
        update = await self._fetch_gumbo(url)
        self._cached_gumbo = update.gumbo
        return update

    async def _next_poll_update(self) -> GameUpdate:
        """Polling fallback — sleep then fetch full gumbo."""
        await asyncio.sleep(self._poll_interval)
        await self._maybe_reconnect()
        url = POLL_FEED_URL.format(game_pk=self._game_pk)
        update = await self._fetch_gumbo(url)
        self._cached_gumbo = update.gumbo
        return update

    async def _fetch_gumbo(self, url: str) -> GameUpdate:
        """Fetch full gumbo, extract flat_state, return GameUpdate."""
        t0 = time.monotonic()
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except httpx.TimeoutException:
            elapsed = (time.monotonic() - t0) * 1000
            raise SourceError("timeout", response_ms=elapsed)
        except httpx.HTTPStatusError as e:
            elapsed = (time.monotonic() - t0) * 1000
            raise SourceError(
                f"{e.response.status_code} {e.response.reason_phrase}",
                status_code=e.response.status_code,
                response_ms=elapsed,
            )
        except httpx.HTTPError as e:
            elapsed = (time.monotonic() - t0) * 1000
            raise SourceError(str(e), response_ms=elapsed)

        elapsed_ms = (time.monotonic() - t0) * 1000
        gumbo = resp.json()
        state = extract_flat_state(gumbo)
        return GameUpdate(flat_state=state, gumbo=gumbo, response_ms=elapsed_ms)

    async def _fetch_json(self, url: str) -> list:
        """Fetch JSON from a URL, return parsed response. Raises on failure."""
        t0 = time.monotonic()
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except httpx.TimeoutException:
            elapsed = (time.monotonic() - t0) * 1000
            raise SourceError("timeout", response_ms=elapsed)
        except httpx.HTTPStatusError as e:
            elapsed = (time.monotonic() - t0) * 1000
            raise SourceError(
                f"{e.response.status_code} {e.response.reason_phrase}",
                status_code=e.response.status_code,
                response_ms=elapsed,
            )
        except httpx.HTTPError as e:
            elapsed = (time.monotonic() - t0) * 1000
            raise SourceError(str(e), response_ms=elapsed)
        return resp.json()

    def set_interval(self, interval: float) -> None:
        self._poll_interval = interval

    async def close(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
