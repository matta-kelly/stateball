"""REST polling implementation of GameSource."""

from __future__ import annotations

import asyncio
import time

import httpx

from data_feed.mlb.live.client import FetchError, fetch_game_data
from data_feed.mlb.live.source import GameUpdate, SourceError


class PollingSource:
    """Polls MLB Stats API on a configurable interval."""

    def __init__(self, game_pk: int, *, timeout: float = 15.0):
        self._game_pk = game_pk
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._interval: float = 2.0
        self._first_call: bool = True

    async def open(self) -> None:
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def next_update(self) -> GameUpdate:
        if self._first_call:
            self._first_call = False
        else:
            await asyncio.sleep(self._interval)

        t0 = time.monotonic()
        try:
            state, gumbo = await fetch_game_data(self._game_pk, self._client)
        except FetchError as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            raise SourceError(str(e), status_code=e.status_code, response_ms=elapsed_ms) from e

        elapsed_ms = (time.monotonic() - t0) * 1000
        return GameUpdate(flat_state=state, gumbo=gumbo, response_ms=elapsed_ms)

    def set_interval(self, interval: float) -> None:
        self._interval = interval

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
