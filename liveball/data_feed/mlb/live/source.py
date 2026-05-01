"""Game data source protocol — abstraction over REST polling vs WebSocket push."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class GameUpdate:
    """Single state update from any data source."""

    flat_state: dict  # same shape as client.extract_flat_state output
    gumbo: dict | None  # full gumbo for polling; None for WebSocket
    response_ms: float  # fetch/receive latency


class SourceError(Exception):
    """Recoverable error from a data source."""

    def __init__(self, message: str, *, status_code: int | None = None, response_ms: float = 0.0):
        super().__init__(message)
        self.status_code = status_code
        self.response_ms = response_ms


@runtime_checkable
class GameSource(Protocol):
    """Protocol for game data sources.

    Lifecycle: open() → [next_update() | set_interval()]* → close()
    """

    async def open(self) -> None:
        """Initialize the source (create client, connect, etc)."""
        ...

    async def next_update(self) -> GameUpdate:
        """Return the next game state update.

        For polling: sleeps for the configured interval, then fetches.
        For WebSocket: blocks until the next event arrives.

        Raises SourceError on recoverable failures.
        """
        ...

    def set_interval(self, interval: float) -> None:
        """Hint the source on timing for the next update.

        Polling sources use this to set their sleep duration.
        WebSocket sources ignore it (server-push model).
        """
        ...

    async def close(self) -> None:
        """Clean up resources (close connections)."""
        ...
