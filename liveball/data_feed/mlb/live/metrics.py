"""In-memory per-game polling metrics — ring buffer + aggregates."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PollEntry:
    ts: float
    response_ms: float
    changed: bool
    error: str | None
    event_type: str | None = None
    mlb_event_ts: str | None = None  # ISO-8601 UTC — currentPlay.about.endTime when isComplete
    interval_s: float | None = None  # sleep interval used after this poll
    inning: int | None = None
    inning_half: str | None = None
    inning_transition: bool = False  # True when inning or inning_half flipped vs previous poll


@dataclass
class GameMetrics:
    game_pk: int
    poll_count: int = 0
    success_count: int = 0
    error_count: int = 0
    changed_count: int = 0
    consecutive_errors: int = 0
    last_error: str | None = None
    last_poll_at: float | None = None
    last_change_at: float | None = None
    avg_response_ms: float = 0.0
    max_response_ms: float = 0.0
    poll_interval_s: float = 0.0
    started_at: float = field(default_factory=time.time)
    abstract_game_state: str = ""
    history: deque[PollEntry] = field(default_factory=lambda: deque(maxlen=15000))

    def record_poll(self, entry: PollEntry, interval: float, state: str) -> None:
        self.history.append(entry)
        self.poll_count += 1
        self.last_poll_at = entry.ts
        self.poll_interval_s = interval
        self.abstract_game_state = state

        if entry.error:
            self.error_count += 1
            self.last_error = entry.error
            self.consecutive_errors += 1
        else:
            self.success_count += 1
            self.consecutive_errors = 0
            self.avg_response_ms = (
                (self.avg_response_ms * (self.success_count - 1) + entry.response_ms)
                / self.success_count
            )
            self.max_response_ms = max(self.max_response_ms, entry.response_ms)

        if entry.changed:
            self.changed_count += 1
            self.last_change_at = entry.ts


# Module-level store — safe because tracker + server share the same event loop
_store: dict[int, GameMetrics] = {}


def get_or_create(game_pk: int) -> GameMetrics:
    if game_pk not in _store:
        _store[game_pk] = GameMetrics(game_pk=game_pk)
    return _store[game_pk]


def remove(game_pk: int) -> None:
    """Remove a game's metrics from the store after Final + flush."""
    _store.pop(game_pk, None)


def snapshot() -> list[dict]:
    """Serialize all game metrics for the REST endpoint."""
    results = []
    for gm in _store.values():
        results.append({
            "game_pk": gm.game_pk,
            "abstract_game_state": gm.abstract_game_state,
            "poll_count": gm.poll_count,
            "success_count": gm.success_count,
            "error_count": gm.error_count,
            "consecutive_errors": gm.consecutive_errors,
            "changed_count": gm.changed_count,
            "last_error": gm.last_error,
            "last_poll_at": gm.last_poll_at,
            "last_change_at": gm.last_change_at,
            "avg_response_ms": round(gm.avg_response_ms, 1),
            "max_response_ms": round(gm.max_response_ms, 1),
            "poll_interval_s": gm.poll_interval_s,
            "started_at": gm.started_at,
            "history": [
                {
                    "ts": e.ts,
                    "response_ms": round(e.response_ms, 1),
                    "changed": e.changed,
                    "error": e.error,
                    "event_type": e.event_type,
                    "mlb_event_ts": e.mlb_event_ts,
                    "interval_s": e.interval_s,
                    "inning": e.inning,
                    "inning_half": e.inning_half,
                    "inning_transition": e.inning_transition,
                }
                for e in gm.history
            ],
        })
    return results
