"""Game state event recorder — live-only, serves the WP chart's markers.

The tracker detects state changes via a dict diff (state != prev_state);
this module classifies each transition into a hierarchy-aware trigger
label that drives marker rendering on the frontend:

    inning changed      → inning_start  (green marker, highest)
    inning_half changed → half_start    (blue marker)
    outs increased      → out_recorded  (red marker)
    batter changed      → pa_start      (no marker; recorded for context)

Only the highest-level transition for a given diff gets written — this
matches the frontend marker hierarchy (end of inning is green, not also
blue+red).

Persistence scope: in-memory buffer + Redis list for live serving only.
No DuckLake write. Historical play data comes through the warehouse
pipeline from its own ingestion, not re-derived here.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_feed.snapshot import GameSnapshot

logger = logging.getLogger(__name__)


def _classify(state: dict, prev_state: dict | None) -> str | None:
    """Pick the single highest-level trigger that applies to this diff.

    Returns None if no trigger-worthy transition occurred (e.g., only
    ball/strike count changed).
    """
    if prev_state is None:
        return None

    # Inning change — highest level
    if state.get("inning") != prev_state.get("inning"):
        return "inning_start"

    # Half change within same inning
    if state.get("inning_half") != prev_state.get("inning_half"):
        return "half_start"

    # Outs increased. Skip when curr_outs == 3: the 3rd out always coincides
    # with a half/inning boundary and the next state transition will fire
    # half_start or inning_start — letting that one carry the marker keeps
    # the hierarchy honest (no red line directly adjacent to its blue/green).
    prev_outs = prev_state.get("outs") or 0
    curr_outs = state.get("outs") or 0
    if curr_outs > prev_outs and curr_outs < 3:
        return "out_recorded"

    # New batter
    if state.get("current_batter_id") != prev_state.get("current_batter_id"):
        return "pa_start"

    return None


def record_transition(
    snapshot: GameSnapshot,
    game_pk: int,
    state: dict,
    prev_state: dict | None,
) -> None:
    """Classify a state diff and, if it warrants a row, write it live.

    Called from the tracker when `state != prev_state`. No-op when the
    diff is count-only or not otherwise interesting.
    """
    trigger = _classify(state, prev_state)
    if trigger is None:
        return

    ts = datetime.now(timezone.utc).isoformat()

    entry = {
        "ts": ts,
        "trigger": trigger,
        "inning": state.get("inning"),
        "inning_half": state.get("inning_half"),
        "outs": state.get("outs"),
        "balls": state.get("balls"),
        "strikes": state.get("strikes"),
        "current_batter_id": state.get("current_batter_id"),
        "current_pitcher_id": state.get("current_pitcher_id"),
    }

    # In-memory buffer (live serving)
    snapshot._game_state_history.setdefault(game_pk, []).append(entry)

    # Redis list (mid-game restart survivability)
    if snapshot._redis is not None:
        asyncio.ensure_future(
            snapshot._r_rpush(f"gamestatehist:{game_pk}", json.dumps(entry))
        )


def record_final(snapshot: GameSnapshot, game_pk: int) -> None:
    """Hook for game-final cleanup. No-op — in-memory buffer and Redis
    list are handled by snapshot.on_game_final. Kept for symmetry with
    market_recorder.record_final.
    """
    return
