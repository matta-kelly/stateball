"""In-play state adjustment for unresolved gumbo events.

When MLB's gumbo has a pitch with isInPlay=true but the play hasn't
resolved yet (about.isComplete=false), the linescore is stale. This
module detects that condition and adjusts the GameState based on the
pitch description:

  "In play, run(s)"  → batting team's score +1
  "In play, out(s)"  → outs +1
  "In play, no out"  → no adjustment

The adjustment is approximate — a grand slam scores 4, a double play
adds 2 outs — but directionally correct. When the resolved gumbo
arrives 2-5s later, to_game_input reads the now-correct linescore
and the adjustment is naturally gone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from sim.engine.core.state import GameState

logger = logging.getLogger(__name__)

_DESCRIPTION_MAP: dict[str, tuple[int, int]] = {
    "In play, run(s)": (1, 0),
    "In play, out(s)": (0, 1),
    "In play, no out": (0, 0),
}


@dataclass
class InPlayAdjustment:
    """Describes a state adjustment for an unresolved in-play event."""

    score_delta: int
    outs_delta: int


def detect(current_play: dict) -> InPlayAdjustment | None:
    """Detect an unresolved in-play event from a gumbo currentPlay dict.

    Returns an InPlayAdjustment if the play is in-play and unresolved,
    or None if the play is resolved, not in play, or data is missing.
    """
    if not current_play:
        return None

    about = current_play.get("about", {})
    if about.get("isComplete"):
        return None

    result = current_play.get("result", {})
    if result.get("event") is not None:
        return None

    play_events = current_play.get("playEvents", [])
    if not play_events:
        return None

    last_event = play_events[-1]
    details = last_event.get("details", {})
    if not details.get("isInPlay"):
        return None

    description = details.get("description", "")
    deltas = _DESCRIPTION_MAP.get(description)
    if deltas is None:
        logger.warning("Unknown in-play description: %r", description)
        return None

    return InPlayAdjustment(score_delta=deltas[0], outs_delta=deltas[1])


def apply(game_state: GameState, current_play: dict) -> GameState:
    """Adjust GameState for an unresolved in-play event.

    If no adjustment is needed, returns game_state unchanged.
    """
    adj = detect(current_play)
    if adj is None:
        return game_state
    if adj.score_delta == 0 and adj.outs_delta == 0:
        return game_state

    kwargs: dict = {}

    if adj.outs_delta:
        kwargs["outs"] = game_state.outs + adj.outs_delta

    if adj.score_delta:
        if game_state.half == 0:
            kwargs["away_score"] = game_state.away_score + adj.score_delta
        else:
            kwargs["home_score"] = game_state.home_score + adj.score_delta

    logger.debug(
        "In-play adjustment: %s → %s",
        adj,
        {k: v for k, v in kwargs.items()},
    )
    return replace(game_state, **kwargs)
