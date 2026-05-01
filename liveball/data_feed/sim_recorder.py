"""Sim result recorder — live-only, serves the WP chart's sim overlay.

Records sim probability outputs to the in-memory buffer and Redis for
live reads. Durable persistence of sim predictions is handled by the
trading engine's journal (packages/execution/trading/shared/flush.py →
lakehouse.main.trading_predictions), not here.
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


def record_result(snapshot: GameSnapshot, game_pk: int, result: dict) -> None:
    """Append sim result to in-memory buffer and Redis list.

    Called from snapshot.update_sim after current-state scalars are written.
    The result dict matches the sim worker output shape.
    """
    ts = datetime.now(timezone.utc).isoformat()

    entry = {
        "ts": ts,
        "p_home_win": result["p_home_win"],
        "p_home_win_se": result["p_home_win_se"],
        "we_baseline": result.get("we_baseline"),
        "n_sims": result.get("n_sims"),
        "duration_ms": result.get("duration_ms"),
        "mean_home_score": result.get("mean_home_score"),
        "mean_away_score": result.get("mean_away_score"),
    }

    # In-memory buffer (live serving)
    snapshot._sim_results.setdefault(game_pk, []).append(entry)

    # Redis list (mid-game restart survivability)
    if snapshot._redis is not None:
        asyncio.ensure_future(
            snapshot._r_rpush(f"simresults:{game_pk}", json.dumps(entry))
        )


def record_final(snapshot: GameSnapshot, game_pk: int) -> None:
    """Hook for game-final cleanup. No-op — in-memory buffer and Redis
    list are handled by snapshot.on_game_final. Kept for symmetry with
    market_recorder.record_final.
    """
    return
