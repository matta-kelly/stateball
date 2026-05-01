"""Per-game SimContext cache — hydrates once, reuses for each PA.

Lives in the main process only. DuckDB connections are not fork-safe,
so all hydration (DuckLake queries) happens here, not in child workers.
"""

import asyncio
import logging

from sim.game_inputs.live_context import SimContext

logger = logging.getLogger(__name__)


class GameHydrator:
    """Cache of SimContexts keyed by game_pk."""

    def __init__(self, conn):
        self._conn = conn
        self._contexts: dict[int, SimContext] = {}
        self._lock = asyncio.Lock()

    async def ensure_hydrated(
        self, game_pk: int, game_date: str, gumbo: dict
    ) -> SimContext:
        """Return cached SimContext, hydrating on first call for this game.

        Serialized via lock because DuckDB connections are not thread-safe —
        concurrent run_in_executor calls with the same conn would deadlock.
        """
        if game_pk in self._contexts:
            return self._contexts[game_pk]

        async with self._lock:
            # Double-check after acquiring lock
            if game_pk in self._contexts:
                return self._contexts[game_pk]

            loop = asyncio.get_running_loop()
            ctx = await loop.run_in_executor(
                None, SimContext.hydrate, game_pk, game_date, gumbo, self._conn
            )
            self._contexts[game_pk] = ctx
            logger.info("[%d] SimContext hydrated", game_pk)
            return ctx

    def remove(self, game_pk: int):
        """Drop cached context for a finished game."""
        self._contexts.pop(game_pk, None)
