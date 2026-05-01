"""In-memory game state snapshot backed by Redis for persistence.

Game data flows through here: schedule fields from SQLite, live state
from trackers, sim results from the dispatcher. The SSE poller and
REST endpoints read exclusively from the snapshot.

Write-through pattern: writes update local dicts AND Redis.
Reads come from local dicts (zero-latency). On startup, local dicts
are hydrated from Redis so state survives pod restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Schedule fields (from SQLite / warehouse)
_SCHEDULE_KEYS = (
    "game_pk",
    "game_date",
    "game_datetime",
    "game_type",
    "status",
    "abstract_game_state",
    "away_team_id",
    "away_team_name",
    "home_team_id",
    "home_team_name",
    "venue_name",
    "last_synced_at",
)

# Live fields (from tracker)
_LIVE_KEYS = (
    "away_score",
    "home_score",
    "inning",
    "inning_half",
    "outs",
    "balls",
    "strikes",
    "runners",
    "current_batter_id",
    "current_batter_name",
    "current_pitcher_id",
    "current_pitcher_name",
    "last_play",
    "live_updated_at",
)

# Sim fields (from dispatcher)
_SIM_KEYS = (
    "sim_p_home_win",
    "sim_p_home_win_se",
    "sim_mean_home_score",
    "sim_mean_away_score",
    "sim_n_sims",
    "sim_we_baseline",
    "sim_updated_at",
    "sim_duration_ms",
)

_ALL_KEYS = _SCHEDULE_KEYS + _LIVE_KEYS + _SIM_KEYS

_HASH_KEYS = frozenset(_SCHEDULE_KEYS + _LIVE_KEYS + _SIM_KEYS)

_INT_FIELDS = frozenset({
    "game_pk", "away_team_id", "home_team_id", "away_score", "home_score",
    "inning", "outs", "balls", "strikes", "current_batter_id",
    "current_pitcher_id", "sim_n_sims",
})
_FLOAT_FIELDS = frozenset({
    "sim_p_home_win", "sim_p_home_win_se", "sim_mean_home_score",
    "sim_mean_away_score", "sim_we_baseline", "sim_duration_ms",
})


def _to_redis(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def _from_redis(key: str, value: str):
    if not value:
        return None
    if key in _INT_FIELDS:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if key in _FLOAT_FIELDS:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value


class GameSnapshot:
    """Central in-memory store for all game state, backed by Redis.

    Pure data store — no change detection logic. Consumers (like the
    SSE poller) own their own diffing.
    """

    def __init__(self, redis_client: aioredis.Redis | None = None):
        self._games: dict[int, dict] = {}
        self._gumbo: dict[int, dict] = {}
        self._game_state_history: dict[int, list[dict]] = {}
        self._sim_results: dict[int, list[dict]] = {}
        self._change_event: asyncio.Event = asyncio.Event()
        self._redis: aioredis.Redis | None = redis_client

    def _notify(self) -> None:
        old_event = self._change_event
        self._change_event = asyncio.Event()
        old_event.set()
        if self._redis is not None:
            asyncio.ensure_future(self._publish_change())

    async def _publish_change(self) -> None:
        try:
            await self._redis.publish("snapshot:changes", "update")
        except Exception:
            pass  # non-critical, local event already fired

    async def wait_for_change(self) -> None:
        await self._change_event.wait()

    async def run_result_listener(self) -> None:
        """Subscribe to sim:results Pub/Sub and write results to snapshot."""
        if self._redis is None:
            logger.info("No Redis — sim result listener disabled")
            return

        from data_feed.sim.redis_queue import SIM_RESULTS_CHANNEL

        while True:
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(SIM_RESULTS_CHANNEL)
                logger.info("Sim result listener subscribed to %s", SIM_RESULTS_CHANNEL)

                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    try:
                        data = json.loads(message["data"])
                        game_pk = data.pop("game_pk")
                        self.update_sim(game_pk, data)
                    except Exception:
                        logger.warning("Failed to process sim result from Pub/Sub", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Result listener disconnected, reconnecting in 5s", exc_info=True)
                await asyncio.sleep(5)

    # --- Redis write helpers (fire-and-forget, never crash) ---

    async def _r_hset(self, key: str, mapping: dict) -> None:
        if self._redis is None:
            return
        try:
            clean = {k: _to_redis(v) for k, v in mapping.items() if v is not None}
            if clean:
                await self._redis.hset(key, mapping=clean)
        except Exception:
            logger.warning("Redis hset failed for %s", key, exc_info=True)

    async def _r_set(self, key: str, value: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(key, value)
        except Exception:
            logger.warning("Redis set failed for %s", key, exc_info=True)

    async def _r_delete(self, *keys: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.delete(*keys)
        except Exception:
            logger.warning("Redis delete failed for %s", keys, exc_info=True)

    async def _r_sadd(self, key: str, *members: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.sadd(key, *members)
        except Exception:
            logger.warning("Redis sadd failed for %s", key, exc_info=True)

    async def _r_srem(self, key: str, *members: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.srem(key, *members)
        except Exception:
            logger.warning("Redis srem failed for %s", key, exc_info=True)

    # --- Startup hydration ---

    async def hydrate_from_redis(self) -> None:
        if self._redis is None:
            logger.info("No Redis client — skipping hydration")
            return

        try:
            pks = await self._redis.smembers("game_pks")
        except Exception:
            logger.warning("Redis hydration failed — starting empty", exc_info=True)
            return

        if not pks:
            logger.info("Redis has no game data — starting fresh")
            return

        now = datetime.now(timezone.utc)
        stale_cutoff = 12 * 3600  # 12 hours

        hydrated = 0
        pruned = 0
        for pk_str in pks:
            pk = int(pk_str)
            try:
                raw = await self._redis.hgetall(f"game:{pk}")
                if not raw:
                    await self._r_srem("game_pks", str(pk))
                    pruned += 1
                    continue

                game = {k: _from_redis(k, v) for k, v in raw.items()}
                game["game_pk"] = pk

                # Prune stale Final games
                state = game.get("abstract_game_state")
                if state in ("Final", "Completed Early") and game.get("game_datetime"):
                    try:
                        dt = datetime.fromisoformat(
                            str(game["game_datetime"]).replace("Z", "+00:00")
                        )
                        if (now - dt).total_seconds() > stale_cutoff:
                            await self._r_delete(
                                f"game:{pk}",
                                f"gumbo:{pk}",
                                f"gamestatehist:{pk}",
                                f"simresults:{pk}",
                            )
                            await self._r_srem("game_pks", str(pk))
                            pruned += 1
                            continue
                    except (ValueError, TypeError):
                        pass

                for k in _ALL_KEYS:
                    game.setdefault(k, None)

                self._games[pk] = game

                gse_raw = await self._redis.lrange(f"gamestatehist:{pk}", 0, -1)
                if gse_raw:
                    self._game_state_history[pk] = [json.loads(h) for h in gse_raw]

                sr_raw = await self._redis.lrange(f"simresults:{pk}", 0, -1)
                if sr_raw:
                    self._sim_results[pk] = [json.loads(h) for h in sr_raw]

                hydrated += 1
            except Exception:
                logger.warning("Failed to hydrate game %d from Redis", pk, exc_info=True)

        logger.info("Hydrated %d games from Redis (pruned %d stale)", hydrated, pruned)

    # --- Schedule ---

    def merge_schedule(self, games: list[dict]) -> None:
        """Merge schedule rows from SQLite. Preserves live + sim fields."""
        for g in games:
            pk = g["game_pk"]
            if pk in self._games:
                has_live = self._games[pk].get("live_updated_at") is not None
                for key in _SCHEDULE_KEYS:
                    if key in g:
                        if has_live and key in ("status", "abstract_game_state"):
                            continue
                        self._games[pk][key] = g[key]
            else:
                self._games[pk] = {k: g.get(k) for k in _ALL_KEYS}

        if self._redis is not None:
            asyncio.ensure_future(self._merge_schedule_redis(games))
        self._notify()

    async def _merge_schedule_redis(self, games: list[dict]) -> None:
        pks = []
        for g in games:
            pk = g["game_pk"]
            pks.append(str(pk))
            fields = {k: g[k] for k in _SCHEDULE_KEYS if k in g and g[k] is not None}
            await self._r_hset(f"game:{pk}", fields)
        if pks:
            await self._r_sadd("game_pks", *pks)

    # --- Live state ---

    def update_live(self, game_pk: int, flat_state: dict, gumbo: dict) -> None:
        game = self._games.get(game_pk)
        if game is None:
            logger.debug("update_live: game_pk %d not in snapshot — skipping", game_pk)
            return

        now = datetime.now(timezone.utc).isoformat()
        for k, v in flat_state.items():
            game[k] = v
        game["live_updated_at"] = now
        self._gumbo[game_pk] = gumbo

        if self._redis is not None:
            redis_fields = {k: v for k, v in flat_state.items() if k in _HASH_KEYS and v is not None}
            redis_fields["live_updated_at"] = now
            asyncio.ensure_future(self._update_live_redis(game_pk, redis_fields, gumbo))
        self._notify()

    async def _update_live_redis(self, game_pk: int, fields: dict, gumbo: dict) -> None:
        await self._r_hset(f"game:{game_pk}", fields)
        await self._r_set(f"gumbo:{game_pk}", json.dumps(gumbo))

    # --- Sim ---

    def update_sim(self, game_pk: int, result: dict) -> None:
        game = self._games.get(game_pk)
        if game is None:
            logger.debug("update_sim: game_pk %d not in snapshot — skipping", game_pk)
            return

        now = datetime.now(timezone.utc).isoformat()
        game["sim_p_home_win"] = result["p_home_win"]
        game["sim_p_home_win_se"] = result["p_home_win_se"]
        game["sim_mean_home_score"] = result["mean_home_score"]
        game["sim_mean_away_score"] = result["mean_away_score"]
        game["sim_n_sims"] = result["n_sims"]
        game["sim_we_baseline"] = result.get("we_baseline")
        game["sim_duration_ms"] = result.get("duration_ms")
        game["sim_updated_at"] = now

        from data_feed import sim_recorder
        sim_recorder.record_result(self, game_pk, result)

        if self._redis is not None:
            redis_fields = {k: game[k] for k in _SIM_KEYS if game.get(k) is not None}
            asyncio.ensure_future(self._r_hset(f"game:{game_pk}", redis_fields))
        self._notify()

    # --- Reads (from local cache, not Redis) ---

    def get_all_games(self) -> list[dict]:
        return sorted(
            self._games.values(),
            key=lambda g: g.get("game_datetime") or "",
        )

    def get_game(self, game_pk: int) -> dict | None:
        return self._games.get(game_pk)

    def get_gumbo(self, game_pk: int) -> dict | None:
        return self._gumbo.get(game_pk)

    def get_game_state_events(self, game_pk: int) -> list[dict]:
        return self._game_state_history.get(game_pk, [])

    def get_sim_results(self, game_pk: int) -> list[dict]:
        return self._sim_results.get(game_pk, [])

    def game_count(self) -> int:
        return len(self._games)

    def live_count(self) -> int:
        return sum(
            1
            for g in self._games.values()
            if g.get("abstract_game_state") == "Live"
        )

    # --- Cleanup ---

    def on_game_final(self, game_pk: int) -> None:
        """Clean up all state when game ends."""
        from data_feed import sim_recorder, game_state_recorder
        sim_recorder.record_final(self, game_pk)
        game_state_recorder.record_final(self, game_pk)

        self._game_state_history.pop(game_pk, None)
        self._sim_results.pop(game_pk, None)
        self._gumbo.pop(game_pk, None)
        self._games.pop(game_pk, None)

        if self._redis is not None:
            asyncio.ensure_future(self._on_final_redis(game_pk))

    async def _on_final_redis(self, game_pk: int) -> None:
        await self._r_delete(
            f"gumbo:{game_pk}",
            f"gamestatehist:{game_pk}",
            f"simresults:{game_pk}",
            f"game:{game_pk}",
        )
        await self._r_srem("game_pks", str(game_pk))
