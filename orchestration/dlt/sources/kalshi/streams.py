"""Kalshi dlt resources.

Landing holds what Kalshi says about a market, projected to the fields anything
downstream actually reads. Parsing, typing beyond the declared schema, and the
join to a real game are all dbt's job — not landing's.
"""

import os
import time

import dlt

from .client import KalshiClient

# Kalshi's MLB regular-season game series. Spring training is KXMLBSTGAME.
MLB_SERIES = "KXMLBGAME"

# Fields kept from the market payload. Kalshi returns more — subtitles, two
# extra expiration timestamps, can_close_early, volume, volume_24h — and none of
# it is read by anything. Widening this later is one entry here plus one in
# MARKETS_COLUMNS.
_KEEP = (
    "ticker",
    "event_ticker",
    "market_type",
    "title",
    "status",
    "result",
    "created_time",
    "updated_time",
    "open_time",
    "close_time",
    "settlement_ts",
)

# Declared landing schema. Types are declared rather than inferred: inference
# reads whatever happened to be in the pull window, which is how the v1 warehouse
# ended up with `volume_fp` instead of `volume` — dlt suffixes a column when it
# sees mixed int/float. A declared type also means a window with no settled
# markets in it still lands `result` and `settlement_ts` as real columns.
MARKETS_COLUMNS = {
    "ticker": {
        "data_type": "text",
        "nullable": False,
        "description": "Kalshi's id for one market, and the primary key. Encodes the "
        "series, the game date, both team abbreviations concatenated "
        "with no delimiter, and which team the market pays out on — "
        "e.g. KXMLBGAME-26AUG06HOUSEA-HOU is the Houston side of "
        "Houston at Seattle on 2026-08-06. Every game has two markets, "
        "one per team.",
    },
    "event_ticker": {
        "data_type": "text",
        "nullable": True,
        "description": "The Kalshi event both of a game's markets belong to. Same "
        "series/date/teams prefix as the ticker, without the trailing "
        "team. Two tickers share one event_ticker.",
    },
    "market_type": {
        "data_type": "text",
        "nullable": True,
        "description": "Kalshi's contract type for the market. Binary for game "
        "winner markets.",
    },
    "title": {
        "data_type": "text",
        "nullable": True,
        "description": "Human-readable market question as Kalshi words it. For "
        "reading in a query result, not for parsing — the ticker is "
        "the machine-readable form.",
    },
    "status": {
        "data_type": "text",
        "nullable": True,
        "description": "Where the market is in its life: unopened, open, closed, "
        "settled. The recorder subscribes to open markets. This is "
        "the field that moves between pulls.",
    },
    "result": {
        "data_type": "text",
        "nullable": True,
        "description": "How the market resolved — yes or no on this ticker's team "
        "winning. Empty until the market settles.",
    },
    "created_time": {
        "data_type": "timestamp",
        "nullable": True,
        "description": "When Kalshi created the market. Well before trading opens; "
        "for when trading opens use open_time.",
    },
    "updated_time": {
        "data_type": "timestamp",
        "nullable": True,
        "description": "When Kalshi last changed the market. Bumps on a status "
        "transition or on settlement.",
    },
    "open_time": {
        "data_type": "timestamp",
        "nullable": True,
        "description": "When trading opens. Not first pitch — Kalshi opens MLB game "
        "markets well ahead of the game, which is what gives the "
        "hourly pull its margin.",
    },
    "close_time": {
        "data_type": "timestamp",
        "nullable": True,
        "description": "When trading closes, which for a game market is around the "
        "final out. The closest thing in this table to 'when is this "
        "game', and what the recorder filters on to decide what to "
        "subscribe to tonight.",
    },
    "settlement_ts": {
        "data_type": "timestamp",
        "nullable": True,
        "description": "When Kalshi settled the market and paid out. Null until "
        "settlement; later than close_time.",
    },
}


def _project(market: dict) -> dict:
    """One API market → one landing row, narrowed to _KEEP."""
    return {k: market.get(k) for k in _KEEP}


@dlt.resource(
    name="markets",
    write_disposition="merge",
    primary_key="ticker",
    columns=MARKETS_COLUMNS,
)
def markets(series_ticker: str = MLB_SERIES, days_back: int = 7, log=None, client=None):
    """MLB game markets for a Kalshi series, merged on ticker.

    Merge rather than append: a market row is the market's current state, and
    the only things that move over its life are status, result, settlement_ts
    and the updated_time that tracks them. Keeping old copies of that would buy
    a status-transition history nothing asks for.

    No status filter — an unopened market matters as much as an open one, since
    knowing about tonight's game before it opens is the entire point of running
    this ahead of the recorder.

    days_back windows on close_time rather than pulling the whole series, since
    the series reaches back years. Seven days covers rescheduled and suspended
    games drifting forward, and at roughly thirty markets a day it is cheap
    enough that there is no reason to tighten it. 0 disables the filter.

    client is anything with KalshiClient's `get`. None builds a KalshiClient
    from the environment, so only that path needs the Kalshi credentials.
    """
    if client is None:
        client = KalshiClient(
            key_id=os.environ["KALSHI_KEY_ID"],
            private_key_pem=os.environ["KALSHI_PRIVATE_KEY"],
        )

    min_close_ts = None
    if days_back > 0:
        min_close_ts = int(time.time()) - (days_back * 86400)

    cursor = None
    total = 0

    while True:
        params = {"series_ticker": series_ticker, "limit": 1000}
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if cursor:
            params["cursor"] = cursor

        response = client.get("/markets", params=params, log=log)

        batch = response.get("markets", [])
        if batch:
            total += len(batch)
            yield [_project(m) for m in batch]

        cursor = response.get("cursor", "")
        if not cursor:
            break

    if log:
        log.info(f"[kalshi] markets: {total} rows")


# dlt has no `description=` on @dlt.resource; additional_table_hints is where a
# table-level hint goes, and it rides into the persisted schema from there.
markets.apply_hints(
    additional_table_hints={
        "description": (
            "Every Kalshi MLB game market Kalshi knows about within the pull window, one row "
            "per ticker, current state. Two rows per game — one market per team. The ticker "
            "string carries the game date and both teams, so a market can be matched to a real "
            "game entirely from this table plus a team abbreviation map; there is no game_pk "
            "here and matching to MLB happens downstream. This is the roster the order book "
            "recorder reads to decide what to subscribe to."
        )
    }
)
