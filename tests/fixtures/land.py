"""Lands the fixture markets in kalshi.markets, so dbt has its source table.

The real resource and the real pipeline make the table; only the client is
replaced, so the landing schema has one definition and no Kalshi credential is
read.
"""

import json
from pathlib import Path

from orchestration.dlt.pipeline import source_pipeline
from orchestration.dlt.sources.kalshi.streams import markets

FIXTURE = Path(__file__).with_name("kalshi_markets.json")


class FixtureClient:
    """Answers `get` from the fixture file and never opens a connection."""

    def get(self, endpoint: str, params: dict | None = None, log=None) -> dict:
        return json.loads(FIXTURE.read_text())


if __name__ == "__main__":
    load = source_pipeline("kalshi").run(markets(client=FixtureClient()))
    load.raise_on_failed_jobs()
    print(load)
