"""MLB extraction + processing + dbt intermediates + feat_vectors."""
import os
from datetime import date, datetime, timedelta, timezone

from dagster import asset, AssetExecutionContext

from ._shared import (
    load_secrets, get_ducklake_connection, run_dbt, handle_schema_drift,
    ensure_extraction_tracking, record_extracted_game_pks,
)
from ._configs import (
    DbtConfig, RawEventsConfig, RawBoxscoresConfig,
    RawGumboEventsConfig, ProcGumboEventsConfig, ProcEventsConfig,
)
from ..dlt.sources.statsapi import boxscores, games, players
from ..dlt.sources.savant import events
from ..dlt.sources.gumbo import events as gumbo_events
from ..dlt.pipeline import run


@asset(
    group_name="mlb",
)
def raw_games(context: AssetExecutionContext):
    """Extract game schedules from MLB Stats API.

    Two-pass extraction:
    1. Forward scan: max(game_date) + 1 → end of current season (new + scheduled games)
    2. Re-fetch: any game where abstract_game_state != 'Final' (unresolved games)

    Proc layer deduplicates via delete+insert on game_pk.
    """
    load_secrets()

    conn = get_ducklake_connection()
    try:
        result = conn.execute(
            "SELECT max(game_date) FROM lakehouse.main.proc_mlb__games"
        ).fetchall()
        max_date = result[0][0] if result and result[0][0] else None

        # Find recent dates with unresolved games (today + last 2 days).
        # Future scheduled games can't have changed state — skip them.
        non_final_rows = conn.execute("""
            SELECT DISTINCT game_date FROM lakehouse.main.proc_mlb__games
            WHERE abstract_game_state IS NOT NULL
            AND abstract_game_state != 'Final'
            AND CAST(game_date AS DATE) <= CURRENT_DATE
            AND CAST(game_date AS DATE) >= CURRENT_DATE - INTERVAL 3 DAY
        """).fetchall()
        refetch_dates = {row[0] for row in non_final_rows}
    except Exception:
        max_date = None
        refetch_dates = set()
    finally:
        conn.close()

    end_date = f"{datetime.now(timezone.utc).year}-11-30"

    # Build set of dates to extract
    extract_dates = set()

    # Pass 1: forward scan (new dates)
    if max_date:
        start_date = (date.fromisoformat(max_date) + timedelta(days=1)).isoformat()
    else:
        start_date = os.environ.get("MLB_START_DATE", "2015-03-28")

    if start_date <= end_date:
        context.log.info(f"Forward scan: {start_date} to {end_date}")
        resource = games(start_date=start_date, end_date=end_date, log=context.log)
        info = run("mlb", "games", resource, log=context.log)
        context.log.info(f"Forward scan done: {info}")
        extract_dates.add("forward")

    # Pass 2: re-fetch unresolved games
    if refetch_dates:
        context.log.info(
            f"Re-fetching {len(refetch_dates)} dates with unresolved games: "
            f"{sorted(refetch_dates)}"
        )
        for d in sorted(refetch_dates):
            resource = games(start_date=d, end_date=d, log=context.log)
            run("mlb", "games", resource, log=context.log)
        extract_dates.add("refetch")

    if not extract_dates:
        context.log.info(f"Already up to date (max_date={max_date})")


@asset(
    group_name="mlb",
    deps=[raw_games],
)
def proc_mlb__games(context: AssetExecutionContext, config: DbtConfig):
    """Process raw games parquet into DuckLake staging table via dbt."""
    load_secrets()
    run_dbt(context, "proc_mlb__games", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[proc_mlb__games],
)
def raw_events(context: AssetExecutionContext, config: RawEventsConfig):
    """Extract play-by-play events for Final games not yet extracted.

    Queries proc_mlb__games for Final games, excludes games already tracked
    in landing.extracted_game_pks. Records successfully extracted game_pks
    after dlt.run() succeeds.
    """
    load_secrets()

    conn = get_ducklake_connection()
    try:
        try:
            ensure_extraction_tracking(conn)
            game_pks = conn.execute("""
                SELECT game_pk FROM lakehouse.main.proc_mlb__games
                WHERE abstract_game_state = 'Final'
                AND status NOT IN ('Postponed', 'Cancelled')
                AND game_pk NOT IN (
                    SELECT game_pk FROM lakehouse.landing.extracted_game_pks
                )
                ORDER BY game_date
            """).fetchall()
        except Exception:
            # Tracking table or proc_mlb__games doesn't exist yet
            game_pks = conn.execute("""
                SELECT game_pk FROM lakehouse.main.proc_mlb__games
                WHERE abstract_game_state = 'Final'
                AND status NOT IN ('Postponed', 'Cancelled')
                ORDER BY game_date
            """).fetchall()
    finally:
        conn.close()

    game_pks = [row[0] for row in game_pks]

    if not game_pks:
        context.log.info("No new games to extract")
        return

    batch = game_pks[:config.batch_size]
    context.log.info(f"Extracting {len(batch)} of {len(game_pks)} games")

    extracted_pks = set()
    resource = events(game_pks=batch, log=context.log, extracted_pks=extracted_pks)
    run("mlb", "events", resource, log=context.log)

    # Track after dlt succeeds — if this fails, games re-extract → QUALIFY handles dupes
    if extracted_pks:
        conn = get_ducklake_connection()
        try:
            record_extracted_game_pks(conn, extracted_pks, log=context.log)
        except Exception as e:
            context.log.warning(f"[tracking] Failed to record: {e}")
        finally:
            conn.close()


@asset(
    group_name="game_context",
    deps=[proc_mlb__games],
)
def raw_boxscores(context: AssetExecutionContext, config: RawBoxscoresConfig):
    """Extract boxscore data (rosters, lineups, SPs) from MLB Stats API.

    Diffs proc_mlb__games against landing.boxscores to find missing game_pks.
    One API call per game. Batched by batch_size per run.
    """
    load_secrets()

    conn = get_ducklake_connection()
    try:
        try:
            game_pks = conn.execute("""
                SELECT g.game_pk FROM lakehouse.main.proc_mlb__games g
                WHERE g.abstract_game_state = 'Final'
                AND g.status NOT IN ('Postponed', 'Cancelled')
                AND g.game_pk NOT IN (
                    SELECT DISTINCT game_pk FROM lakehouse.landing.boxscores
                )
                ORDER BY g.game_date
            """).fetchall()
        except Exception:
            game_pks = conn.execute("""
                SELECT game_pk FROM lakehouse.main.proc_mlb__games
                WHERE abstract_game_state = 'Final'
                AND status NOT IN ('Postponed', 'Cancelled')
                ORDER BY game_date
            """).fetchall()
    finally:
        conn.close()

    game_pks = [row[0] for row in game_pks]

    if not game_pks:
        context.log.info("No new boxscores to fetch")
        return

    batch = game_pks[:config.batch_size]
    context.log.info(f"Fetching {len(batch)} of {len(game_pks)} boxscores")

    resource = boxscores(game_pks=batch, log=context.log)
    run("mlb", "boxscores", resource, log=context.log)


@asset(
    group_name="game_context",
    deps=[raw_boxscores],
)
def proc_mlb__rosters(context: AssetExecutionContext, config: DbtConfig):
    """Process raw boxscores into roster table via dbt."""
    load_secrets()
    handle_schema_drift(context, "proc_mlb__rosters")
    run_dbt(context, "proc_mlb__rosters", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[raw_events],
    pool="s3",
)
def proc_mlb__events(context: AssetExecutionContext, config: ProcEventsConfig):
    """Process raw events from DuckLake landing table via dbt.

    Pre-queries for unprocessed game_pks, batches them, and passes
    to dbt as vars so DuckLake can prune files (1 file per game).
    Triggered by proc_events_sensor, not eager automation.
    """
    load_secrets()
    handle_schema_drift(context, "proc_mlb__events")

    conn = get_ducklake_connection()
    try:
        try:
            game_pks = conn.execute("""
                SELECT game_pk FROM lakehouse.main.proc_mlb__games
                WHERE abstract_game_state = 'Final'
                AND status NOT IN ('Postponed', 'Cancelled')
                AND game_type IN ('R', 'F', 'D', 'L', 'W')
                AND game_pk NOT IN (
                    SELECT DISTINCT game_pk FROM lakehouse.main.proc_mlb__events
                )
                ORDER BY game_date
            """).fetchall()
        except Exception:
            # proc_mlb__events doesn't exist yet — get all playable games
            game_pks = conn.execute("""
                SELECT game_pk FROM lakehouse.main.proc_mlb__games
                WHERE abstract_game_state = 'Final'
                AND status NOT IN ('Postponed', 'Cancelled')
                AND game_type IN ('R', 'F', 'D', 'L', 'W')
                ORDER BY game_date
            """).fetchall()
    finally:
        conn.close()

    game_pks = [row[0] for row in game_pks]

    if not game_pks:
        context.log.info("No unprocessed games — nothing to do")
        return

    context.log.info(f"Processing {len(game_pks)} unprocessed games")

    run_dbt(context, "proc_mlb__events", vars={"batch_game_pks": game_pks}, full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[proc_mlb__games],
)
def raw_gumbo_events(context: AssetExecutionContext, config: RawGumboEventsConfig):
    """Extract play-by-play events from MLB Gumbo for Final games not yet extracted.

    Diffs proc_mlb__games against landing.gumbo_events to find missing game_pks.
    One API call per game. Batched by batch_size per run. Per-game error isolation.
    """
    load_secrets()

    conn = get_ducklake_connection()
    try:
        try:
            game_pks = conn.execute("""
                SELECT g.game_pk FROM lakehouse.main.proc_mlb__games g
                WHERE g.abstract_game_state = 'Final'
                AND g.status NOT IN ('Postponed', 'Cancelled')
                AND g.game_type IN ('R', 'F', 'D', 'L', 'W')
                AND g.game_pk NOT IN (
                    SELECT DISTINCT game_pk FROM lakehouse.landing.gumbo_events
                )
                ORDER BY g.game_date
            """).fetchall()
        except Exception:
            game_pks = conn.execute("""
                SELECT game_pk FROM lakehouse.main.proc_mlb__games
                WHERE abstract_game_state = 'Final'
                AND status NOT IN ('Postponed', 'Cancelled')
                AND game_type IN ('R', 'F', 'D', 'L', 'W')
                ORDER BY game_date
            """).fetchall()
    finally:
        conn.close()

    game_pks = [row[0] for row in game_pks]

    if not game_pks:
        context.log.info("No new games to fetch from Gumbo")
        return

    batch = game_pks[:config.batch_size]
    context.log.info(f"Fetching {len(batch)} of {len(game_pks)} games from Gumbo")

    resource = gumbo_events(game_pks=batch, log=context.log)
    run("mlb", "gumbo_events", resource, log=context.log)


@asset(
    group_name="mlb",
    deps=[raw_gumbo_events],
    pool="s3",
)
def proc_mlb__gumbo_events(context: AssetExecutionContext, config: ProcGumboEventsConfig):
    """Process raw Gumbo events from DuckLake landing table via dbt.

    Pre-queries for unprocessed game_pks (games landed but not yet proc'd),
    caps to batch_size to stay under ARG_MAX on the dbt --vars arg during
    fresh backfill, and passes the list so DuckLake can prune files.
    """
    load_secrets()
    handle_schema_drift(context, "proc_mlb__gumbo_events")

    conn = get_ducklake_connection()
    try:
        try:
            game_pks = conn.execute("""
                SELECT DISTINCT l.game_pk
                FROM lakehouse.landing.gumbo_events l
                WHERE l.game_pk NOT IN (
                    SELECT DISTINCT game_pk FROM lakehouse.main.proc_mlb__gumbo_events
                )
                ORDER BY l.game_pk
            """).fetchall()
        except Exception:
            # proc_mlb__gumbo_events doesn't exist yet — process everything landed
            game_pks = conn.execute(
                "SELECT DISTINCT game_pk FROM lakehouse.landing.gumbo_events ORDER BY game_pk"
            ).fetchall()
    finally:
        conn.close()

    game_pks = [row[0] for row in game_pks]

    if not game_pks:
        context.log.info("No unprocessed gumbo games — nothing to do")
        return

    batch = game_pks[:config.batch_size]
    context.log.info(f"Processing {len(batch)} of {len(game_pks)} unprocessed gumbo games")

    run_dbt(context, "proc_mlb__gumbo_events", vars={"batch_game_pks": batch}, full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[proc_mlb__events],
    pool="s3",
)
def int_mlb__batter_counts(context: AssetExecutionContext, config: DbtConfig):
    """Incremental daily batter counting stats split by pitcher handedness."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__batter_counts")
    run_dbt(context, "int_mlb__batter_counts", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[proc_mlb__events],
    pool="s3",
)
def int_mlb__pitcher_counts(context: AssetExecutionContext, config: DbtConfig):
    """Incremental daily pitcher counting stats split by batter handedness."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__pitcher_counts")
    run_dbt(context, "int_mlb__pitcher_counts", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[int_mlb__batter_counts],
    pool="s3",
)
def int_mlb__batters(context: AssetExecutionContext, config: DbtConfig):
    """Cumulative batter stats with season + career rate stats."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__batters")
    run_dbt(context, "int_mlb__batters", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[int_mlb__pitcher_counts],
    pool="s3",
)
def int_mlb__pitchers(context: AssetExecutionContext, config: DbtConfig):
    """Cumulative pitcher stats with season + career rate stats."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__pitchers")
    run_dbt(context, "int_mlb__pitchers", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[proc_mlb__events],
    pool="s3",
)
def int_mlb__pitcher_arsenal_counts(context: AssetExecutionContext, config: DbtConfig):
    """Incremental daily pitcher arsenal aggregations (all pitches, not just terminal)."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__pitcher_arsenal_counts")
    run_dbt(context, "int_mlb__pitcher_arsenal_counts", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[int_mlb__pitcher_arsenal_counts],
    pool="s3",
)
def int_mlb__pitcher_arsenal(context: AssetExecutionContext, config: DbtConfig):
    """Cumulative pitcher arsenal stats: pitch mix, velocity, spin, delivery."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__pitcher_arsenal")
    run_dbt(context, "int_mlb__pitcher_arsenal", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[proc_mlb__events],
    pool="s3",
)
def int_mlb__batter_arsenal_counts(context: AssetExecutionContext, config: DbtConfig):
    """Incremental daily batter wOBA by pitch category x pitcher handedness."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__batter_arsenal_counts")
    run_dbt(context, "int_mlb__batter_arsenal_counts", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[int_mlb__batter_arsenal_counts],
    pool="s3",
)
def int_mlb__batter_arsenal(context: AssetExecutionContext, config: DbtConfig):
    """Cumulative batter arsenal stats: wOBA by pitch type, season + career."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__batter_arsenal")
    run_dbt(context, "int_mlb__batter_arsenal", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[proc_mlb__events],
    pool="s3",
)
def int_mlb__batter_discipline_counts(context: AssetExecutionContext, config: DbtConfig):
    """Incremental daily batter plate discipline counts by pitch category x pitcher handedness."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__batter_discipline_counts")
    run_dbt(context, "int_mlb__batter_discipline_counts", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[int_mlb__batter_discipline_counts],
    pool="s3",
)
def int_mlb__batter_discipline(context: AssetExecutionContext, config: DbtConfig):
    """Cumulative batter discipline stats: chase + whiff rates by pitch type, season + career."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__batter_discipline")
    run_dbt(context, "int_mlb__batter_discipline", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[int_mlb__batters, int_mlb__batter_arsenal, int_mlb__batter_discipline],
    pool="s3",
)
def int_mlb__batter_profile(context: AssetExecutionContext, config: DbtConfig):
    """Wide batter profile: batters + arsenal + discipline joined."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__batter_profile")
    run_dbt(context, "int_mlb__batter_profile", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[int_mlb__pitchers, int_mlb__pitcher_arsenal],
    pool="s3",
)
def int_mlb__pitcher_profile(context: AssetExecutionContext, config: DbtConfig):
    """Wide pitcher profile: pitchers + arsenal joined."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__pitcher_profile")
    run_dbt(context, "int_mlb__pitcher_profile", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    deps=[proc_mlb__events],
    pool="s3",
)
def int_mlb__game_state(context: AssetExecutionContext, config: DbtConfig):
    """In-game state features: game state, outing counters, context. Mirrors sim engine state tracking."""
    load_secrets()
    handle_schema_drift(context, "int_mlb__game_state")
    run_dbt(context, "int_mlb__game_state", full_refresh=config.full_refresh)


@asset(
    group_name="analysis",
    deps=[int_mlb__game_state, proc_mlb__gumbo_events, proc_mlb__games],
    pool="s3",
)
def ana_mlb__event_state(context: AssetExecutionContext, config: DbtConfig):
    """Per-pitch state enriched with gumbo validity-window timestamps.

    Consumer-facing analysis table. Joins int_mlb__game_state × proc_mlb__gumbo_events
    (pitch rows) on (game_pk, at_bat_number, pitch_number). Anchor timestamp is
    prev_event_end_ts — when this row's pre-state became valid.
    """
    load_secrets()
    handle_schema_drift(context, "ana_mlb__event_state")
    run_dbt(context, "ana_mlb__event_state", full_refresh=config.full_refresh)


@asset(
    group_name="mlb",
    pool="s3",
    deps=[
        int_mlb__game_state,
        int_mlb__batter_profile,
        int_mlb__pitcher_profile,
    ],
)
def feat_mlb__vectors(context: AssetExecutionContext, config: DbtConfig):
    """Training vectors with ASOF-joined batter/pitcher/arsenal profiles."""
    load_secrets()
    handle_schema_drift(context, "feat_mlb__vectors")
    run_dbt(context, "feat_mlb__vectors", full_refresh=config.full_refresh)


@asset(
    group_name="game_context",
    deps=[proc_mlb__events],
)
def raw_players(context: AssetExecutionContext):
    """Fetch player details for any new player IDs in events.

    Diffs distinct player IDs from proc_mlb__events against ref_mlb__players.
    Fetches only new IDs from MLB Stats API bulk endpoint.
    """
    load_secrets()

    conn = get_ducklake_connection()
    try:
        # All distinct player IDs from events
        all_ids = conn.execute("""
            SELECT DISTINCT batter_id AS player_id FROM lakehouse.main.proc_mlb__events
            UNION
            SELECT DISTINCT pitcher_id FROM lakehouse.main.proc_mlb__events
        """).fetchall()
        all_ids = {row[0] for row in all_ids}

        # Existing IDs in ref table
        try:
            existing = conn.execute(
                "SELECT player_id FROM lakehouse.main.ref_mlb__players"
            ).fetchall()
            existing_ids = {row[0] for row in existing}
        except Exception:
            existing_ids = set()
    finally:
        conn.close()

    new_ids = sorted(all_ids - existing_ids)

    if not new_ids:
        context.log.info("No new player IDs to fetch")
        return

    context.log.info(f"Fetching {len(new_ids)} new players (of {len(all_ids)} total)")
    resource = players(player_ids=new_ids, log=context.log)
    run("mlb", "players", resource, log=context.log)


@asset(
    group_name="game_context",
    deps=[raw_players],
)
def ref_mlb__players(context: AssetExecutionContext, config: DbtConfig):
    """Build player reference table from raw player parquet via dbt."""
    load_secrets()
    run_dbt(context, "ref_mlb__players", full_refresh=config.full_refresh)
