"""Baseball Savant dlt resources."""
import dlt


@dlt.resource(write_disposition="append")
def events(game_pks: list[int], log=None, extracted_pks: set | None = None):
    """Fetch pitch-level Statcast data for a batch of games from Baseball Savant.

    Args:
        game_pks: List of MLB game IDs
        log: Optional logger
        extracted_pks: Optional set — only game_pks that returned data are added.
            Games where Savant returns None are skipped and retried next run.

    Yields:
        Flat pitch-level dicts per game (~104 columns per pitch).
        Multiple yields are collected into a single parquet file by dlt.
        Per-game errors are caught and logged without killing the batch.
    """
    from pybaseball import statcast_single_game

    for game_pk in game_pks:
        if log:
            log.info(f"[events] Fetching Savant data for game {game_pk}")
        try:
            df = statcast_single_game(game_pk)
            if df is None:
                if log:
                    log.info(f"[events] No Savant data for game {game_pk}, will retry next run")
                continue
            if log:
                log.info(f"[events] Game {game_pk}: {len(df)} pitches")
            records = df.where(df.notna(), None).to_dict(orient="records")
            yield records
            if extracted_pks is not None:
                extracted_pks.add(game_pk)
        except Exception as e:
            if log:
                log.warning(f"[events] Game {game_pk} failed, skipping: {e}")
