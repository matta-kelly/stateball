"""Maintenance assets (manual trigger only)."""
import time

import duckdb
from dagster import asset, AssetExecutionContext

from ._shared import load_secrets, get_ducklake_connection


# Compaction config
TARGET_FILE_SIZE = "256MB"                     # catalog-wide target (was default 512MB)
MERGE_MAX_FILE_SIZE_BYTES = 64 * 1024 * 1024   # 64MB — only merge files below this
MIN_FILE_COUNT = 50                            # skip tables with fewer files than this
AVG_SIZE_CEILING_BYTES = 16 * 1024 * 1024      # skip tables whose avg file >= 16MB
COMPACT_BATCH_SIZE = 10                        # files merged per proc call (memory cap)
MAX_BATCHES_PER_TABLE = 500                    # safety valve vs infinite loops
DUCKLAKE_MAX_RETRY = 100                       # DuckLake's own commit-retry budget (default 10)


def _file_count(conn: duckdb.DuckDBPyConnection, table: str, schema: str) -> int:
    row = conn.execute(
        "SELECT count(*) "
        "FROM __ducklake_metadata_lakehouse.lakehouse.ducklake_data_file df "
        "JOIN __ducklake_metadata_lakehouse.lakehouse.ducklake_table tbl "
        "  ON df.table_id = tbl.table_id "
        "JOIN __ducklake_metadata_lakehouse.lakehouse.ducklake_schema sch "
        "  ON tbl.schema_id = sch.schema_id "
        "WHERE df.end_snapshot IS NULL "
        "  AND tbl.end_snapshot IS NULL "
        "  AND tbl.table_name = ? AND sch.schema_name = ?",
        [table, schema],
    ).fetchone()
    return row[0] if row else 0


def _discover_fragmented_tables(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str, int, int]]:
    """Return (schema, table, n_files, total_bytes) for tables needing compaction."""
    rows = conn.execute(
        "SELECT sch.schema_name, tbl.table_name, "
        "       count(*) AS n_files, "
        "       sum(df.file_size_bytes) AS total_bytes "
        "FROM __ducklake_metadata_lakehouse.lakehouse.ducklake_data_file df "
        "JOIN __ducklake_metadata_lakehouse.lakehouse.ducklake_table tbl "
        "  ON df.table_id = tbl.table_id "
        "JOIN __ducklake_metadata_lakehouse.lakehouse.ducklake_schema sch "
        "  ON tbl.schema_id = sch.schema_id "
        "WHERE df.end_snapshot IS NULL "
        "  AND tbl.end_snapshot IS NULL "
        "GROUP BY sch.schema_name, tbl.table_name "
        f"HAVING count(*) >= {MIN_FILE_COUNT} "
        f"   AND avg(df.file_size_bytes) < {AVG_SIZE_CEILING_BYTES} "
        "ORDER BY count(*) DESC"
    ).fetchall()
    return [(r[0], r[1], r[2], int(r[3] or 0)) for r in rows]


def _configure_session(conn: duckdb.DuckDBPyConnection) -> None:
    """Session settings for robust compaction — retry budget + target size."""
    conn.execute(f"SET ducklake_max_retry_count = {DUCKLAKE_MAX_RETRY}")
    conn.execute(f"CALL lakehouse.set_option('target_file_size', '{TARGET_FILE_SIZE}')")


def _compact_one_table(
    context: AssetExecutionContext,
    schema: str,
    table: str,
    files_before: int,
) -> dict:
    """Compact a single table with a FRESH connection. Per-batch error isolation.

    Returns a dict summary: files_before, files_after, batches_ok, batches_failed, elapsed, status.
    Never raises — errors recorded in the summary.
    """
    t_start = time.monotonic()
    summary = {
        "schema": schema, "table": table,
        "files_before": files_before, "files_after": files_before,
        "batches_ok": 0, "batches_failed": 0,
        "elapsed": 0.0, "status": "unknown",
    }

    conn = None
    try:
        conn = get_ducklake_connection()
        _configure_session(conn)
    except Exception:
        context.log.exception(f"[{schema}.{table}] connection setup failed — skipping")
        summary["status"] = "connect_failed"
        summary["elapsed"] = time.monotonic() - t_start
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return summary

    try:
        prev = files_before
        for batch_num in range(1, MAX_BATCHES_PER_TABLE + 1):
            # One batch = one CALL + one CHECKPOINT. Error in either isolates to this batch.
            try:
                t0 = time.monotonic()
                conn.execute(
                    "CALL ducklake_merge_adjacent_files("
                    "  'lakehouse', ?, "
                    "  schema => ?, "
                    "  max_compacted_files => ?, "
                    "  max_file_size => ?"
                    ")",
                    [table, schema, COMPACT_BATCH_SIZE, MERGE_MAX_FILE_SIZE_BYTES],
                )
                conn.execute("CHECKPOINT")
                elapsed_batch = time.monotonic() - t0
                summary["batches_ok"] += 1
            except Exception as e:
                summary["batches_failed"] += 1
                context.log.warning(
                    f"[{schema}.{table}] batch {batch_num} failed: {type(e).__name__}: {e}"
                )
                # Try to recover the connection — one bad batch shouldn't poison the rest
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    conn = get_ducklake_connection()
                    _configure_session(conn)
                except Exception:
                    context.log.exception(
                        f"[{schema}.{table}] reconnect failed — giving up on this table"
                    )
                    summary["status"] = "conn_lost"
                    break
                # If too many consecutive batch failures, stop
                if summary["batches_failed"] >= 3 and summary["batches_ok"] == 0:
                    context.log.warning(
                        f"[{schema}.{table}] 3 consecutive failures, no progress — stopping"
                    )
                    summary["status"] = "early_give_up"
                    break
                continue

            # Progress check
            try:
                curr = _file_count(conn, table, schema)
            except Exception:
                context.log.exception(f"[{schema}.{table}] file count check failed")
                summary["status"] = "count_failed"
                break

            merged = prev - curr
            context.log.info(
                f"[{schema}.{table}] batch {batch_num}: {prev} → {curr} files "
                f"(merged {merged}, {elapsed_batch:.1f}s)"
            )

            if curr >= prev:
                # No progress — we're done (or stuck at a floor)
                summary["status"] = "done"
                break
            prev = curr
        else:
            # Hit MAX_BATCHES_PER_TABLE without finishing
            context.log.warning(
                f"[{schema}.{table}] hit max batches ({MAX_BATCHES_PER_TABLE}) — stopping"
            )
            summary["status"] = "max_batches"

        try:
            summary["files_after"] = _file_count(conn, table, schema)
        except Exception:
            pass
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    summary["elapsed"] = time.monotonic() - t_start
    return summary


@asset(group_name="maintenance", pool="s3")
def cleanup_dbt_orphans(context: AssetExecutionContext):
    """Drop orphaned __dbt_tmp tables left by failed dbt runs."""
    load_secrets()
    conn = get_ducklake_connection()
    try:
        tmp_tables = conn.execute(
            "SELECT table_name FROM __ducklake_metadata_lakehouse.information_schema.tables "
            "WHERE table_schema = 'main' AND table_name LIKE '%__dbt_tmp'"
        ).fetchall()
        if tmp_tables:
            for (table_name,) in tmp_tables:
                context.log.info(f"Dropping orphan table: {table_name}")
                conn.execute(f'DROP TABLE IF EXISTS lakehouse.main."{table_name}"')
            context.log.info(f"Dropped {len(tmp_tables)} orphan __dbt_tmp tables")
        else:
            context.log.info("No orphan __dbt_tmp tables found")
    finally:
        conn.close()


@asset(group_name="maintenance", pool="s3")
def compact_ducklake(context: AssetExecutionContext):
    """Merge small parquet files across fragmented DuckLake tables.

    Resilience design:
    - One FRESH connection per table — no cross-table session state
    - Each batch = one CALL + one CHECKPOINT — progress persists even if a later batch fails
    - Per-table try/except — one failing table doesn't kill the whole asset
    - Per-batch try/except with reconnect — one bad batch reconnects and retries the next
    - DuckLake retry budget raised to 100 per session (from default 10)
    - Auto-discover fragmented tables — no hardcoded list
    """
    load_secrets()

    t_total = time.monotonic()

    # Discovery on its own connection
    discovery_conn = get_ducklake_connection()
    try:
        tables = _discover_fragmented_tables(discovery_conn)
    finally:
        discovery_conn.close()

    if not tables:
        context.log.info("No fragmented tables found — nothing to compact")
        return

    context.log.info(f"Discovered {len(tables)} fragmented tables:")
    for schema, table, n_files, total_bytes in tables:
        context.log.info(
            f"  {schema}.{table}: {n_files} files, "
            f"{total_bytes / 1024 / 1024:.1f} MB, "
            f"avg {total_bytes / max(n_files, 1) / 1024:.1f} KB/file"
        )

    # Compact each table in isolation
    summaries = []
    for schema, table, files_before, _ in tables:
        context.log.info(f"--- Compacting {schema}.{table} ({files_before} files) ---")
        summary = _compact_one_table(context, schema, table, files_before)
        summaries.append(summary)
        context.log.info(
            f"[{schema}.{table}] done: {summary['files_before']} → {summary['files_after']} "
            f"files, status={summary['status']}, "
            f"batches ok/failed={summary['batches_ok']}/{summary['batches_failed']}, "
            f"elapsed={summary['elapsed']:.1f}s"
        )

    # Final run summary
    elapsed_total = time.monotonic() - t_total
    total_before = sum(s["files_before"] for s in summaries)
    total_after = sum(s["files_after"] for s in summaries)
    ok_tables = [s for s in summaries if s["status"] == "done"]
    partial_tables = [s for s in summaries if s["status"] in ("max_batches", "early_give_up")]
    failed_tables = [s for s in summaries if s["status"] in ("connect_failed", "conn_lost", "count_failed", "unknown")]

    context.log.info("=" * 60)
    context.log.info("[compact_ducklake] Run summary:")
    context.log.info(f"  tables_total:   {len(summaries)}")
    context.log.info(f"  tables_done:    {len(ok_tables)}")
    context.log.info(f"  tables_partial: {len(partial_tables)}")
    context.log.info(f"  tables_failed:  {len(failed_tables)}")
    context.log.info(f"  files_before:   {total_before}")
    context.log.info(f"  files_after:    {total_after}")
    context.log.info(f"  files_merged:   {total_before - total_after}")
    context.log.info(f"  elapsed_total:  {elapsed_total:.1f}s")
    for s in summaries:
        context.log.info(
            f"    [{s['status']:>14}] {s['schema']}.{s['table']}: "
            f"{s['files_before']} → {s['files_after']} "
            f"(ok/failed: {s['batches_ok']}/{s['batches_failed']}, {s['elapsed']:.1f}s)"
        )


@asset(group_name="maintenance", pool="s3")
def gc_ducklake(context: AssetExecutionContext):
    """Expire old snapshots and delete unreferenced S3 files."""
    load_secrets()
    conn = get_ducklake_connection()
    try:
        conn.execute(f"SET ducklake_max_retry_count = {DUCKLAKE_MAX_RETRY}")
        context.log.info("Expiring snapshots older than 1 day")
        conn.execute(
            "CALL ducklake_expire_snapshots('lakehouse', older_than => now() - INTERVAL '1 day')"
        )
        context.log.info("Cleaning up unreferenced files")
        conn.execute(
            "CALL ducklake_cleanup_old_files('lakehouse', cleanup_all => true)"
        )
        context.log.info("GC complete")
    finally:
        conn.close()
