"""Run differential bucketing — single source of truth.

Used by: sampling query (run.py), estimator runtimes (naive_mc.py, smc.py),
calibration builders (calibration_builders.py).

13 buckets: single run_diff for [-5, +5], grouped tails beyond.
"""

from __future__ import annotations


def rd_to_bucket(rd: int) -> str:
    """Map integer run differential to bucket label."""
    if rd <= -6:
        return "rd_lte_m6"
    if rd >= 6:
        return "rd_gte_p6"
    if rd == 0:
        return "rd_0"
    if rd < 0:
        return f"rd_m{abs(rd)}"
    return f"rd_p{rd}"


def bucket_to_rd(bucket: str) -> int:
    """Map bucket label to representative run differential (for WE lookups)."""
    if bucket == "rd_lte_m6":
        return -7
    if bucket == "rd_gte_p6":
        return 7
    if bucket == "rd_0":
        return 0
    # rd_m3 → -3, rd_p2 → 2
    s = bucket.replace("rd_", "")
    if s.startswith("m"):
        return -int(s[1:])
    if s.startswith("p"):
        return int(s[1:])
    return 0


def state_key(inning: int, run_diff: int) -> str:
    """Build state key from inning and run differential."""
    return f"{inning}|{rd_to_bucket(run_diff)}"


ALL_BUCKETS = [rd_to_bucket(rd) for rd in range(-6, 7)]

N_BUCKETS = len(ALL_BUCKETS)

# SQL fragment for use in sampling queries. Matches rd_to_bucket logic exactly.
SQL_CASE = """CASE
        WHEN CAST(v.run_diff AS INT) <= -6 THEN 'rd_lte_m6'
        WHEN CAST(v.run_diff AS INT) = -5 THEN 'rd_m5'
        WHEN CAST(v.run_diff AS INT) = -4 THEN 'rd_m4'
        WHEN CAST(v.run_diff AS INT) = -3 THEN 'rd_m3'
        WHEN CAST(v.run_diff AS INT) = -2 THEN 'rd_m2'
        WHEN CAST(v.run_diff AS INT) = -1 THEN 'rd_m1'
        WHEN CAST(v.run_diff AS INT) = 0 THEN 'rd_0'
        WHEN CAST(v.run_diff AS INT) = 1 THEN 'rd_p1'
        WHEN CAST(v.run_diff AS INT) = 2 THEN 'rd_p2'
        WHEN CAST(v.run_diff AS INT) = 3 THEN 'rd_p3'
        WHEN CAST(v.run_diff AS INT) = 4 THEN 'rd_p4'
        WHEN CAST(v.run_diff AS INT) = 5 THEN 'rd_p5'
        ELSE 'rd_gte_p6'
    END"""

