import json

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.auth.dependencies import require_admin
from backend.auth.models import User
from backend.db import get_conn

router = APIRouter()

_EXPECTED_TYPES = ["xgboost_sim", "xgboost_live", "baserunning", "pitcher_exit", "win_expectancy", "feature_manifest", "n_lookup", "stopping_thresholds", "gamma_schedule", "horizon_cutoff"]


def _serialize_artifact(item: dict) -> dict:
    """Parse metrics JSON and serialize timestamps."""
    if isinstance(item.get("metrics"), str):
        try:
            item["metrics"] = json.loads(item["metrics"])
        except (json.JSONDecodeError, TypeError):
            pass
    if item.get("created_at") is not None:
        item["created_at"] = str(item["created_at"])
    return item


def _slot_col(slot: str) -> str:
    return {"prod": "is_prod", "test": "is_test"}[slot]


def _validate_slot(artifacts: dict) -> dict:
    missing = [t for t in _EXPECTED_TYPES if t not in artifacts]
    checks = [
        {
            "name": "all_slots_filled",
            "passed": not missing,
            "detail": f"missing: {', '.join(missing)}" if missing else f"all {len(_EXPECTED_TYPES)} assigned",
        }
    ]

    for xg_type, expected_feats in [("xgboost_live", 172), ("xgboost_sim", 170)]:
        xg = artifacts.get(xg_type)
        if xg:
            metrics = xg.get("metrics", {})
            n_features = metrics.get("n_features", 0)
            checks.append({
                "name": f"{xg_type}_features",
                "passed": n_features == expected_feats,
                "detail": f"{n_features} features" + ("" if n_features == expected_feats else f" (expected {expected_feats})"),
            })
            classes = metrics.get("classes", [])
            checks.append({
                "name": f"{xg_type}_classes",
                "passed": len(classes) > 0,
                "detail": f"{len(classes)} classes",
            })

    return {"valid": all(c["passed"] for c in checks), "checks": checks}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/modeling/overview")
def get_modeling_overview():
    """Single endpoint for the modeling overview page.

    One query, returns prod and test artifacts with validation.
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT artifact_id, artifact_type, run_id, s3_path, "
            "created_at, metrics, is_prod, is_test "
            "FROM lh.main.artifact_registry "
            "WHERE is_prod = TRUE OR is_test = TRUE"
        ).fetchall()
        cols = [d[0] for d in conn.description]

        prod: dict = {}
        test: dict = {}
        for row in rows:
            item = _serialize_artifact(dict(zip(cols, row)))
            atype = item["artifact_type"]
            if item.get("is_prod"):
                prod[atype] = item
            if item.get("is_test"):
                test[atype] = item

        return {
            "prod": {"artifacts": prod, "validation": _validate_slot(prod)},
            "test": {"artifacts": test, "validation": _validate_slot(test)},
        }
    finally:
        conn.close()


@router.get("/artifacts")
def list_artifacts(type: str | None = Query(None)):
    conn = get_conn()
    try:
        if type:
            rows = conn.execute(
                "SELECT * FROM lh.main.artifact_registry "
                "WHERE artifact_type = ? ORDER BY created_at DESC",
                [type],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM lh.main.artifact_registry ORDER BY created_at DESC"
            ).fetchall()

        cols = [d[0] for d in conn.description]
        return [_serialize_artifact(dict(zip(cols, row))) for row in rows]
    finally:
        conn.close()


@router.patch("/artifacts/{artifact_id:path}/slot")
def set_artifact_slot(
    artifact_id: str,
    slot: str = Query(...),
    active: bool = Query(True),
    _user: User = Depends(require_admin()),
):
    """Assign or unassign an artifact to/from a slot.

    Only one artifact per (type, slot) — previous holder is cleared.
    An artifact can be in both prod and test simultaneously.
    """
    if slot not in ("prod", "test"):
        raise HTTPException(status_code=400, detail="slot must be 'prod' or 'test'")

    conn = get_conn()
    try:
        col = _slot_col(slot)

        rows = conn.execute(
            "SELECT artifact_type FROM lh.main.artifact_registry WHERE artifact_id = ?",
            [artifact_id],
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Artifact not found")
        artifact_type = rows[0][0]

        if not active:
            conn.execute(
                f"UPDATE lh.main.artifact_registry SET {col} = FALSE WHERE artifact_id = ?",
                [artifact_id],
            )
        else:
            conn.execute(
                f"UPDATE lh.main.artifact_registry SET {col} = FALSE "
                f"WHERE artifact_type = ? AND {col} = TRUE",
                [artifact_type],
            )
            conn.execute(
                f"UPDATE lh.main.artifact_registry SET {col} = TRUE WHERE artifact_id = ?",
                [artifact_id],
            )

        return {"artifact_id": artifact_id, "slot": slot, "active": active}
    finally:
        conn.close()


@router.delete("/artifacts/{artifact_id:path}")
def delete_artifact(artifact_id: str, _user: User = Depends(require_admin())):
    """Delete an artifact. Blocked if assigned to any slot."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT is_prod, is_test FROM lh.main.artifact_registry WHERE artifact_id = ?",
            [artifact_id],
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Artifact not found")

        is_prod, is_test = rows[0]
        slots = []
        if is_prod:
            slots.append("prod")
        if is_test:
            slots.append("test")
        if slots:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete: assigned to {', '.join(slots)}. Unassign first.",
            )

        conn.execute(
            "DELETE FROM lh.main.artifact_registry WHERE artifact_id = ?",
            [artifact_id],
        )
        return {"deleted": artifact_id}
    finally:
        conn.close()


@router.get("/artifacts/{artifact_id:path}/metrics")
def get_artifact_metrics(artifact_id: str):
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT metrics FROM lh.main.artifact_registry WHERE artifact_id = ?",
            [artifact_id],
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Artifact not found")
        raw = rows[0][0]
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {"raw": raw}
        return raw or {}
    finally:
        conn.close()
