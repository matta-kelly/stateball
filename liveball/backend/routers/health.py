import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health():
    try:
        conn = get_conn()
        conn.execute("SELECT 1")
        conn.close()
    except Exception:
        logger.warning("Health check failed: DuckLake unreachable")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "stateball"},
        )
    return {"status": "ok", "service": "stateball"}
