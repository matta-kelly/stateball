import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.auth.middleware import AuthMiddleware
from backend.auth.router import router as auth_router
from backend.routers import artifacts, games, health, sim, stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stateball", docs_url="/api/docs")


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.add_middleware(AuthMiddleware)

app.include_router(auth_router, prefix="/api/auth")
app.include_router(health.router, prefix="/api")
app.include_router(artifacts.router, prefix="/api/v1")
app.include_router(sim.router, prefix="/api/v1")
app.include_router(games.router, prefix="/api/v1")
app.include_router(stats.router, prefix="/api/v1")

# Serve React SPA — mount static assets first for real files (js/css/etc),
# then a catch-all for client-side routes that returns index.html.
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    index_html = static_dir / "index.html"
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        file = static_dir / path
        if path and file.exists() and file.is_file():
            return FileResponse(file)
        return FileResponse(index_html)
