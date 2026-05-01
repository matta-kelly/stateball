import logging

from starlette.types import ASGIApp, Receive, Scope, Send

from backend.auth.jwt import verify_token
from backend.auth.models import Role, User

logger = logging.getLogger(__name__)

OPEN_PATHS = (
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/register",
    "/api/health",
    "/api/docs",
    "/api/openapi.json",
    "/api/v1/games/trackable",
)
OPEN_PREFIXES = ("/api/auth/invite/",)


def _parse_cookies(header: str) -> dict[str, str]:
    cookies = {}
    for pair in header.split(";"):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            cookies[k.strip()] = v.strip()
    return cookies


def _client_ip(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            return value.decode().split(",")[0].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


class AuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]

        if not path.startswith("/api") or path in OPEN_PATHS or any(path.startswith(p) for p in OPEN_PREFIXES):
            await self.app(scope, receive, send)
            return

        # Extract token from cookie header
        token = None
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"cookie":
                cookies = _parse_cookies(header_value.decode())
                token = cookies.get("token")
                break

        if not token:
            logger.warning("Auth: no token on %s from %s", path, _client_ip(scope))
            await self._send_401(send)
            return

        try:
            payload = verify_token(token)
            user = User(
                id=payload["sub"],
                username=payload["username"],
                role=Role(payload["role"]),
            )
            scope.setdefault("state", {})
            scope["state"]["user"] = user
        except Exception as exc:
            logger.warning("Auth: invalid token on %s from %s: %s", path, _client_ip(scope), exc)
            await self._send_401(send)
            return

        await self.app(scope, receive, send)

    async def _send_401(self, send: Send) -> None:
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"detail":"Not authenticated"}',
        })
