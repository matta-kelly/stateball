import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.auth.db import get_pg
from backend.auth.dependencies import require_admin
from backend.auth.jwt import create_token
from backend.auth.models import Role, User
from backend.auth.passwords import hash_password, verify_password
from backend.config import settings

router = APIRouter()

INVITE_TTL_HOURS = 48


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str


class SetPasswordRequest(BaseModel):
    password: str


# ---------------------------------------------------------------------------
# Public endpoints (no auth required)
# ---------------------------------------------------------------------------

@router.post("/login")
def login(body: LoginRequest):
    with get_pg() as conn:
        row = conn.execute(
            "SELECT id, username, password, role, status FROM users WHERE username = %s",
            (body.username,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    status = row[4]
    if status == "pending":
        raise HTTPException(status_code=403, detail="Account pending approval")
    if status == "approved":
        raise HTTPException(status_code=403, detail="Account approved — check your invite link to set a password")
    if status != "active":
        raise HTTPException(status_code=403, detail="Account not active")

    if not verify_password(row[2], body.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = User(id=str(row[0]), username=row[1], role=Role(row[3]))
    token = create_token(user)

    response = JSONResponse(content={"username": user.username, "role": user.role.value})
    response.set_cookie(
        key="token",
        value=token,
        httponly=True,
        secure=settings.jwt_cookie_secure,
        samesite="lax",
        max_age=settings.jwt_ttl_hours * 3600,
        path="/",
    )
    return response


@router.post("/logout")
def logout():
    response = JSONResponse(content={"detail": "Logged out"})
    response.delete_cookie(key="token", path="/")
    return response


@router.post("/register", status_code=201)
def register(body: RegisterRequest):
    username = body.username.strip()
    if not username or len(username) < 2:
        raise HTTPException(status_code=400, detail="Username must be at least 2 characters")

    with get_pg() as conn:
        existing = conn.execute(
            "SELECT id, status FROM users WHERE username = %s",
            (username,),
        ).fetchone()

        if existing:
            status = existing[1]
            if status == "pending":
                raise HTTPException(status_code=409, detail="Request already pending")
            raise HTTPException(status_code=409, detail="Username taken")

        conn.execute(
            "INSERT INTO users (username, password, role, status) VALUES (%s, %s, %s, %s)",
            (username, "", "user", "pending"),
        )
        conn.commit()

    return {"detail": "Registration request submitted"}


@router.get("/invite/{token}")
def validate_invite(token: str):
    with get_pg() as conn:
        row = conn.execute(
            "SELECT username, invite_expires FROM users "
            "WHERE invite_token = %s AND status = 'approved'",
            (token,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Invalid or expired invite")

    if row[1] and row[1] < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="Invite link has expired")

    return {"valid": True, "username": row[0]}


@router.post("/invite/{token}")
def accept_invite(token: str, body: SetPasswordRequest):
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    with get_pg() as conn:
        row = conn.execute(
            "SELECT id, invite_expires FROM users "
            "WHERE invite_token = %s AND status = 'approved'",
            (token,),
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Invalid or expired invite")

        if row[1] and row[1] < datetime.now(timezone.utc):
            raise HTTPException(status_code=410, detail="Invite link has expired")

        conn.execute(
            "UPDATE users SET password = %s, status = 'active', "
            "invite_token = NULL, invite_expires = NULL WHERE id = %s",
            (hash_password(body.password), row[0]),
        )
        conn.commit()

    return {"detail": "Password set — you can now sign in"}


# ---------------------------------------------------------------------------
# Authenticated endpoints
# ---------------------------------------------------------------------------

@router.get("/me")
def me(request: Request):
    user: User = request.state.user
    return {"id": user.id, "username": user.username, "role": user.role.value}


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@router.get("/admin/users")
def list_users(_user: User = Depends(require_admin())):
    with get_pg() as conn:
        rows = conn.execute(
            "SELECT id, username, role, status, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()

    return [
        {
            "id": str(r[0]),
            "username": r[1],
            "role": r[2],
            "status": r[3],
            "created_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]


@router.post("/admin/approve/{user_id}")
def approve_user(user_id: str, request: Request, _user: User = Depends(require_admin())):
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=INVITE_TTL_HOURS)

    with get_pg() as conn:
        row = conn.execute(
            "SELECT status FROM users WHERE id = %s::uuid", (user_id,)
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        if row[0] not in ("pending", "approved"):
            raise HTTPException(status_code=400, detail=f"Cannot approve user with status '{row[0]}'")

        conn.execute(
            "UPDATE users SET status = 'approved', invite_token = %s, invite_expires = %s WHERE id = %s::uuid",
            (token, expires, user_id),
        )
        conn.commit()

    scheme = request.headers.get("x-forwarded-proto", "https")
    host = request.headers.get("host", "stateball.example.com")
    invite_url = f"{scheme}://{host}/invite/{token}"

    return {"invite_url": invite_url}


@router.post("/admin/deny/{user_id}")
def deny_user(user_id: str, _user: User = Depends(require_admin())):
    with get_pg() as conn:
        row = conn.execute(
            "SELECT status FROM users WHERE id = %s::uuid", (user_id,)
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        conn.execute("DELETE FROM users WHERE id = %s::uuid", (user_id,))
        conn.commit()

    return {"detail": "User denied and removed"}


@router.delete("/admin/users/{user_id}")
def delete_user(user_id: str, request: Request, _user: User = Depends(require_admin())):
    current_user: User = request.state.user
    if current_user.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    with get_pg() as conn:
        row = conn.execute(
            "SELECT username, role FROM users WHERE id = %s::uuid", (user_id,)
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        if row[1] == "admin":
            raise HTTPException(status_code=400, detail="Cannot delete admin users")

        conn.execute("DELETE FROM users WHERE id = %s::uuid", (user_id,))
        conn.commit()

    return {"detail": f"User '{row[0]}' deleted"}
