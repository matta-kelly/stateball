from fastapi import HTTPException, Request

from backend.auth.models import Role, User


def require_admin() -> User:
    def dependency(request: Request) -> User:
        user: User = request.state.user
        if user.role != Role.admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user
    return dependency
