"""Simple session-based authentication."""

import secrets
from fastapi import Request, HTTPException, Response
from functools import wraps

from .config import AUTH_PASSWORD

# In-memory session store (just a set of valid tokens)
_sessions: set[str] = set()


def create_session() -> str:
    """Create a new session token."""
    token = secrets.token_urlsafe(32)
    _sessions.add(token)
    return token


def validate_session(token: str | None) -> bool:
    """Check if a session token is valid."""
    if not token:
        return False
    return token in _sessions


def delete_session(token: str) -> None:
    """Remove a session token."""
    _sessions.discard(token)


def verify_password(password: str) -> bool:
    """Verify the provided password against AUTH_PASSWORD."""
    if not AUTH_PASSWORD:
        # No password set = auth disabled (for local dev)
        return True
    return secrets.compare_digest(password, AUTH_PASSWORD)


def get_session_token(request: Request) -> str | None:
    """Extract session token from cookie."""
    return request.cookies.get("session")


def require_auth(request: Request) -> None:
    """Raise 401 if not authenticated. Use as a dependency."""
    if not AUTH_PASSWORD:
        # Auth disabled if no password configured
        return
    token = get_session_token(request)
    if not validate_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
