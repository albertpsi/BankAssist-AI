"""Local login endpoint (Lab 4, ADR-0010, FR-22)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from bankassist.api.schemas import LoginRequest, LoginResponse
from bankassist.config import Settings
from bankassist.errors import AuthenticationError
from bankassist.security.jwt_tokens import issue_token
from bankassist.security.passwords import verify_password
from bankassist.tools import banking_data

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse, summary="Authenticate with a demo account")
def login(payload: LoginRequest, request: Request) -> LoginResponse:
    """Verify credentials against the seeded ``users`` table and issue a JWT.

    Deliberately generic on failure (FR-22): a bad username and a bad password get
    the identical ``authentication_error``.
    """
    settings: Settings = request.app.state.settings

    with banking_data.session(settings.banking_db_path) as conn:
        row = conn.execute(
            "SELECT id, password_hash, customer_id, role, is_active FROM users WHERE username = ?",
            (payload.username,),
        ).fetchone()

    if (
        row is None
        or not row["is_active"]
        or not verify_password(payload.password, row["password_hash"])
    ):
        raise AuthenticationError("Invalid username or password.")

    token = issue_token(
        settings=settings, user_id=row["id"], role=row["role"], customer_id=row["customer_id"]
    )
    return LoginResponse(access_token=token, role=row["role"], customer_id=row["customer_id"])
