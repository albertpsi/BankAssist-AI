import time

import jwt as pyjwt
import pytest
from pydantic import SecretStr

from bankassist.config import Settings
from bankassist.errors import AuthenticationError
from bankassist.security.jwt_tokens import decode_token, issue_token


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test-key", jwt_expiry_minutes=1)


def test_issue_then_decode_roundtrips_claims(settings: Settings):
    token = issue_token(
        settings=settings, user_id="USR-001", role="CUSTOMER", customer_id="CUST001"
    )
    claims = decode_token(token, settings=settings)
    assert claims["sub"] == "USR-001"
    assert claims["role"] == "CUSTOMER"
    assert claims["customer_id"] == "CUST001"


def test_decode_rejects_garbage_token(settings: Settings):
    with pytest.raises(AuthenticationError):
        decode_token("not-a-jwt", settings=settings)


def test_decode_rejects_expired_token(settings: Settings):
    past = int(time.time()) - 10
    token = pyjwt.encode(
        {"sub": "USR-001", "role": "CUSTOMER", "customer_id": "CUST001", "exp": past},
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(AuthenticationError):
        decode_token(token, settings=settings)


def test_decode_rejects_wrong_signature(settings: Settings):
    token = issue_token(
        settings=settings, user_id="USR-001", role="CUSTOMER", customer_id="CUST001"
    )
    other = settings.model_copy(update={"jwt_secret": SecretStr("a-different-secret")})
    with pytest.raises(AuthenticationError):
        decode_token(token, settings=other)
