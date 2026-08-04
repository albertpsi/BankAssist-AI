from bankassist.guardrails.redaction import redact, redact_mapping
from bankassist.security.passwords import hash_password


def test_redacts_bearer_token():
    text = "Authorization: Bearer abc123.def456-ghi"
    assert "abc123" not in redact(text)
    assert "[REDACTED]" in redact(text) or "Bearer [REDACTED]" in redact(text)


def test_redacts_jwt():
    jwt = (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    text = f"token was {jwt}"
    result = redact(text)
    assert jwt not in result
    assert "[REDACTED_JWT]" in result


def test_redacts_openai_api_key():
    text = "key=sk-abcdefghijklmnopqrstuvwx"
    result = redact(text)
    assert "sk-abcdefghijklmnopqrstuvwx" not in result
    assert "[REDACTED_API_KEY]" in result


def test_redacts_bcrypt_hash():
    real_hash = hash_password("Demo@Pass123")
    text = f"hash: {real_hash}"
    result = redact(text)
    assert real_hash not in result
    assert "[REDACTED_HASH]" in result


def test_leaves_normal_text_untouched():
    text = "Your recent transaction at Swiggy was ₹450.00 on 2026-07-30."
    assert redact(text) == text


def test_redact_mapping_masks_known_field_names():
    data = {"password_hash": "$2b$12$abc", "note": "hello"}
    result = redact_mapping(data)
    assert result["password_hash"] == "[REDACTED]"
    assert result["note"] == "hello"
