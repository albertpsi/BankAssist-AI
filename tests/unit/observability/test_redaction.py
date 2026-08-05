"""Custom AgentOps span/trace attributes must never carry credentials, full
card numbers, or full account numbers (Lab 6 §8)."""

from __future__ import annotations

from bankassist.observability.redaction import sanitize_attributes


def test_drops_credential_shaped_keys() -> None:
    clean = sanitize_attributes(
        {
            "authorization": "Bearer sk-abc123",
            "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123",
            "api_key": "sk-abcdefghijklmnopqrstuvwx",
            "route": "banking",
        }
    )
    assert clean == {"route": "banking"}


def test_masks_card_and_account_shaped_values_in_strings() -> None:
    clean = sanitize_attributes({"summary": "Card 4111111111111111 flagged"})
    assert "4111111111111111" not in clean["summary"]


def test_redacts_secret_shaped_values_in_strings() -> None:
    clean = sanitize_attributes({"summary": "token was sk-abcdefghijklmnopqrstuvwx"})
    assert "sk-abcdefghijklmnopqrstuvwx" not in clean["summary"]


def test_does_not_drop_legitimate_token_count_attributes() -> None:
    """A raw substring check on "token" would wrongly also catch `output_tokens`
    (an LLM token count, not a credential) — this must stay a whole-word match."""
    clean = sanitize_attributes({"input_tokens": 120, "output_tokens": 45, "access_token": "x"})
    assert clean == {"input_tokens": 120, "output_tokens": 45}


def test_passes_through_safe_scalars() -> None:
    clean = sanitize_attributes({"latency_ms": 12.5, "grounded": True, "count": 3, "note": None})
    assert clean == {"latency_ms": 12.5, "grounded": True, "count": 3, "note": None}


def test_stringifies_unknown_shapes_without_content() -> None:
    clean = sanitize_attributes({"payload": {"account_id": "acc-1", "balance": 500}})
    assert clean["payload"] == "<dict>"
