"""Exception hierarchy (FR-L1-4)."""

from __future__ import annotations

import pytest

from bankassist.errors import BankAssistError, ConfigurationError, LLMError


def test_base_error_carries_code_status_and_message() -> None:
    err = BankAssistError("something broke")

    assert err.code == "internal_error"
    assert err.http_status == 500
    assert err.message == "something broke"
    assert err.details == {}


@pytest.mark.parametrize(
    ("error_cls", "expected_code", "expected_status"),
    [
        (ConfigurationError, "configuration_error", 500),
        (LLMError, "llm_error", 502),
    ],
)
def test_subclasses_declare_their_own_code_and_status(
    error_cls: type[BankAssistError], expected_code: str, expected_status: int
) -> None:
    err = error_cls("boom")

    assert err.code == expected_code
    assert err.http_status == expected_status
    assert isinstance(err, BankAssistError)


def test_details_are_serialized() -> None:
    err = ConfigurationError("bad field", details={"field": "llm_provider"})

    assert err.to_dict() == {
        "code": "configuration_error",
        "message": "bad field",
        "details": {"field": "llm_provider"},
    }


def test_error_is_catchable_as_exception() -> None:
    with pytest.raises(Exception, match="boom"):
        raise LLMError("boom")
