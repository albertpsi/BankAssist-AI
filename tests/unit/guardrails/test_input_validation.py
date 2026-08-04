from bankassist.guardrails.input_validation import validate_message_shape


def test_blank_message_is_blocked():
    result = validate_message_shape("   ")
    assert result.allowed is False
    assert result.action == "BLOCK"


def test_empty_message_is_blocked():
    result = validate_message_shape("")
    assert result.allowed is False


def test_oversized_message_is_blocked():
    result = validate_message_shape("a" * 2001, max_chars=2000)
    assert result.allowed is False
    assert result.metadata["length"] == 2001


def test_normal_message_is_allowed():
    result = validate_message_shape("What documents are accepted for KYC?")
    assert result.allowed is True
    assert result.action == "ALLOW"
