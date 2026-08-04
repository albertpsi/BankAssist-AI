from bankassist.guardrails.masking import is_masked_card, mask_sensitive_identifiers


def test_already_masked_card_format_recognized():
    assert is_masked_card("****-****-****-4321") is True
    assert is_masked_card("4111111111111111") is False


def test_masks_raw_card_number_in_text():
    text = "Your card number is 4111 1111 1111 1234, please confirm."
    result = mask_sensitive_identifiers(text)
    assert "4111 1111 1111 1234" not in result
    assert result.endswith("1234, please confirm.") or "1234" in result


def test_masks_ssn_shaped_string():
    text = "SSN on file: 123-45-6789"
    result = mask_sensitive_identifiers(text)
    assert "123-45-6789" not in result
    assert "***-**-****" in result


def test_never_touches_amounts_or_dates():
    text = "Your balance is ₹150000.00 as of 2026-08-01."
    assert mask_sensitive_identifiers(text) == text


def test_never_touches_short_ids():
    text = "Transaction TX1007 for account ACC-1."
    assert mask_sensitive_identifiers(text) == text
