from bankassist.security.passwords import hash_password, verify_password


def test_hash_then_verify_succeeds():
    hashed = hash_password("Demo@Pass123")
    assert verify_password("Demo@Pass123", hashed) is True


def test_verify_rejects_wrong_password():
    hashed = hash_password("Demo@Pass123")
    assert verify_password("wrong", hashed) is False


def test_hash_is_never_plaintext():
    hashed = hash_password("Demo@Pass123")
    assert "Demo@Pass123" not in hashed


def test_verify_rejects_malformed_hash_without_raising():
    assert verify_password("anything", "not-a-real-hash") is False
