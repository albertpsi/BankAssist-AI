from bankassist.caching.eligibility import CacheEligibility, classify_eligibility


def test_policy_route_with_no_tool_call_is_globally_cacheable():
    decision = classify_eligibility(route="POLICY", customer_scoped_tool_invoked=False)
    assert decision.eligibility is CacheEligibility.GLOBAL_CACHEABLE


def test_banking_route_is_never_cacheable():
    decision = classify_eligibility(route="BANKING", customer_scoped_tool_invoked=False)
    assert decision.eligibility is CacheEligibility.NOT_CACHEABLE


def test_dispute_route_is_never_cacheable():
    decision = classify_eligibility(route="DISPUTE", customer_scoped_tool_invoked=True)
    assert decision.eligibility is CacheEligibility.NOT_CACHEABLE


def test_unclassified_route_fails_closed():
    decision = classify_eligibility(route=None, customer_scoped_tool_invoked=False)
    assert decision.eligibility is CacheEligibility.NOT_CACHEABLE


def test_unknown_route_fails_closed():
    decision = classify_eligibility(route="SOMETHING_NEW", customer_scoped_tool_invoked=False)
    assert decision.eligibility is CacheEligibility.NOT_CACHEABLE


def test_policy_looking_request_that_actually_invoked_a_customer_tool_is_bypassed():
    """ADR-0006 rule 2: the check runs twice — once before lookup, once before
    store using what the request *actually* did."""
    decision = classify_eligibility(route="POLICY", customer_scoped_tool_invoked=True)
    assert decision.eligibility is CacheEligibility.NOT_CACHEABLE


def test_session_cacheable_is_never_produced():
    """Lab 7 amendment #4: SESSION_CACHEABLE is documented but unimplemented —
    no combination of inputs should ever classify to it."""
    for route in ("POLICY", "BANKING", "DISPUTE", "CLARIFICATION", "UNSUPPORTED", None):
        for invoked in (True, False):
            decision = classify_eligibility(route=route, customer_scoped_tool_invoked=invoked)
            assert decision.eligibility is not CacheEligibility.SESSION_CACHEABLE
