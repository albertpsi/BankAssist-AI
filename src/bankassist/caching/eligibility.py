"""Cache eligibility classification (Lab 7, reuses ADR-0006's rule).

ADR-0006 ("Semantic cache eligibility and customer-data bypass") already
established the correct policy for Lab 2-6's SQLite-backed design: eligibility
is decided *positively*, defaults to bypass, and is re-checked after execution
in case a request that looked cacheable ended up touching customer state. Lab 7
does not reinvent that policy — it gives it a typed, three-way vocabulary
(``CacheEligibility``) so the *decision itself* is a first-class, traceable
value passed to both the semantic cache and the tool cache, per ADR-0013.

Only two of the three values are implemented in Lab 7 (see the Lab 7 plan
amendment: "explicitly classify requests as GLOBAL_CACHEABLE or NOT_CACHEABLE;
leave SESSION_CACHEABLE documented but unimplemented"). ``SESSION_CACHEABLE`` is
reserved for a future per-session, per-customer cache tier that is not built
here — see "Revisit when" below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Routes ADR-0006 already treats as free of customer identity. Only routes with
# a real call site that actually invokes `classify_eligibility` for them belong
# here — an unrecognized, unclassified, or not-yet-wired route is deliberately
# left out and falls through to the fail-closed default below. (`POLICY` is the
# only route `agents/graph.py`'s `policy_node` currently passes through.)
_GLOBALLY_CACHEABLE_ROUTES = frozenset({"POLICY"})


class CacheEligibility(StrEnum):
    """The three-way eligibility classification (Lab 7 amendment #4).

    GLOBAL_CACHEABLE
        Safe to cache and serve to *any* requester: no customer identity, no
        mutable state — policy, FAQ, KYC, card terms, dispute policy, general
        banking information. Implemented in Lab 7.

    SESSION_CACHEABLE
        A future tier for content that is safe to cache *scoped to one
        customer's session* (e.g. that customer's own account summary,
        re-asked within the same conversation) but must never be served across
        customers. **Not implemented in Lab 7** — every code path that would
        otherwise produce this value currently falls through to
        ``NOT_CACHEABLE`` instead, so no session-scoped cache exists yet and no
        cache key ever needs to carry a session/customer identity.

    NOT_CACHEABLE
        The ADR-0006 default: any customer-scoped, mutable, or unclassified
        request. Never looked up, never stored.
    """

    GLOBAL_CACHEABLE = "GLOBAL_CACHEABLE"
    SESSION_CACHEABLE = "SESSION_CACHEABLE"  # documented, unimplemented (see above)
    NOT_CACHEABLE = "NOT_CACHEABLE"


@dataclass(frozen=True)
class EligibilityDecision:
    eligibility: CacheEligibility
    reason: str


def classify_eligibility(
    *, route: str | None, customer_scoped_tool_invoked: bool
) -> EligibilityDecision:
    """Classify one request for the semantic cache, per ADR-0006's rule.

    ``route`` is the supervisor's classified route (``POLICY``, ``BANKING``,
    ``DISPUTE``, ``CLARIFICATION``, or ``None``/unclassified).
    ``customer_scoped_tool_invoked`` reflects what the request *actually did*,
    not just what it was classified as — this is the ADR-0006 "check runs
    twice" rule: called once before lookup with ``False`` (nothing has run
    yet), and again before store with the real value.

    An unrecognized or unclassified route is bypassed (fails closed), exactly
    as ADR-0006 requires.
    """
    if customer_scoped_tool_invoked:
        return EligibilityDecision(
            CacheEligibility.NOT_CACHEABLE,
            "A customer-scoped tool was invoked; customer data is never cached (ADR-0006).",
        )
    if route in _GLOBALLY_CACHEABLE_ROUTES:
        return EligibilityDecision(
            CacheEligibility.GLOBAL_CACHEABLE,
            f"Route '{route}' carries no customer identity and no mutable state.",
        )
    return EligibilityDecision(
        CacheEligibility.NOT_CACHEABLE,
        f"Route '{route}' is not on the cacheable allowlist (fail-closed default, ADR-0006).",
    )
