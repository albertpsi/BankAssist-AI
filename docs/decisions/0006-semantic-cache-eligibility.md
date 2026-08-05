# ADR-0006 — Semantic cache eligibility and customer-data bypass

**Status:** Accepted — storage backend superseded by [ADR-0013](0013-redis-caching-layer.md)
**Date:** 2026-08-03

> **2026-08-05 update:** the *eligibility policy* below (what may be cached, the
> fail-closed default, the twice-checked rule) is unchanged and remains authoritative.
> Only the storage backend changes: Lab 7 (ADR-0013) implements this cache against
> **Redis**, not the SQLite plan implied when this ADR was written. Every reference to
> "the cache" below should be read as "the cache, now Redis-backed."

## Context

Lab 7 introduces a semantic response cache: embed the incoming query, compare it to stored
queries, and serve a previous answer when similarity exceeds a threshold. For repeated
policy and FAQ questions this eliminates the generation call entirely, which is the largest
single cost lever in the system.

It is also the component most able to cause a **privacy failure** rather than merely a wrong
answer. A semantic cache matches on *meaning*, not on identity. "What did I spend last
month?" and "How much did I spend last month?" are near-identical embeddings — and if the
first was asked by customer A and the second by customer B, a naive cache serves B customer
A's transaction summary. That is a data breach, and it would be produced by a system that is
working exactly as a generic cache is supposed to work.

The cache must therefore be **context-aware**, and this is a banking security and privacy
trade-off worth recording explicitly rather than burying in a config flag.

## Decision

Every request makes an **explicit eligibility decision before lookup and again before
store**, and the decision **defaults to bypass**.

| | Eligible for caching | Bypass (the default) |
|---|---|---|
| **Content** | Policy, product terms, fees, FAQ, general banking education | Transactions, customer profiles, dispute cases, balances, any personalized banking state |
| **Signal** | Route is `policy` or `general` **and** no customer-scoped tool was invoked | Everything else, including unknown or unclassified routes |
| **Lookup** | Performed | Never performed |
| **Store** | Performed after generation | Never performed |

Design rules that follow:

1. **Eligibility is established positively, never assumed.** An unrecognized or
   unclassified route is a bypass. Adding a new route defaults it to bypass until someone
   deliberately marks it cacheable — the failure mode of forgetting is a cache miss, not a
   leak.
2. **The check runs twice.** Once before lookup (using the classified route) and once before
   store (using what the request *actually* did). A request that looked like a policy
   question but ended up invoking a customer-scoped tool is not written to the cache.
3. **No customer identity in the cache at all.** Eligible entries contain no customer
   identity because customer-scoped requests never reach the cache. Cross-customer leakage
   is therefore structurally impossible rather than prevented by a key-scoping convention
   that could be got wrong.
4. **Output guardrails run on cache hits.** A cached response is not a trusted response.
5. **Entries expire (TTL).** A policy document update must not be shadowed indefinitely by
   a stale cached answer.
6. **The decision is traced** with its reason, so any cache hit can be audited after the
   fact — which is the same standard applied to guardrail verdicts.

## Alternatives considered

**Cache everything, key by `customer_id`** — the obvious approach: include the customer in
the cache key so entries cannot cross customers. Rejected. It makes safety depend on every
future call site remembering to pass the right key, it caches personalized data at rest for
no real benefit (customer-specific questions repeat rarely, so the hit rate is near zero
anyway), and a single omission is a breach rather than a bug.

**Exclusion list** — cache by default, exclude known-unsafe routes. Rejected: the failure
mode of forgetting to add a route to the list is a data leak. Defaulting to bypass inverts
that so the failure mode of forgetting is a missed optimization.

**Cache the retrieved context but not the answer** — a middle ground that caches retrieval
work while regenerating per customer. Rejected for this project as unnecessary complexity:
retrieval is local and cheap here (embeddings and reranking run on CPU at zero marginal
cost), so the saving would be latency-only while the generation call — the actual cost — is
still made. Worth revisiting in a deployment where retrieval is a paid API call.

**No semantic cache at all** — safest, and removes a whole class of risk. Rejected: it is a
required Lab 7 capability, and the eligibility rule above makes it safe for exactly the
content where it also happens to be effective.

## Consequences

**Makes easy:** a genuinely safe cache. Policy and FAQ questions — which repeat often across
users and carry no personal data — get the full benefit, which is where nearly all of the
achievable hit rate lives anyway. Every cache decision is auditable in the trace.

**Makes hard:** no caching benefit for customer-specific requests, so those keep their full
latency and cost. Accepted: the hit rate there would be near zero regardless, so almost
nothing is given up.

**Accepted:** a slightly more involved cache API (an eligibility object rather than a plain
key/value get/set) in exchange for a safety property that holds structurally rather than by
convention.

**Test obligation:** an explicit test asserts that a customer-specific request is never
stored and never served from cache. Per the testing skill, that test may not be weakened —
it is a governance test, not a performance one.

## Revisit when

Retrieval becomes a paid external call (making context-level caching worth its complexity),
per-customer caching gains a real hit rate under production traffic patterns, or a
requirement appears for cached personalized content — which would need a fundamentally
different design with per-customer isolation and a documented retention policy.
