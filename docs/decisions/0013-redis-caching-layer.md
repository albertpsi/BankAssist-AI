# ADR-0013 — Redis caching layer for Lab 7 cost optimization

**Status:** Accepted
**Date:** 2026-08-05
**Supersedes (in part):** the storage backend named in [ADR-0006](0006-semantic-cache-eligibility.md)
and in `CLAUDE.md` §4 ("Context-aware semantic cache (embeddings + SQLite)").

## Context

ADR-0006 designed the semantic cache's *eligibility policy* — what may be cached, what
must never be, and the fail-closed default — against an in-process, SQLite-backed
storage plan. That eligibility policy is correct and is not revisited here.

Lab 7's brief requires the storage backend itself to be **Redis**, running locally in
Docker, and requires two further caches beyond the semantic response cache: an
**embedding cache** and a **tool response cache**, all Redis-backed. This is a real
technology-stack change under `CLAUDE.md` §3 ("Do not add a dependency that is not
listed in the technology-stack doc without raising it as an ADR first") and was taken
through the project's Gate 3 (scope approval) before implementation — see the Lab 7
planning artifact (`docs/plan/lab-07-cost-optimization-plan.md`), approved with
amendments.

## Decision

1. **Storage backend is Redis**, not SQLite, for all three Lab 7 caches (semantic
   response, embedding, tool response). SQLite is no longer part of the caching design
   for any lab from this point forward. `CLAUDE.md` §4's caching row should be read as
   superseded by this ADR.
2. **ADR-0006's eligibility rule is reused, not replaced.** `bankassist.caching.eligibility`
   implements it as a typed three-way classification: `GLOBAL_CACHEABLE`,
   `SESSION_CACHEABLE`, `NOT_CACHEABLE`.
   - `GLOBAL_CACHEABLE` — implemented. Exactly ADR-0006's "eligible" case: policy, FAQ,
     KYC, card terms, dispute policy, general banking information; no customer identity,
     no mutable state.
   - `NOT_CACHEABLE` — implemented. ADR-0006's fail-closed default: any customer-scoped,
     mutable, or unclassified request.
   - `SESSION_CACHEABLE` — **documented, not implemented.** Reserved for a future
     per-session, per-customer cache tier (e.g. "that customer's own account summary,
     re-asked in the same conversation"). No code path in Lab 7 produces this value; it
     exists as a named placeholder so a future lab does not have to redesign the
     eligibility vocabulary to add it. See "Revisit when" below.
3. **Similarity search uses native Redis vector search (RediSearch `FT.SEARCH ... KNN`)**
   whenever the connected Redis exposes the `search` module, which the recommended
   local image (`redis/redis-stack-server`) always does. A bounded Python-side cosine
   scan is used only as a documented fallback when RediSearch is unavailable (a plain
   `redis:*` image, or a test double like `fakeredis` that implements neither `MODULE
   LIST` nor `FT.*`) — this is the specific, logged technical blocker the Lab 7
   amendment asked to have found and documented, not a default implementation choice.
4. **Embedding cache keys hash the model identifier together with the text**
   (`SHA256(model + "\0" + text)`), not merely prefix the key with the model name. A
   future embedding model change cannot collide with, or silently reuse, a
   differently-dimensioned vector produced by a different model.
5. **Tool cache keys are versioned, argument-hash-based**, never an entity id:
   `tool:{label}:{cache_key_version}:{sha256(canonical_json(args))}`. A cache key never
   contains a raw account id, transaction id, or document id in cleartext, and a global
   `CACHE_KEY_VERSION` bump invalidates every tool-cache and semantic-cache entry at
   once without touching the embedding cache (which stays valid across a corpus/version
   change, since it is keyed by model+text, not by version).
6. **Every cache decision — eligibility, hit, miss, and bypass — is recorded**, not only
   hits, via `ExecutionEvent`s and (when AgentOps is enabled) `update_metadata`. This
   matches the same auditability standard ADR-0006 already set for guardrail verdicts.
7. **Redis is entirely optional at runtime.** `REDIS_ENABLED` defaults to `false`; when
   false, or when Redis is unreachable, every cache degrades to the exact Labs 1-6
   uncached behavior — never an exception, never a blocked request. Same posture as
   Pinecone (`rag/vector_store.py`) and AgentOps (`observability/agentops_client.py`).

## Alternatives considered

**Keep SQLite, add Redis only for the two new caches** — rejected. Running two different
cache storage technologies side by side for what is conceptually one caching layer adds
operational complexity (two failure modes, two invalidation stories) for no benefit at
this project's scale; Redis alone serves all three caches identically well.

**RedisVL or a dedicated vector database (e.g. reusing Pinecone) for the semantic
cache** — rejected for this lab's scope. Pinecone is already the vector store for policy
retrieval (ADR-0007); doubling it as the semantic-cache store would blur the boundary
between "retrieved knowledge" and "cached answers" and add a second paid-service
dependency for a demo-scale cache. RediSearch, already available in the recommended
Docker image, is sufficient and keeps the whole caching layer on one piece of local
infrastructure.

**Implement the session-scoped cache tier now** — rejected for Lab 7. No current route
in the application needs it (BANKING/DISPUTE responses are per-request, not repeated
verbatim within a session at a rate that would justify the added privacy-boundary
complexity), and building it speculatively would violate the project's scope-discipline
principle (`CLAUDE.md` §3). `SESSION_CACHEABLE` is named now so a future lab that does
need it extends an existing vocabulary instead of redesigning it.

## Consequences

**Makes easy:** one consistent caching technology, cache statistics that work across
process boundaries (FastAPI and Streamlit are separate processes, per `CLAUDE.md` §4)
without a shared filesystem, and native vector search without hand-rolled ANN code in
the common case.

**Makes hard:** the Python cosine fallback path exists for correctness (the app must not
break without RediSearch) but is explicitly not the intended production path — some
tests exercise it deliberately (`fakeredis` has no RediSearch), which is documented in
those tests rather than presented as the primary design.

**Accepted:** Redis becomes a new local infrastructure dependency (`docker-compose.yml`),
where the project previously ran with no local database process at all (Pinecone and
OpenAI are the only network dependencies through Lab 6). This is accepted because it is
the Lab 7 brief's explicit requirement and is optional at runtime.

## Explicitly deferred (out of scope for Lab 7)

Per the approved Lab 7 scope, none of the following are implemented, and none of this
ADR's caching layer depends on them existing later:

- **Prompt caching / provider-native prompt caching** (e.g. OpenAI's automatic prompt
  caching, or a system-prompt cache). `CLAUDE.md` §8 already documents byte-stable
  system prompts as a prerequisite for provider prompt caching to work "best-effort" —
  that remains true and unaffected, but no BankAssist code measures or relies on it.
- **Model routing** (choosing a cheaper/faster model per request based on complexity).
- **Prompt/context compression** or a **token budget manager**.

These may be revisited as future cost-optimization work but are deliberately excluded
from this ADR's scope so the Redis caching layer can be reviewed and evaluated on its
own.

## Revisit when

- `SESSION_CACHEABLE` gains an actual use case (a route with a real, safe, per-customer
  repeat-question pattern) — it would need its own key design (customer-scoped, not
  global) and its own ADR extension, not a silent reuse of the global cache's keys.
- RediSearch becomes unavailable in the target deployment environment as a hard
  constraint (not just a local dev convenience) — the Python fallback's bounded
  candidate scan would need to become the primary, tuned path rather than a documented
  blocker.
- Any of the explicitly deferred techniques above are scoped as a future lab.
