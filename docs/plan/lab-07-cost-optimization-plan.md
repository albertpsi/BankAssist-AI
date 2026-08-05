# Lab 7 — Cost Optimization Architecture (Planning Artifact)

**Status:** APPROVED WITH AMENDMENTS (2026-08-04). Implementation proceeded under the
amendments below; see [ADR-0013](../decisions/0013-redis-caching-layer.md) for the
resulting design of record.

## Approved amendments (incorporated into the implementation)

1. Semantic similarity search uses native Redis vector search (RediSearch `FT.SEARCH
   ... KNN`) via `redis/redis-stack-server`, not Python-side cosine, except as a
   documented fallback when RediSearch is unavailable (§2 of ADR-0013).
2. Embedding cache keys hash `model + text` together (`SHA256(model + "\0" + text)`),
   not just a model-prefixed key.
3. Tool cache keys are versioned, argument-hash-based (`tool:{label}:{version}:{hash}`),
   never a raw entity id.
4. Cache eligibility is a typed three-way classification —
   `GLOBAL_CACHEABLE`/`SESSION_CACHEABLE`/`NOT_CACHEABLE` — reusing ADR-0006's rule.
   `SESSION_CACHEABLE` is documented but has no implemented code path in Lab 7.
5. Every cache decision (eligibility, hit, miss, bypass) is recorded via
   `ExecutionEvent`s and AgentOps metadata, not only hits.
6. Streamlit gained an "Optimization Summary" panel (cache decisions, LLM
   skipped/executed, estimated latency/cost saved).
7. A real before/after benchmark (`scripts/lab7_benchmark.py`) complements the
   extrapolated demo estimates in §7-8 below.
8. RedisInsight/key-browsing screenshots added to the screenshot plan (§12).
9. [ADR-0013](../decisions/0013-redis-caching-layer.md) records the SQLite→Redis move
   and explicitly defers prompt caching, model routing, and token budgeting.

The sections below are the original planning artifact and are left intact as the
design rationale; where an amendment changed a specific design point (e.g. "plain
Redis + bounded candidate scan" in §4.1/§13/Open Questions), the amendment above and
ADR-0013 are authoritative over the original text.

---
**Scope:** Semantic Cache, Embedding Cache, Tool Response Cache — all backed by Redis.
Explicitly out of scope: prompt caching, system-prompt caching, model routing, prompt/context
compression, token budget manager.

---

## 0. Conflict to resolve before approval

[`CLAUDE.md`](../../CLAUDE.md) §4 and [ADR-0006](../decisions/0006-semantic-cache-eligibility.md)
currently specify the semantic cache as **"embeddings + SQLite,"** in-process, no new
infrastructure. The Lab 7 brief specifies **Redis** for all three caches. This is a real
tech-stack change (CLAUDE.md §3: *"Do not add a dependency that is not listed in the
technology-stack doc without raising it as an ADR first"* — Gate 3, Scope approval).

This plan assumes **Redis is approved** for Lab 7 and treats it as superseding the SQLite
line in ADR-0006 (a new ADR-0013 will record this, not silently edit history). If Redis is
*not* wanted, say so now — the design below does not otherwise change.

---

## 1. Current architecture (Labs 1–6, unchanged)

```
                         User (Streamlit / API client)
                                    │
                                    ▼
                          FastAPI (auth_dependency,
                          routes/agent.py, routes/rag.py)
                                    │
                                    ▼
                    Input Guardrails (nemo_adapter, input_validation,
                              tool_authorization)
                                    │
                                    ▼
                    LangGraph Supervisor (agents/graph.py, supervisor.py)
                        route: BANKING | DISPUTE | POLICY | CLARIFICATION
                     ┌──────────────┼───────────────┐
                     ▼              ▼               ▼
              Banking Agent   Dispute Agent    Policy Agent
                     │              │               │
                     │              │               ▼
                     │              │     Enterprise RAG Pipeline
                     │              │   (classify → rewrite → hybrid
                     │              │    retrieve[vector+BM25] → RRF →
                     │              │    rerank → prompt_builder)
                     │              │        │              │
                     │              │        ▼              ▼
                     │              │  OpenAIEmbedder   PineconeVectorStore
                     │              │  (text-embedding-3-small)  + BM25 (local)
                     ▼              ▼
              tools/dispatcher.py → scoped_tools.py (RBAC + ownership,
                 banking_data.py / SQLite)
                     │
                     ▼
              LLMClient → OpenAI (gpt-4o-mini / gpt-4.1-mini)
                     │
                     ▼
              Output Guardrails (masking, redaction, citation checks)
                     │
                     ▼
              Tracer (span.py) + Observability (agentops_client.py,
                 decorators.py) → AgentOps + JSONL trace
                     │
                     ▼
                Response → User
```

Key facts this plan builds on:
- **Embeddings**: `OpenAIEmbedder` (`rag/embeddings.py`) — one call site for corpus
  (`embed_documents`) and query (`embed_query`) embedding, already traced with `SpanType.EMBEDDING`.
- **Retrieval**: `PineconeVectorStore.query()` (`rag/vector_store.py`) is the only vector
  lookup; enterprise pipeline also runs local BM25 + rerank.
- **Tools**: every scoped tool call passes through `tools/dispatcher.py::call_tool`, which
  already standardizes timing + `ExecutionEvent` emission — the natural interception point
  for a tool cache decorator/wrapper.
- **Routing signal for eligibility**: `agents/supervisor.py::decide_route` produces the
  `route` (`BANKING` / `DISPUTE` / `POLICY` / `CLARIFICATION`) already used by ADR-0006 as
  the customer-data bypass signal.
- **Observability**: `observability/agentops_client.py` + `decorators.py` is the existing,
  reusable hook for emitting new event types — no new tracing system needed.
- **Config**: `bankassist/config.py` `Settings` (env + `.env`) is the single place new
  `REDIS_*` / cache config would be added — never hardcoded, per existing convention.

---

## 2. Target architecture

```
                                User
                                 │
                                 ▼
                    Guardrails → Supervisor (route classified)
                                 │
                     Is route policy-eligible? (ADR-0006 rule,
                     reused verbatim — not reinvented for Lab 7)
                                 │
                 ┌───────────────┴───────────────┐
              ELIGIBLE                        NOT ELIGIBLE
                 │                                 │
                 ▼                                 ▼
        SEMANTIC CACHE (Redis)          Normal pipeline, cache bypassed
        embed query → cosine sim             entirely (no lookup, no store)
        vs stored query vectors
        threshold check
                 │
        ┌────────┴────────┐
      HIT                MISS
        │                  │
        ▼                  ▼
  Return cached      EMBEDDING CACHE (Redis)
  response,          SHA256(text) → cached vector?
  run output          │            │
  guardrails        yes           no
  (ADR-0006 rule    (reuse)    OpenAI Embeddings API → store
  4: cache hits         │            │
  are not trusted)      └─────┬──────┘
                               ▼
                    Enterprise RAG Pipeline
                    (vector + BM25 + RRF + rerank)
                               │
                               ▼
                    TOOL RESPONSE CACHE (Redis)
                    — deterministic tools only (§5) —
                    checked inside dispatcher.call_tool
                               │
                               ▼
                        Agent Execution → OpenAI
                               │
                               ▼
                  Store response in Semantic Cache
                  (only if still eligible post-execution —
                  ADR-0006 rule 2, re-checked after the fact)
                               │
                               ▼
                            AgentOps
                  (cache hit/miss/store events, Redis latency)
```

Redis is introduced as **one new infrastructure dependency**, reached only through
configuration (`REDIS_URL`, never a hardcoded host), and only from three new modules under
a new `bankassist/caching/` package. No existing module's behavior changes when Redis is
unavailable or disabled — see §9 (fallback).

---

## 3. New package layout (files to be created — not yet written)

```
src/bankassist/caching/
├─ __init__.py
├─ redis_client.py        # single Redis connection factory, from Settings; the only
│                          # module that imports redis-py, mirrors vector_store.py's
│                          # "one module owns the SDK" convention
├─ semantic_cache.py       # SemanticCache: eligibility check (reuses ADR-0006 logic),
│                          # embed → cosine search over stored vectors → threshold →
│                          # hit/miss, store
├─ embedding_cache.py      # EmbeddingCache: SHA256(text) → Redis GET/SET, wraps
│                          # OpenAIEmbedder without modifying it (decorator/wrapper,
│                          # not a rewrite)
├─ tool_cache.py           # ToolCache: deterministic-tool allowlist, key builder,
│                          # get/set, TTL
├─ models.py               # CacheDecision, CacheEvent, CacheStats pydantic models
└─ eligibility.py          # shared eligibility rule extracted from ADR-0006 so
                            # semantic_cache.py and tool_cache.py both call one
                            # function rather than duplicating the policy
```

### Modified files (existing behavior preserved, cache is additive)

| File | Change |
|---|---|
| `config.py` | Add `redis_url`, `redis_enabled`, `semantic_cache_ttl_seconds`, `semantic_cache_similarity_threshold`, `embedding_cache_ttl_seconds`, `tool_cache_ttl_seconds`, cache key version fields |
| `rag/embeddings.py` | `OpenAIEmbedder` gains an optional `EmbeddingCache` collaborator (constructor param, default `None` = today's behavior unchanged) |
| `tools/dispatcher.py` | `call_tool` gains an optional cache lookup/store around `fn()` for allowlisted deterministic tools only; RBAC/ownership check in the tool still always runs first — the cache never bypasses authorization |
| `agents/graph.py` / `supervisor.py` (read-only integration point) | Semantic cache lookup added at the point the route decision is known, before agent dispatch |
| `observability/decorators.py`, `agentops_client.py` | New event types: `semantic_cache_hit/miss/store`, `embedding_cache_hit/miss`, `tool_cache_hit/miss`, `redis_latency_ms` |
| `api/routes/*.py` | New `GET /api/v1/cache/stats` route |
| `ui/streamlit_app.py` / `agentic_app.py` | Cache status indicators (hit/miss/source/latency) on each response |
| `requirements.txt` | `redis>=5.0.0` |
| `docs/decisions/0006-semantic-cache-eligibility.md` | Superseded-by note pointing at new ADR-0013 |
| `docs/decisions/0013-redis-caching-layer.md` (new) | Records the SQLite→Redis change and the three-cache design |
| `docs/architecture/architecture.md`, `technology-stack.md` | Add Redis, add Lab 7 architecture section |
| `docs/requirements/project-requirements.md` | Add Redis-specific FRs under existing FR-7, or FR-7.x additions |
| `.env.example` | `REDIS_URL=`, `REDIS_ENABLED=` placeholders |
| `docker-compose.yml` (new, repo root) | Local Redis service |

No existing file's tests should need behavior changes — only additive tests (§7).

---

## 4. Cache designs

### 4.1 Semantic Cache
- **Eligible content only**: Policy, FAQ, KYC, Card Terms, Dispute Policy, general banking
  info — i.e. exactly the `POLICY` (and general/non-customer-scoped) route per ADR-0006,
  reused rather than reimplemented.
- **Never cached**: accounts, transactions, balances, dispute status, customer profile — any
  route that touched a customer-scoped tool. Eligibility is re-checked *after* execution
  (ADR-0006 rule 2) before store, in case a "policy-looking" question ended up calling a
  scoped tool.
- **Flow**: query → embed (via Embedding Cache, §4.2) → cosine similarity search against
  stored query vectors in Redis → configurable threshold (`SEMANTIC_CACHE_SIMILARITY_THRESHOLD`,
  default proposed `0.95`) → hit returns cached response (still passed through output
  guardrails per ADR-0006 rule 4) → miss runs the normal pipeline and stores the result.
- **Similarity search implementation note**: Redis alone (no RediSearch/vector module
  assumed available) does exact-key lookups, not vector search. Plan: maintain a small
  in-memory-at-request-time candidate set by storing vectors under `semantic:vectors:*`
  and computing cosine similarity in Python against a bounded candidate list (capped, e.g.
  latest N=500 eligible entries, or one list per coarse topic bucket) — acceptable at
  demo scale (NFR-3 targets ~1s for a cache hit). If RedisSearch/RedisVL is available in
  the environment we can use it instead — this needs confirming before implementation
  starts (see Open Question below).

### 4.2 Embedding Cache
- **Key**: `embedding:{model}:{sha256(text)}`. Model included in the key so a future model
  change can't silently return a stale-dimension vector.
- **Flow**: SHA256 the input text → Redis GET → hit reuses the vector; miss calls
  `OpenAIEmbedder._embed` (unchanged), then stores.
- Wraps the existing `OpenAIEmbedder` transparently — `embed_documents` and `embed_query`
  keep their exact signatures and tracing.
- Same model only: `text-embedding-3-small`, per scope.

### 4.3 Tool Response Cache
- **Cacheable** (deterministic, non-customer, read-only): Policy Lookup, FAQ Lookup, Card
  Agreement, KYC Documents, RBI Rules — see cacheability matrix (§5).
- **Never cached**: any tool in `tools/scoped_tools.py` (accounts, transactions, dispute
  eligibility/creation) — all take a `SecurityContext` and are customer-scoped by
  definition.
- **Enforcement point**: `tools/dispatcher.py::call_tool` — cache check wraps `fn()`,
  gated by an explicit allowlist of tool `label`s (not a blocklist — same "eligible must be
  named positively" principle as ADR-0006, to fail closed on new tools).
- RBAC/ownership authorization inside the tool still executes on every call — cache never
  substitutes for authorization; it caches the tool's *output*, not the authorization
  decision, and only for tools where the input has no customer identity to authorize against.

---

## 5. Cacheability matrix

| Tool / content | Customer-scoped? | Deterministic? | Cache? | TTL (proposed) |
|---|---|---|---|---|
| Policy Lookup | No | Yes | **Yes** | 24h |
| FAQ Lookup | No | Yes | **Yes** | 24h |
| Card Agreement | No | Yes | **Yes** | 24h |
| KYC Documents | No | Yes | **Yes** | 24h |
| RBI Rules | No | Yes | **Yes** | 24h |
| `get_customer_accounts` | Yes | No (mutable balances) | **No** | — |
| `get_recent_transactions` | Yes | No | **No** | — |
| `get_transaction_details` | Yes | No | **No** | — |
| `check_dispute_eligibility` | Yes | No (depends on live dispute state) | **No** | — |
| `create_dispute` | Yes | No (write) | **No** | — |
| JWT / auth calls | Yes | N/A | **No** | — |

---

## 6. Redis key design

```
semantic:query:{sha256(normalized_query)}       → cached response payload + metadata
semantic:vectors:{route}:{sha256(query)}         → query embedding, for similarity scan
embedding:{model}:{sha256(text)}                → embedding vector (JSON/msgpack)
tool:policy:{sha256(args)}                       → policy lookup result
tool:faq:{sha256(args)}                          → FAQ lookup result
tool:agreement:{sha256(args)}                    → card agreement result
tool:kyc:{sha256(args)}                          → KYC document result
tool:rbi:{sha256(args)}                          → RBI rules result
cache:version                                    → global schema/version string
```

- **Versioning**: every key is prefixed implicitly by a `cache:version` value baked into
  the key (e.g. `semantic:v2:query:{hash}`) so a corpus/schema change invalidates old
  entries by bumping the version rather than a flush.
- **TTL**: semantic and tool caches default 24h (policy documents change infrequently but
  do change); embedding cache TTL can be much longer (30d) since a given text's embedding
  never changes for a fixed model — but is still time-bounded, never "forever," per brief.
- **Invalidation**:
  - TTL expiry (baseline).
  - Version bump on policy corpus re-ingestion (`rag/ingest.py` already runs on corpus
    update — plan adds a step there to bump `cache:version`, invalidating semantic + tool
    caches at once without touching the embedding cache, which is keyed by model+text and
    stays valid across corpus changes).
  - Manual admin invalidation via the stats/cache API (flush by prefix) — exposed only as
    an internal capability for the demo, not public.

---

## 7. Savings estimate (demo assumptions — not measured production numbers)

**Assumptions** (stated explicitly, per brief):
- 1,000 requests/day.
- 40% are repeated/semantically-similar policy questions → semantic cache hit-eligible.
- 60% of embedding calls are reusable (repeated or overlapping query text/corpus chunks).
- 30% of deterministic tool calls hit the tool cache.
- Baseline (no cache): every request = 1 embedding call + 1–2 LLM calls (classifier +
  generation) + 1 retrieval + tool calls as routed.
- Approx per-request cost baseline (gpt-4o-mini @ ~$0.15/1M input, ~$0.60/1M output;
  text-embedding-3-small @ ~$0.02/1M tokens — **verify against current OpenAI pricing
  before quoting in the final submission**, per CLAUDE.md §4).

| Metric | Baseline | With caching | Reduction |
|---|---|---|---|
| LLM generation calls/day (policy route) | ~1,000 × policy-share | −40% of policy-route calls | Semantic cache eliminates generation entirely on hit |
| Embedding calls/day | ~1,000 | −60% | Embedding cache reuse |
| Tool calls/day (deterministic tools) | policy-tool volume | −30% | Tool cache |
| Avg latency (cache hit) | ~5–10s (full pipeline, NFR-3) | <1s (NFR-3 target) | ~85–90% on hits |
| Estimated token spend | 100% | ~55–65% (weighted by hit rates above, policy-route only) | ~35–45% overall reduction on the policy-route slice of traffic |

Full numeric table (with worked arithmetic per assumption, not just percentages) will be
computed in the documentation deliverable, not estimated loosely in prose — flagging that
as a to-do for the docs pass, not skipped here for space.

---

## 8. Performance comparison table (to be populated with the same assumptions)

| Metric | Current (Labs 1–6) | Optimized (Lab 7) |
|---|---|---|
| Avg LLM calls/request (policy route) | 2 (classify + generate) | ~1.2 (weighted by 40% hit rate) |
| Embedding calls/request | 1+ | ~0.4 (weighted by 60% reuse) |
| Tool cache hit rate | 0% (no cache) | ~30% (deterministic tools) |
| Semantic cache hit rate | 0% | ~40% (policy route only) |
| Avg latency (policy route) | ~5–10s | ~1s on hit, unchanged on miss |
| Estimated cost/1000 req (policy slice) | baseline | ~55–65% of baseline |
| Throughput (req/s, policy route) | pipeline-bound | cache-hit path is Redis-bound, materially higher |

---

## 9. Fallback / risk: Redis unavailable

- All three caches wrap every Redis call in a try/except that logs and falls through to
  the uncached path — matching the existing "never take down the app" posture already used
  for AgentOps (`agentops_client.py`) and Pinecone credential absence (`config.py`).
- `redis_enabled` config flag, default `False` for a machine with no Redis — same
  "off unless explicitly configured" convention as AgentOps/Pinecone.
- A Redis outage degrades to today's Lab 1–6 behavior exactly: no cache, correct answers,
  higher latency/cost. It never changes correctness.

---

## 10. Tests to add

- Semantic cache: hit (near-duplicate query), miss (dissimilar query), threshold boundary,
  bypass for customer-scoped route, bypass re-checked after execution (a policy-classified
  query that ends up calling a scoped tool is not stored) — this is the ADR-0006 governance
  test, extended to Redis.
- Embedding cache: hit (same text, same model), miss (new text), miss on model change
  (key includes model), TTL expiry.
- Tool cache: hit/miss per allowlisted tool, confirms scoped tools are never cached even if
  mistakenly passed the same cache wrapper (a negative/regression test), TTL expiry.
- Versioning: cache:version bump invalidates prior entries.
- Redis unavailable: every cache call fails soft, request still completes correctly
  (integration test with Redis client mocked/down).
- Regression: full existing Labs 1–6 suite still passes unmodified with `redis_enabled=False`
  (default) and with it `True` against a real local Redis.
- `GET /api/v1/cache/stats` returns correct counters.

---

## 11. Documentation updates

- `docs/decisions/0013-redis-caching-layer.md` (new ADR) — records SQLite→Redis decision,
  references and partially supersedes ADR-0006.
- `docs/architecture/architecture.md` — Lab 7 section: current + target diagrams (§1, §2
  above), Redis key design, cacheability matrix.
- `docs/requirements/project-requirements.md` — FR-7.x additions for Redis-specific
  behavior (cache stats endpoint, key versioning) if not already covered by existing FR-7.
- `docs/labs/lab-07-cost-optimization.md` (new) — the lab write-up: architecture, savings
  estimate, comparison tables, trade-offs, screenshots.
- `.env.example` — new Redis variables.

---

## 12. Screenshot plan

1. Redis running in Docker (`docker ps` / Docker Desktop).
2. Redis Insight browsing cache keys (if used).
3. Semantic cache hit — trace/log showing hit + latency.
4. Embedding cache hit — trace/log.
5. Tool cache hit — trace/log.
6. AgentOps trace showing new cache event types.
7. `GET /api/v1/cache/stats` response.
8. Streamlit UI cache indicators (hit/miss/source/latency) on a live response.
9. Current-architecture diagram.
10. Target-architecture diagram.
11. `pytest` run, all green, including new cache tests.
12. Self-review output (code-review skill).
13. Git diff/PR ready for approval.

---

## 13. Risks

- **Redis unavailable** — mitigated by fail-soft fallback (§9).
- **Cold cache** — first requests after a deploy see no benefit; expected, not a bug.
- **Cache invalidation** — TTL + version bump is coarse (invalidates everything on corpus
  change) rather than surgical per-document invalidation; acceptable at this scale, flagged
  as a known limitation.
- **Stale policy documents** — bounded by TTL (24h) even without an explicit re-ingestion
  event, so staleness is capped.
- **Memory usage** — bounded by TTL + key volume at demo scale; no eviction policy tuning
  planned beyond Redis defaults (`maxmemory-policy` noted as a config knob, not tuned here).
- **Similarity threshold tuning** — no vector-search module assumed (see Open Question);
  threshold and candidate-set size are demo-scale approximations, explicitly called out as
  such in the docs, not presented as production-grade ANN search.

---

## 14. Scope estimate

New/modified files: ~6 new modules (`caching/`), ~10 modified files, 1 new ADR, 1 new lab
doc, ~15–20 new tests, `docker-compose.yml`, `requirements.txt` bump. Comparable in size to
a single prior lab (e.g. Lab 5's guardrail layer). Estimated 1 implementation session
following approval.

---

## Open questions for approval

1. **Redis vs. the SQLite line in CLAUDE.md/ADR-0006** — confirm Redis is approved (§0).
2. **Vector similarity search**: plain Redis (Python-side cosine over a bounded candidate
   list) vs. requiring RediSearch/RedisVL for real vector search. Plain Redis is simpler and
   matches "Redis running locally in Docker" from the brief; RedisVL is more realistic
   enterprise architecture but is a bigger dependency add. Recommend **plain Redis +
   bounded candidate scan** for this lab's scope, revisit if hit-rate/latency looks wrong
   in testing.
3. **Similarity threshold default** — proposing `0.95` (cosine) as the starting
   configurable default; confirm or adjust.

---

**WAITING FOR HUMAN APPROVAL TO BEGIN LAB 7 IMPLEMENTATION**
