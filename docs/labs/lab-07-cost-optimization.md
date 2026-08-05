# Lab 7 — Cost Optimization Architecture

**Status:** Implemented, tested, self-reviewed. Screenshot evidence is pending local
capture (Docker Desktop / RedisInsight are not available in the implementation
environment) — see §6.

Design documents: [ADR-0013](../decisions/0013-redis-caching-layer.md) (Redis decision),
[ADR-0006](../decisions/0006-semantic-cache-eligibility.md) (eligibility policy, reused
unchanged), [architecture.md §14](../architecture/architecture.md#14-lab-7--cost-optimization-architecture-adr-0013),
[planning artifact](../plan/lab-07-cost-optimization-plan.md) (approved with amendments).

## 1. What was built

Three Redis-backed caches under `src/bankassist/caching/`:

- **Semantic response cache** (`semantic_cache.py`) — embed → RediSearch KNN (or a
  documented Python cosine fallback) → threshold → serve/store. Eligibility reuses
  ADR-0006's rule via `eligibility.py`'s `classify_eligibility`.
- **Embedding cache** (`embedding_cache.py`) — `SHA256(model + text)` → Redis, wraps
  `rag/embeddings.py::OpenAIEmbedder` as an optional collaborator.
- **Tool response cache** (`tool_cache.py`) — versioned, argument-hash keys, a positive
  allowlist (`TOOL_CACHE_ALLOWLIST`) of deterministic, non-customer tools; wired into
  `tools/dispatcher.py::call_tool` as optional parameters.

Integration points: `agents/graph.py`'s `policy_node` (semantic cache), `api/app.py`
(builds the shared Redis client + caches once per process), `api/routes/agent.py` and
`api/routes/rag.py` (thread the shared embedding cache and semantic cache through),
`api/routes/cache.py` (`GET /api/v1/cache/stats`), `ui/agentic_app.py` (Optimization
Summary panel).

Every cache is optional at every layer: `REDIS_ENABLED=false` (default) or an
unreachable Redis produces the exact Labs 1–6 behavior, never an exception.

## 2. Cacheability matrix

| Tool / content | Customer-scoped? | Cache? |
|---|---|---|
| Policy / FAQ / Card Agreement / KYC / RBI lookups | No | **Yes** (`TOOL_CACHE_ALLOWLIST`) |
| `get_customer_accounts`, `get_recent_transactions`, `get_transaction_details` | Yes | **No** |
| `check_dispute_eligibility`, `create_dispute` | Yes | **No** |

## 3. Test coverage added

| Area | File |
|---|---|
| Eligibility classification (all route/tool-invoked combinations, `SESSION_CACHEABLE` never produced) | `tests/unit/caching/test_eligibility.py` |
| Embedding cache (model-bound keys, hit/miss, Redis-down fallback) | `tests/unit/caching/test_embedding_cache.py`, `tests/unit/test_rag_embeddings.py::TestEmbeddingCacheIntegration` |
| Tool cache (argument-hash keys, versioning, allowlist enforcement, TTL) | `tests/unit/caching/test_tool_cache.py`, `tests/unit/tools/test_dispatcher_tool_cache.py` |
| Semantic cache (hit/miss, threshold boundary, ADR-0006 re-check-before-store, RediSearch-unavailable fallback) | `tests/unit/caching/test_semantic_cache.py` |
| Redis client (disabled, unreachable, RediSearch detection) | `tests/unit/caching/test_redis_client.py` |
| Stats aggregation | `tests/unit/caching/test_stats.py` |
| `GET /api/v1/cache/stats` | `tests/integration/test_cache_stats_api.py` |
| Full Labs 1–6 regression | entire existing suite, run unmodified |

`fakeredis` (a real Redis protocol implementation, in-memory) is used throughout rather
than mocks — it also happens to lack `MODULE LIST`/`FT.*`, so it doubles as the exact
"RediSearch unavailable" scenario the semantic-cache fallback path needs.

**Result at last run:** full suite green (`pytest`, no `-k`/skip), `ruff check` clean
across `src/` and `tests/`.

## 4. Real before/after benchmark

`scripts/lab7_benchmark.py` runs the actual `policy_node` code path (via
`agents.graph.build_graph`) N times with a repeated question, once with the semantic
cache disabled and once enabled, against `StubLLMClient` and a fake enterprise pipeline
(no live OpenAI calls — reproducible, free to run). A sample run (`--requests 15`):

```
BEFORE (no semantic cache): 15/15 pipeline invocations, mean latency 55.7 ms
AFTER  (semantic cache enabled): 1/15 pipeline invocations, mean latency 8.1 ms
Measured pipeline calls avoided: 14 / 15
Measured mean latency reduction: 47.7 ms
```

The **relative** effect (calls avoided, latency drop) is real and measured; the
**absolute** stub latency stands in for a real generation call, so it is not a
production number. Dollar-cost figures are the demo assumptions from §5 applied to the
exact measured hit count, not derived independently.

## 5. Extrapolated savings estimate (demo assumptions, not measured production traffic)

See [the planning artifact §7–8](../plan/lab-07-cost-optimization-plan.md#7-savings-estimate-demo-assumptions--not-measured-production-numbers)
for the full worked table under stated assumptions (1,000 req/day, 40% repeated policy
questions, 60% embedding reuse, 30% tool cache hits). Those percentages are declared
assumptions, not measurements — the benchmark in §4 above is the measured evidence;
this section estimates what it implies at demo-scale daily volume.

## 6. Screenshot plan (pending local capture)

The implementation environment has no Docker Desktop / RedisInsight / live OpenAI
credential, so the following require the user to run locally and capture:

1. Redis running in Docker (`docker compose up redis`, then `docker ps`).
2. RedisInsight (or `redis-cli --scan`) showing `semantic:vec:*`, `embedding:*`,
   `tool:*` keys after a few real requests.
3. A semantic cache hit in the API logs / trace (repeat a policy question twice).
4. An embedding cache hit.
5. A tool cache hit (once a real deterministic tool call exists in a future lab —
   today's cacheability matrix has no live call site, see §7).
6. An AgentOps trace showing the new cache event types.
7. `GET /api/v1/cache/stats` response.
8. Streamlit's Optimization Summary panel on a live turn.
9. `pytest` full run, green.
10. This self-review's findings.

## 7. Known limitation

The application does not yet have a standalone, non-customer-scoped "Policy Lookup /
FAQ Lookup / Card Agreement / KYC / RBI Rules" **tool** distinct from the Enterprise RAG
pipeline itself — policy answers are generated end-to-end by `EnterpriseRagPipeline`,
not fetched via a discrete tool call. `ToolCache` and `TOOL_CACHE_ALLOWLIST` are fully
implemented, tested, and wired into `tools/dispatcher.py::call_tool`, but have no live
call site in the current agent graph to exercise in production — only the semantic
cache (on the full policy answer) is active end-to-end today. This was flagged during
implementation rather than worked around by inventing a tool the rest of the
architecture doesn't otherwise need (`CLAUDE.md` §3 scope discipline); it is a natural
fit if/when a future lab splits retrieval into named, cacheable lookup tools.

## 8. Explicitly out of scope (per approved plan and ADR-0013)

Prompt caching / provider-native prompt caching, model routing, prompt/context
compression, token budget manager.
