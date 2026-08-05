# Lab 6 — AgentOps observability + evaluation

**Status:** Implemented, pending Gate 2 (git/PR) approval.
**Relates to:** [ADR-0012](../decisions/0012-agentops-observability.md) (supersedes the
tracing half of [ADR-0004](../decisions/0004-custom-tracing-and-evaluation.md))

## Problem

Labs 1–5 built `Span`/`Tracer` (never surfaced anywhere) and `ExecutionEvent` (the
BankAssist-facing workflow visualization). Neither answers Lab 6's actual question —
"how did the agentic system execute": nested LLM/tool spans, token counts, cost,
per-node latency, and errors, inspectable the way an operator would actually use them.
There was also no evaluation subsystem measuring whether the system's answers, routing,
tool selection, retrieval, and guardrail decisions are *correct* — only that they
*happened* (`ExecutionEvent`) or that individual functions behave as unit-tested.

## Objective

AgentOps becomes the operational observability system: sessions, agent execution, LLM
calls, tool calls, latency, tokens, cost, and errors are inspectable from its dashboard.
A small, curated golden evaluation dataset and deterministic scorers measure
application quality (routing, tool selection, retrieval, citations, guardrail
block/allow rates) separately from AgentOps' operational telemetry, and a failed
evaluation case correlates to its AgentOps trace for diagnosis.

## Functional requirements

- **FR-1** `agentops.init()` runs once at FastAPI startup, gated on
  `Settings.agentops_enabled` and a non-blank `AGENTOPS_API_KEY`; off by default.
- **FR-2** AgentOps' native LangGraph auto-instrumentation is enabled by default,
  **with one confirmed exception**: `agentops==0.4.21`'s node-wrapping is incompatible
  with LangGraph nodes that accept a `config: RunnableConfig` parameter (this project's
  `SecurityContext`/HITL pattern, ADR-0010) and crashes every graph invocation until
  excluded. `observability/agentops_client.py` disables only that specific broken piece
  before `agentops.init()` runs; OpenAI auto-instrumentation and the custom span layer
  (FR-4) are unaffected and are what actually make the multi-agent hierarchy visible in
  this lab. See ADR-0012, "Known upstream bug and workaround", for the full
  reproduction and fix.
- **FR-3** AgentOps' native OpenAI auto-instrumentation captures every LLM call's model,
  latency, input/output/total tokens, and cost without per-call-site changes to
  `LLMClient`/`OpenAIClient`.
- **FR-4** Custom AgentOps spans exist for boundaries automatic instrumentation cannot
  see: the supervisor's routing decision, each Enterprise RAG stage (classify, rewrite,
  vector retrieval, BM25 retrieval, RRF, rerank, generation), each of the five scoped
  tool calls (`get_customer_accounts`, `get_recent_transactions`,
  `get_transaction_details`, `check_dispute_eligibility`, `create_dispute`), each
  guardrail verdict (NeMo input/output rail), and the whole `/agent/chat` /
  `/agent/resume` request boundary.
- **FR-5** The HITL pause/resume moment is marked on trace metadata (a LangGraph
  `interrupt()` spans two separate graph invocations, so it cannot be a single function
  span).
- **FR-6** Custom trace/span attributes are sanitized before being sent: credential-shaped
  keys are dropped outright; string values are passed through the existing deterministic
  `redact`/`mask_sensitive_identifiers` patterns.
- **FR-7** All AgentOps integration is isolated in `src/bankassist/observability/`; no
  other module imports `agentops` directly.
- **FR-8** Every AgentOps call site is a no-op pass-through when AgentOps is disabled —
  verified by the full existing Labs 1–5 test suite passing unmodified, plus dedicated
  observability unit tests, all without network access.
- **FR-9** A golden evaluation dataset (`evaluation/golden_dataset.yaml`) of ~20–25 cases
  spans policy RAG, banking agent, dispute agent, routing, multi-turn, and
  security/guardrail (must-block / must-allow) categories.
- **FR-10** Deterministic metrics score each case: retrieval (Hit@K, Recall@K, MRR),
  generation (response exists, refusal match, citation match, forbidden content absent),
  agent (routing accuracy, tool selection, approval gating, mutation-never-precedes-
  approval), and guardrail (attack block rate, legitimate allow rate).
- **FR-11** `evaluation/runner.py` executes the dataset against an injected `Executor`,
  scores every case, and never aborts on one case's failure.
- **FR-12** `evaluation/executor.py` (`GraphExecutor`) drives the real, running
  application through its actual HTTP boundary (`/agent/chat`, `/agent/resume`) for a
  live run — not a shortcut that bypasses guardrails or routing.
- **FR-13** `evaluation/report.py` renders a compact Markdown report: totals, routing
  accuracy, tool selection accuracy, retrieval Hit@K/MRR, citation accuracy, attack
  block rate, legitimate allow rate, average latency — only metrics with applicable
  cases, never a fabricated value.
- **FR-14** LLM-as-judge is **not** implemented in this pass (see Out of scope) —
  deterministic checks cover every case in the current dataset.

## Non-functional requirements

- **NFR-1 (determinism)** The default `pytest` run never depends on AgentOps network
  availability or a live LLM; `evaluation/` unit tests use a stub `Executor`, matching
  the `StubLLMClient` pattern used throughout the rest of the suite.
- **NFR-2 (safety)** An AgentOps SDK failure at any point degrades to "no telemetry,"
  never to an application failure — `fail_safe=True` plus a defensive try/except around
  acquiring (not calling) the AgentOps decorator in every custom span helper.
- **NFR-3 (no double-execution)** A wrapped function is called exactly once regardless
  of whether AgentOps is enabled, disabled, or itself raises — critical for the one
  mutating tool call, `create_dispute` (caught and fixed in self-review; see the change
  report).
- **NFR-4 (privacy)** No JWT, API key, Authorization header, secret, full card number,
  full account number, or raw system prompt is ever attached as custom AgentOps
  metadata.
- **NFR-5 (observability)** The Labs 1–5 test suite, `ruff`, and app startup all remain
  green with this change.

## Assumptions

- The demo/synthetic dataset (`scripts/seed_banking_data.py`, customer `CUST001`) is
  used by `GraphExecutor` for live evaluation runs; no new synthetic data was needed.
- AgentOps' documented public API (v0.4.21, verified this session against
  `docs.agentops.ai/v2` and the project's `README.md`) is stable enough to build
  against; no private/undocumented API is used.

## Out of scope

- Grafana, Prometheus, Jaeger, ELK, Kafka, an OpenTelemetry Collector, LangSmith, or
  Phoenix alongside AgentOps.
- Replacing or extending `ExecutionEvent` or the Streamlit workflow UI.
- LLM-as-judge scoring (optional per the lab brief; deferred — the current dataset is
  fully covered by deterministic checks, keeping Lab 6 from growing past its intended
  size).
- Lab 7 optimization (prompt caching, semantic caching, model-tier routing, before/after
  cost comparison) — this lab only establishes the baseline AgentOps and the evaluation
  report capture.
- Running `scripts/run_evaluation.py` against a live OpenAI/Pinecone/AgentOps account
  and capturing dashboard screenshots — that is the next step after this report, and
  needs the user's real credentials (see "Screenshot checkpoints" below).

## Acceptance criteria

- **AC-1** `agentops>=0.4.21` is declared in `requirements.txt` and installs cleanly
  alongside the existing dependency set.
- **AC-2** `AGENTOPS_ENABLED`, `AGENTOPS_API_KEY`, `AGENTOPS_PROJECT`,
  `AGENTOPS_ENVIRONMENT` are documented in `.env.example`, defaulting to disabled.
- **AC-3** `create_app()` starts successfully with AgentOps disabled (default) and with
  no `AGENTOPS_API_KEY` set.
- **AC-4** The full Labs 1–5 automated test suite passes unmodified.
- **AC-5** New unit tests cover: AgentOps init no-op/enable/failure paths, decorator/`run`
  pass-through and dynamic-name spans, the double-execution regression, and attribute
  redaction (including the token-vs-tokens false-positive regression).
- **AC-6** `evaluation/golden_dataset.yaml` has 20–25 cases covering policy RAG, banking
  agent, dispute agent, routing, multi-turn, must-block, and must-allow categories.
- **AC-7** `evaluation/runner.py` and `evaluation/metrics/*` are covered by deterministic
  unit tests exercising at least one case per metric family, with no network access.
- **AC-8** `ruff check .` is clean across the full repository.
- **AC-9** An ADR (0012) documents the AgentOps adoption decision, including trade-offs,
  and updates ADR-0004's status to reflect the superseded scope.
- **AC-10** `POST /agent/chat` completes successfully (no `TypeError`) with
  `AGENTOPS_ENABLED=true` against a real graph invocation on both the policy and
  dispute-agent (HITL) paths — verified directly, not just unit-tested, since this
  regression was only caught by running the live app.

## Screenshot checkpoints (next step, requires live credentials)

Per the approved plan, once `AGENTOPS_API_KEY` (and a live `OPENAI_API_KEY`, and
optionally a populated Pinecone index) are available:

1. Set `AGENTOPS_ENABLED=true` and a real `AGENTOPS_API_KEY` in `.env`.
2. Run the app (`uvicorn` or `scripts/run_api_dev.py`) and drive a few chat turns —
   policy question, banking summary, and a full dispute HITL flow — through the
   Streamlit UI or directly against `/api/v1/agent/chat`.
3. Open [app.agentops.ai](https://app.agentops.ai) and capture: project overview,
   session/trace list, one complete multi-agent trace, the supervisor routing decision,
   a tool invocation trace, a RAG pipeline trace, token/cost for one LLM call, latency
   breakdown, a guardrail-blocked request, and the HITL dispute trace.
4. Run `python scripts/run_evaluation.py` for the evaluation report and, if a case
   fails, its AgentOps trace id (or find it in the dashboard by session name).
