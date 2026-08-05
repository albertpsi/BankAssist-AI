# ADR-0012 — AgentOps for Lab 6 operational observability, instead of a custom dashboard

**Status:** Accepted
**Date:** 2026-08-04
**Relates to:** [ADR-0004](0004-custom-tracing-and-evaluation.md) (superseded, tracing half
only — the evaluation half stands unchanged), [ADR-0009](0009-langgraph-agent-orchestration.md)
(LangGraph orchestration, directly auto-instrumented),
[ADR-0011](0011-nemo-guardrails-for-ai-semantic-rails.md) (guardrail architecture,
custom-spanned at its verdict boundary)

## Context

Lab 6 asks for the system's agentic execution — sessions, agent hops, LLM calls, tool
calls, latency, tokens, cost, and errors — to be inspectable from an AI-operational
observability platform, not reconstructed by hand from logs or a second in-house
dashboard. The project already has two observability-shaped systems from earlier labs:
`Span`/`Tracer` (`src/bankassist/tracing/`, wired only around OpenAI calls, never read
by any UI), and `ExecutionEvent` (the BankAssist-branded workflow visualization the
Streamlit UI renders for the demo audience). Neither answers "how did the agentic
system execute" at the level Lab 6 requires: neither shows nested LLM/tool spans, token
counts, cost, or per-node latency in a form an operator would actually use to diagnose a
production agent.

## Decision

Adopt **AgentOps (`agentops` Python SDK, verified at v0.4.21, Python 3.9–3.13, PyPI
`agentops`)** as the sole AI-operational observability platform for Lab 6, instead of
building a second, custom dashboard that would functionally reproduce it.

- `agentops.init()` runs once, in `create_app()`'s startup path
  (`src/bankassist/api/app.py`), gated on `Settings.agentops_enabled` and a non-blank
  `AGENTOPS_API_KEY` — off unless both are explicitly configured, same posture as
  Pinecone (`Settings.has_pinecone_credential`).
- AgentOps' native LangGraph and OpenAI auto-instrumentation is used as-is: no
  hand-rolled node/edge/state tracking, no manual wrapping of every OpenAI call.
- A small, deliberate set of custom spans (`src/bankassist/observability/decorators.py`,
  exposing `operation`/`agent_span`/`tool_span`/`workflow`/`trace`/`run`) covers the
  BankAssist-specific boundaries automatic instrumentation cannot see on its own: the
  supervisor's routing decision, each Enterprise RAG stage, each scoped tool call, each
  guardrail verdict, the HITL pause/resume moment, and the whole-request boundary.
- All AgentOps integration is isolated inside `src/bankassist/observability/` — no other
  module imports `agentops` directly (verified: `grep -rn "^import agentops\|^from
  agentops"` outside that package returns nothing).

## Responsibility boundary

```
ExecutionEvent   → BankAssist-facing workflow visualization → "What happened?"
Existing logging → application diagnostics
AgentOps         → AI operational observability            → "How did it execute?"
Evaluation       → quality measurement                      → "Was it correct?"
```

`ExecutionEvent` is unchanged by this ADR — it is not replaced, extended, or duplicated.
`Span`/`Tracer` is superseded for observability purposes: it wrapped exactly one call
site (`llm/openai_client.py`) and nothing ever read `Tracer.spans()`. AgentOps' OpenAI
auto-instrumentation now satisfies the "every LLM call records model, tokens, latency"
requirement (CLAUDE.md §8) at the platform level. `Span`/`Tracer` is left in place
(nothing calls for its removal, and the request-level `X-Trace-Id` header/log
correlation it enables still has value), but it is not extended for Lab 6 and should be
considered legacy rather than dual-maintained.

## Privacy / redaction

AgentOps' current documentation surfaces no built-in PII redaction or payload-masking
feature (checked directly, not assumed from memory — see the Lab 6 planning message in
this session for the sources consulted). Its own OpenAI/LangGraph auto-instrumentation
is AgentOps' concern, not ours, and is exactly what Lab 6 wants captured. For the
*custom* attributes BankAssist code attaches (route, tool name, guardrail verdict,
latency), `src/bankassist/observability/redaction.py` sanitizes every value through the
project's existing deterministic `guardrails.redaction.redact` /
`guardrails.masking.mask_sensitive_identifiers` patterns before it is sent, and drops
any attribute whose *key* looks credential-shaped outright (JWT, token, API key,
password, secret, system prompt) using whole-word matching — a naive substring check
was caught in self-review dropping legitimate attributes like `output_tokens` (contains
"token") and fixed before this was reported.

## Alternatives considered

**Build a second observability dashboard in Streamlit, reading `Span`/`ExecutionEvent`
directly.** Rejected per the lab brief's explicit instruction: this would reproduce
AgentOps rather than adopt it, and would need to reinvent token/cost/latency
aggregation AgentOps already provides well.

**LangSmith, Phoenix, or an OpenTelemetry Collector + Jaeger/Grafana stack.** Rejected:
the lab brief names AgentOps specifically, and stacking multiple observability
platforms for one lab is scope creep the project's own principles reject (CLAUDE.md §3,
"Scope discipline").

**Wrap every function in the codebase with a custom span decorator for maximal trace
detail.** Rejected: automatic instrumentation already covers LLM/LangGraph internals;
adding a span to every helper function would produce an impressive-looking but noisy
trace without adding diagnostic value, and was explicitly out of scope per the lab
brief ("do not manually wrap every single function").

## New dependency

- `agentops>=0.4.21` — installed cleanly into the project's Python 3.13 virtualenv
  alongside the full existing dependency set (`nemoguardrails`, `langgraph`,
  `langchain-openai`, `pinecone`, `sentence-transformers`); no version conflicts
  observed on `pip install`.

## Consequences

**Makes easy:** sessions, nested agent/tool/LLM spans, token counts, cost, and latency
are visible in a purpose-built dashboard with two lines of integration code
(`agentops.init()` + native LangGraph/OpenAI instrumentation) rather than a
hand-maintained second UI; the custom span layer is small and targeted, not a
wrap-everything exercise.

**Makes hard / accepted limits:** telemetry now leaves the local application boundary
to a third-party SaaS whenever `AGENTOPS_ENABLED=true` — a real trade-off for a banking
teaching application, mitigated by the redaction layer and by AgentOps being off by
default. The integration is vendor-specific: swapping observability platforms later
means re-touching `src/bankassist/observability/` (one package, by design) rather than
scattered call sites. AgentOps requires network reachability and a valid API key to do
anything; `fail_safe=True` plus this project's own try/except-and-fall-through pattern
in every decorator ensures an AgentOps outage degrades to "no telemetry," never to an
application failure — verified in self-review, where an overly broad exception handler
around the wrapped function call itself (not just around acquiring the AgentOps
decorator) was found to risk **double-invoking** a mutating tool call like
`create_dispute` if the tool raised while AgentOps was enabled, and was fixed with a
regression test before this was reported as done.

## Known upstream bug and workaround

Confirmed against a real, running instance of this application with
`AGENTOPS_ENABLED=true`: `agentops==0.4.21`'s LangGraph auto-instrumentation
(`agentops.instrumentation.agentic.langgraph.instrumentation.LanggraphInstrumentor
._wrap_add_node`) replaces every graph node function with a wrapper accepting only
`(state)`, silently dropping `config`. Every BankAssist node accepts `(state, config)`
— the standard, documented way a LangGraph node reads `config["configurable"]` (this
project's `SecurityContext`, ADR-0010) — so **every single graph invocation raised
`TypeError: ...got an unexpected keyword argument 'config'`** with AgentOps enabled.
This is not BankAssist-specific: any LangGraph application whose nodes read `config`
hits the same bug.

A first fix attempt called `instrumentor.uninstrument()` immediately after
`agentops.init()`. That did not hold under reproduction: AgentOps installs a global
`builtins.__import__` hook that re-scans and silently re-instruments LangGraph on the
*next* relevant import, regardless of an explicit uninstall call. The working fix
(`src/bankassist/observability/agentops_client.py:_exclude_broken_langgraph_node_instrumentation`)
instead removes `"langgraph"` from AgentOps' `AGENTIC_LIBRARIES`/`TARGET_PACKAGES`
registries **before** `agentops.init()` runs, so LangGraph is never targeted for
instrumentation in the first place — verified by reproducing the original `TypeError`
against a real `graph.invoke()` call on both the policy and dispute-agent paths, then
confirming it no longer occurs with the workaround applied.

**Consequence:** AgentOps' *native* LangGraph node/edge/execution-path auto-detection
(Lab 6 requirements §6, "USE IT" if the SDK provides it) is not usable with this
project's node signature as of `agentops==0.4.21`. OpenAI auto-instrumentation and
every custom span in `observability/decorators.py` (which wrap functions directly, not
via LangGraph's `add_node`) are unaffected and are what actually makes the multi-agent
architecture visible in the AgentOps dashboard for this lab — the supervisor routing
span, the RAG-stage spans, the tool spans, the guardrail spans, and the whole-request
trace collectively reconstruct the hierarchy the native instrumentation would have
shown automatically, had it worked.

## Revisit when

Re-test the native LangGraph auto-instrumentation against a newer `agentops` release —
remove the exclusion workaround once the upstream `config`-argument bug is fixed.
AgentOps ships a documented redaction/masking configuration option — at that point,
evaluate whether it can replace or simplify the hand-rolled sanitation in
`observability/redaction.py`. Also revisit if Lab 7's cost-optimization work needs
per-call cost data at a granularity AgentOps' dashboard does not expose, at which point
a targeted export (not a second dashboard) would be the next incremental step.
