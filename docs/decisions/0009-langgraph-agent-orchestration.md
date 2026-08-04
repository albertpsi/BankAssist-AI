# ADR-0009 — Adopt LangGraph for Lab 4 multi-agent orchestration

**Status:** Proposed
**Date:** 2026-08-04
**Supersedes:** [ADR-0002](0002-hand-written-orchestration.md) (hand-written orchestration)

## Context

ADR-0002 chose hand-written orchestration for the Lab 4 Supervisor/Policy/Banking/
Dispute agents specifically because, at that point, no agent needed genuine suspension:
every tool was read-only, there were no conditional cycles, and durable resumability
was explicitly listed as *not* a requirement.

Lab 4's brief now requires:

- A **state-changing** tool (`create_dispute`) that must pause mid-workflow for human
  approval and resume later — a real suspend/resume, not a same-turn "if message == yes"
  check.
- **Session-scoped, checkpointed state** across multiple user turns (multi-turn
  transaction reference resolution).
- A **visual execution graph** driven by real orchestration events, not a hand-rolled
  approximation of one.

These three requirements are exactly the ones ADR-0002 flagged as the trigger to revisit
("durable resumability becomes a requirement" — see its "Revisit when" section). That
condition is now met.

## Decision

Adopt **LangGraph** as the orchestration layer for Lab 4, scoped strictly to
orchestration: routing, state threading, and interrupt/resume. All business logic —
retrieval (`EnterpriseRagPipeline`), SQL access, customer scoping, dispute eligibility
rules, and masking — stays first-party, exactly where it already lives. LangGraph nodes
are thin: they call into existing `bankassist` services/tools and translate the result
into state updates and `ExecutionEvent`s.

Core rule carried into implementation: **LangGraph owns orchestration; BankAssist owns
business logic.** SQL, retrieval, validation, scoping, and dispute rules do not move
into graph node bodies merely because LangGraph is introduced.

## Why Lab 4 is the right boundary

Lab 4 is the first point in the roadmap where the system needs a real suspend/resume
primitive and multi-turn checkpointed state. Introducing LangGraph earlier (Labs 2–3)
would have added graph machinery with nothing that needed suspension. Introducing it
later (Lab 5+) would mean guardrails, tracing, and cost accounting (Labs 5–7) get built
against the hand-written loop and then have to be retrofitted onto LangGraph's node
model — strictly more rework than building them against LangGraph's model once, now,
before they exist.

## Alternatives reconsidered

**Keep hand-written orchestration, add interrupt/resume ourselves.** Possible — a queue
row plus a "paused" state flag can simulate suspension. Rejected: this is reinventing a
narrower, untested version of exactly what LangGraph provides, for the one requirement
(HITL interrupt) that most benefits from a library's tested semantics. It also does
nothing for the checkpointed multi-turn state requirement, which would need its own
mechanism anyway.

**CrewAI.** Still rejected for the reasons in ADR-0002: role/persona-based delegation
does not match a supervisor-with-scoped-tools design, and it has no first-class
interrupt/resume primitive comparable to LangGraph's.

**AutoGen.** Still rejected: conversational multi-agent handoff is a poor fit for
deterministic routing and is harder to bound than explicit conditional edges.

**Anthropic SDK tool runner.** Still a reasonable *intra-agent* tool loop, but does not
cover inter-agent routing, shared graph state, or interrupt/resume across a supervisor
and specialists — the exact gap ADR-0002 already identified.

## What LangGraph is used for (and only this)

- `StateGraph` over a typed `BankAssistState`.
- `add_conditional_edges` from the Supervisor node on `state.route`.
- Built-in checkpointer (`MemorySaver`, in-memory) keyed by `session_id` as `thread_id`,
  for session-scoped multi-turn state — not durable cross-session memory.
- `interrupt()` inside a `prepare_dispute` node and `Command(resume=...)` to continue,
  for the one state-changing tool.

## What stays first-party (unchanged from ADR-0002's intent)

- `EnterpriseRagPipeline` and all Lab 3 retrieval stages — reused, not reimplemented.
- SQLite-backed scoped tools and customer-ownership checks.
- Dispute eligibility rules and the dispute state machine's business rules.
- Tracing (`bankassist.tracing`) — the existing `Tracer`/`Span` model keeps recording
  LLM calls, retrieval, and tool execution exactly as it does today; `ExecutionEvent` is
  a separate, deliberately smaller, UI-facing model (see "Impact on tracing" below).
- The `LLMClient` interface — LangGraph nodes call structured-output requests through
  it exactly as any other agent code would; no LangChain LLM wrapper is introduced.

## Trade-offs accepted

**Makes easy:** genuine suspend/resume with tested semantics; typed shared state across
agents without hand-rolled plumbing; conditional routing expressed declarively;
visualization has a real backing event source (graph execution) rather than a simulated
one.

**Makes hard / gives up:** the exact-control trace story ADR-0002 valued is now split —
LangGraph's own execution drives graph structure, while `bankassist.tracing` and the new
`ExecutionEvent` model still do the recording BankAssist controls. This is deliberately
bridged by an event-emission convention (nodes/tools explicitly emit `ExecutionEvent`s;
nothing is inferred from LangGraph internals or from the final answer text — Lab 4 brief
§22), so the write-up still explains *our* architecture, not LangGraph's internals, even
though LangGraph is now doing the suspension. A larger transitive dependency
(`langgraph` + its `langgraph-checkpoint` dependency) is added; both are widely used,
non-typosquat packages, consistent with CLAUDE.md §6's dependency check.

## Impact on tracing (ADR-0004) and future labs

`bankassist.tracing.Tracer`/`Span` is untouched — it keeps recording LLM calls,
retrieval, and tool spans exactly as Labs 1–3 left it, called from inside the same
service/tool code the graph nodes invoke. `ExecutionEvent` is a new, intentionally
thinner model whose only job is answering "what executed, for this UI" (Lab 4 brief
§14, §29). Lab 6 is expected to extend `ExecutionEvent` (or correlate it with trace
spans by `request_id`) to add latency/cost/quality — that extension point is designed
in now, not implemented now.

## Testing strategy

- Unit-test each graph node's business-logic call in isolation with a stubbed
  `LLMClient` and stubbed tools (no live LLM, per CLAUDE.md §9).
- Integration-test full graph runs for each of the five demo scenarios (policy,
  banking, dispute-approve, dispute-reject, cross-customer rejection), asserting on
  `state.execution_events` order/content, not on log scraping.
- Dedicated tests for: interrupt is actually raised (graph halts before
  `create_dispute` runs), resume applies exactly once, a second resume against a
  resolved interrupt is rejected, and checkpointed state survives across two API calls
  with the same `session_id`.
- Existing Lab 1–3 suite runs unmodified as a regression gate.

## Consequences

**Revisit when:** LangGraph's checkpointer needs to become durable across process
restarts (would then need Postgres/Redis — explicitly deferred per the Lab 4 brief), or
guardrail integration (Lab 5) turns out not to fit cleanly as pre/post hooks around graph
nodes.
