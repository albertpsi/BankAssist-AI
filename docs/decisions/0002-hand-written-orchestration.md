# ADR-0002 — Hand-written agent orchestration over a framework

**Status:** Proposed
**Date:** 2026-08-03

## Context

Lab 4 requires a Supervisor agent routing to Policy, Banking, and Dispute specialists, with
traced handoffs and scoped tool use. Labs 5, 6, and 7 then layer guardrails, tracing, and
cost accounting *through* that orchestration.

Mature frameworks exist — LangGraph, CrewAI, AutoGen — that supply the supervisor pattern,
the tool-calling loop, and state management out of the box.

## Decision

Write the orchestration ourselves: a supervisor that classifies intent and routes, and an
agent base class running a bounded tool-calling loop. Roughly 200 lines of first-party code,
no orchestration framework.

## Alternatives considered

**LangGraph** — the closest fit; graph-based orchestration with checkpointing and a
supervisor pattern. Rejected because the graph abstraction sits directly on top of the three
things Labs 5–7 are *about*. Tracing becomes "whatever LangGraph emits, plus our own", the
guardrail hooks have to fit its node model, and the cost accounting has to be recovered from
its callbacks. The write-up would then be explaining LangGraph's execution model rather than
agent architecture.

**CrewAI** — role-based agents with a friendly API. Rejected: opinionated about agent
personas and delegation in ways that don't match a supervisor-with-scoped-tools design, and
its abstractions are further from the trace we need.

**AutoGen** — conversational multi-agent. Rejected: the conversation-between-agents model
is a poor fit for deterministic routing, and it is harder to bound.

**Anthropic SDK tool runner** — would handle the loop within one agent. A reasonable middle
ground, and its per-turn hooks would support the guardrail gating. Rejected for the *inter*-
agent layer, which it does not cover; adopting it only for the intra-agent loop would mean
two loop implementations with different trace shapes. May be revisited if the hand-written
loop proves fiddly.

## Consequences

**Makes easy:** exact control over the trace (every hop, reason, and tool call is ours to
record), guardrails placed precisely where we want them, loop bounds we can prove in tests,
and a write-up that explains the architecture directly rather than through a dependency.
Also removes a large transitive dependency tree.

**Makes hard:** we reimplement routing, state, retry, and loop bounds. There is no ecosystem
tooling, no community-maintained patterns, and no checkpointing/resumability — a failure
mid-dispute loses the flow.

**Accepted:** ~200 lines of code we own and must test, in exchange for a system whose every
hop is visible.

## Revisit when

The agent graph grows beyond ~5 agents or needs conditional cycles, durable resumability
becomes a requirement, or the hand-written loop starts accumulating framework-shaped
complexity of its own.
