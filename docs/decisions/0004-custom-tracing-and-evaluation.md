# ADR-0004 — Custom tracing and evaluation over hosted AgentOps tools

**Status:** Superseded by [ADR-0012](0012-agentops-observability.md) (tracing/observability
half only) — 2026-08-04
**Date:** 2026-08-03

> **2026-08-04 update:** the tracing half of this decision is reversed by
> [ADR-0012](0012-agentops-observability.md): AgentOps is explicitly the platform
> requested for Lab 6, and its LangGraph/OpenAI auto-instrumentation makes "outsourcing
> the observability layer removes the thing being graded" (this ADR's rejection
> reasoning below) no longer the operative concern — the custom instrumentation layer
> at BankAssist-specific boundaries *is* the graded implementation detail, on top of
> AgentOps rather than instead of it. The **evaluation** half of this ADR stands:
> `evaluation/` (Lab 6) is still a first-party deterministic-scorer harness, not a
> hosted evaluation product — see `docs/requirements/lab-06-agentops-evaluation.md`.
> `Span`/`Tracer` (the tracer this ADR specified) is left in place but not extended
> further; ADR-0012 explains why.

## Context

Lab 6 requires capturing trace id, agent execution, agent handoffs, LLM calls, tool calls,
RAG retrievals with retrieved documents, guardrail interventions, latency, token usage,
errors, and evaluation scores — and then evaluating answer relevance, groundedness, context
relevance, citation correctness, guardrail success rate, task completion, latency, tokens,
and cost.

LangSmith, Arize Phoenix, Langfuse, RAGAS, and DeepEval all address parts of this.

## Decision

Build a custom span tracer that writes JSONL to disk, with an OpenTelemetry-shaped span
model (`span_id`, `parent_span_id`, `trace_id`, `type`, timing, status, typed attributes),
plus a Streamlit trace viewer. Build the evaluation harness as deterministic scorers where
the question has a factual answer, and LLM-as-judge scorers only where genuine judgement is
required.

Roughly 150 lines for the tracer and 200 for the evaluation harness.

## Alternatives considered

**LangSmith** — the strongest hosted option, excellent trace UI. Rejected: requires an
account and sends traces off-machine, and — decisively — the Lab 6 deliverable would become
a screenshot of LangSmith's product rather than of our system. The lab grades technical
implementation detail; outsourcing the observability layer removes the thing being graded.

**Arize Phoenix** — runs locally, good RAG-specific views. Rejected more narrowly: it adds a
second process and a substantial dependency tree, and the same "whose UI is this" problem
applies, if less sharply. The closest call of the four.

**OpenTelemetry + Jaeger/Tempo** — the correct production answer. Rejected for this project:
a collector and a backend are infrastructure the brief excludes. The span model here is
deliberately shaped compatibly, so adopting OTel later is an exporter change rather than a
rewrite.

**RAGAS / DeepEval** — credible evaluation libraries with published metric implementations.
Rejected: heavy dependency trees, and their metric internals would be quoted rather than
explained. Since citation correctness and guardrail success are *exact* checks, using an LLM
judge for them (as some of these libraries do) would add cost and variance for no
information.

## Consequences

**Makes easy:** zero setup, no accounts, no second process. The trace file is a plain-text
artifact that can be opened, diffed, and pasted into the submission. Metric definitions are
ours to state precisely in the write-up. Evaluation runs offline for everything except the
LLM-as-judge scorers.

**Makes hard:** no distributed tracing, no production-grade trace UI, no sampling or
retention policy. We validate our own metric implementations, with no published benchmark to
check them against. The trace viewer is one more thing to build.

**Accepted:** a viewer that is functional rather than polished, and the ~350 lines of
first-party code across both concerns.

## Revisit when

Traces need to span more than one process, trace volume outgrows flat files, or the
evaluation suite needs to run in CI against published benchmark comparisons.
