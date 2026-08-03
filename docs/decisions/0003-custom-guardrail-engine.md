# ADR-0003 — Custom layered guardrail engine

**Status:** Proposed
**Date:** 2026-08-03

## Context

Lab 5 requires input guardrails (prompt injection, jailbreak, PII, out-of-domain),
financial-safety guardrails (the education-vs-personalized-advice boundary), tool
guardrails, output guardrails (PII leakage, unsupported claims, grounding, citations), and
RAG guardrails (retrieved content treated as untrusted).

Every guardrail decision must be visible in the trace with a rule id and rationale (FR-5.16),
and the financial-safety boundary must be testable in **both** directions — over-blocking is
a defect, not a safe default (FR-5.7).

## Decision

Build a custom layered engine: an ordered pipeline of checks that runs cheapest-and-most-
certain first (deterministic regex rules → heuristics → economical-tier LLM classifier), each
returning a typed `GuardrailVerdict { rule_id, decision, severity, rationale, matched_span }`,
short-circuiting on the first `block`, and emitting every verdict to the tracer.

## Alternatives considered

**NVIDIA NeMo Guardrails** — the most complete option, with Colang for defining rails.
Rejected: learning and then *explaining* a DSL costs more than the rails themselves, the
verdict shape is its own rather than ours, and its trace integration would not line up with
the span model Lab 6 needs.

**Guardrails AI** — validator-based, good library of pre-built validators. Rejected: the
validator model is oriented toward output validation and structured-output enforcement; the
input-side injection and jailbreak work, and the financial-safety boundary specifically, would
be custom anyway. Adding the dependency to get half the surface is a poor trade.

**A single LLM call that judges everything** — one classifier prompt covering injection,
jailbreak, PII, and advice. Rejected: it is the most attackable design (the guardrail itself
is a prompt), it costs a full call on every request, and it produces one opaque verdict
instead of the per-rule attribution the trace requires. A regex for an unmasked PAN cannot be
talked out of firing; a classifier can.

**Provider-side moderation endpoints** — Rejected: they cover general content safety, not
banking-domain financial advice or prompt injection against our tools, which is the whole
point of Lab 5.

## Consequences

**Makes easy:** exact verdict shape, per-rule attribution in the trace, cheap deterministic
checks that run in microseconds and are trivially unit-tested, and a guardrail architecture
we can explain layer by layer in the write-up. Also lets us build the allow-set corpus first
and treat over-blocking as a test failure.

**Makes hard:** we own the rule corpus. There is no community-maintained set of injection
patterns behind us, so coverage is only as good as the attack corpus we write, and it will
drift as new attack shapes appear.

**Accepted:** guardrails here are defence in depth, not proof. The evaluation suite measures
guardrail success rate precisely so the residual failure rate is a published number rather
than an assumption.

## Revisit when

The rule corpus grows past the point where hand-maintenance is credible, the system faces
untrusted public traffic, or a compliance requirement demands an auditable third-party
control.
