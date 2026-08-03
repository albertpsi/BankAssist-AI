# Architecture Decision Records

One file per decision that is expensive to reverse or that a future reader would otherwise
have to re-derive. Numbered sequentially, never renumbered.

**Write an ADR when** a change alters the technology stack, adds a dependency, changes a
cross-cutting contract (trace schema, guardrail verdict shape, `LLMClient`), or picks
between two credible options where the losing option is not obviously wrong.

**Don't write one** for a decision that is obvious in the code, or that can be reversed in
an afternoon.

## Template

```markdown
# ADR-NNNN — <Title>

**Status:** Proposed | Accepted | Superseded by ADR-NNNN
**Date:** YYYY-MM-DD

## Context
What forced a decision. Constraints that were live at the time.

## Decision
What we are doing. One paragraph, stated plainly.

## Alternatives considered
Each option, and the specific reason it lost — not a generic dismissal.

## Consequences
What this makes easy, what it makes hard, and what we accept as a result.

## Revisit when
The condition that would make this decision wrong.
```

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-technology-stack.md) | Technology stack selection | Proposed |
| [0002](0002-hand-written-orchestration.md) | Hand-written agent orchestration over a framework | Proposed |
| [0003](0003-custom-guardrail-engine.md) | Custom layered guardrail engine | Proposed |
| [0004](0004-custom-tracing-and-evaluation.md) | Custom tracing and evaluation over hosted AgentOps tools | Proposed |
| [0005](0005-llm-provider-abstraction.md) | OpenAI as the initial provider behind an LLMClient abstraction | Accepted (amended 2026-08-03) |
| [0006](0006-semantic-cache-eligibility.md) | Semantic cache eligibility and customer-data bypass | Accepted |

ADRs 0001–0004 are **Accepted** as of the planning approval on 2026-08-03. ADR-0005 was
amended at that approval — OpenAI replaces Anthropic as the initial provider. ADR-0006 was
added at the same approval to record the cache eligibility decision.
