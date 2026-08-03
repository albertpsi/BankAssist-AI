# BankAssist AI

A governed, multi-agent banking service and credit-card dispute assistant, built as the
deliverable for an Enterprise Agentic AI hands-on lab.

> **This is a teaching artifact, not a production banking system.** All customer, account,
> transaction, and dispute data is synthetic. There are no real banking integrations, no
> payment rails, and no real PII. Nothing here is compliance-certified, and nothing here
> should be represented as such.

---

## What it demonstrates

| Lab | Capability | Status |
|-----|-----------|--------|
| 1 | AI-assisted software delivery — spec → design → plan → code → tests → PR | 🟡 planning artifacts complete |
| 2 | Basic RAG over banking policy documents | ⬜ not started |
| 3 | Enterprise multi-stage RAG — classify → rewrite → hybrid retrieve → filter → rerank → cite | ⬜ not started |
| 4 | Multi-agent orchestration — Supervisor → Policy / Banking / Dispute agents | ⬜ not started |
| 5 | Enterprise guardrails — input, financial-safety, tool, output, RAG | ⬜ not started |
| 6 | AgentOps — tracing, trace viewer, automated evaluation | ⬜ not started |
| 7 | Cost optimization — prompt caching, semantic caching, cost observability | ⬜ not started |

**Current state: end of the Lab 1 planning phase.** This repository contains
specifications, architecture, decisions, and an implementation plan. No application code
exists yet, by design — the workflow this project demonstrates requires human approval of
the plan before implementation begins.

## Documentation

| Document | What's in it |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Persistent project context: conventions, security rules, approval gates, Definition of Done |
| [`docs/requirements/project-requirements.md`](docs/requirements/project-requirements.md) | Problem, objectives, users, FRs, NFRs, assumptions, constraints, out-of-scope, acceptance criteria |
| [`docs/architecture/architecture.md`](docs/architecture/architecture.md) | Components, data flow, RAG / agent / guardrail / observability / cost architecture, security, trade-offs |
| [`docs/architecture/technology-stack.md`](docs/architecture/technology-stack.md) | Environment inspection, stack selection, rejected alternatives, risks |
| [`docs/plan/implementation-plan.md`](docs/plan/implementation-plan.md) | The seven-lab build plan with exit criteria and a scope-cut order |
| [`docs/decisions/`](docs/decisions/) | Architecture Decision Records |
| [`docs/labs/`](docs/labs/) | Per-lab evidence and the assembled submission document |

## Planned stack

Python 3.13 · FastAPI · Streamlit · ChromaDB · `rank_bm25` · `sentence-transformers`
(MiniLM embeddings + cross-encoder reranking) · SQLite · OpenAI SDK behind an `LLMClient`
abstraction · `pytest` · `ruff`.

Economical models are the default for every operation; a stronger model is optional and
used only for selected LLM-as-judge evaluation cases. Model ids and prices are
configuration, never code. No second API credential is required to complete any lab.

Orchestration, guardrails, tracing, evaluation, and caching are first-party code —
see [ADR-0002](docs/decisions/0002-hand-written-orchestration.md) through
[ADR-0004](docs/decisions/0004-custom-tracing-and-evaluation.md) for why.

## Getting started

Not yet runnable. Once Lab 2 lands:

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
```bash
copy .env.example .env
```

Then fill in an API key in `.env`, seed the data, and start the API and UI. Exact commands
will be documented here as they are built.

## Development workflow

This repository is **spec-first and human-in-the-loop**. Two gates are hard stops:

1. **Plan approval** — no application code is written until a human approves the spec,
   design, and implementation plan.
2. **Publish approval** — no commit, push, branch, or PR happens until a human approves the
   change report.

The workflow is encoded in [`.claude/skills/feature-development`](.claude/skills/feature-development/SKILL.md),
with [`testing`](.claude/skills/testing/SKILL.md) and
[`code-review`](.claude/skills/code-review/SKILL.md) covering the quality gates in between.

Never push to `main`. Never force-push. Never commit secrets.
