# ADR-0001 — Technology stack selection

**Status:** Proposed
**Date:** 2026-08-03

## Context

BankAssist AI must demonstrate seven enterprise AI capabilities — AI-assisted delivery,
basic RAG, enterprise RAG, multi-agent orchestration, guardrails, AgentOps, and cost
optimization — within roughly 2–3 days of one engineer's time, on a Windows machine, and
produce screenshots, diagrams, and code excerpts for a graded submission.

The project brief explicitly excludes microservices, Kubernetes, Kafka, classical ML, real
banking integrations, complex authentication, and large frontend applications.

Environment inspection found: Python 3.13.1 (and 3.12), Node 22, .NET 10, Java 11, Docker
29.6, Git 2.42, `gh` 2.96 authenticated. Already-installed Python packages include
`sentence-transformers` 3.3.1, `torch`, `fastapi`, `pydantic`, `numpy`, `litellm`, and
`openai`. `OPENAI_API_KEY` is set; `ANTHROPIC_API_KEY` is not.

## Decision

Python 3.13 modular monolith. FastAPI + Uvicorn for the API, Streamlit for the demo UI,
ChromaDB for vector storage, `rank_bm25` for keyword retrieval, `sentence-transformers`
for both embeddings (`all-MiniLM-L6-v2`) and cross-encoder reranking
(`ms-marco-MiniLM-L-6-v2`), SQLite for synthetic banking data, `pytest` + `ruff` for
quality, and first-party code for orchestration, guardrails, tracing, evaluation, and
caching.

Approximately 13 direct dependencies, 6 of which are already installed. Full detail and
per-choice reasoning in [`../architecture/technology-stack.md`](../architecture/technology-stack.md).

## Alternatives considered

**A heavier, more production-shaped stack** — Postgres + pgvector, OpenSearch for BM25,
Docker Compose, a React frontend. Rejected: every component adds setup time and a Windows
failure mode, in exchange for realism the submission does not grade. The scaling path from
this stack to that one is documented instead.

**A lighter stack** — plain JSON files instead of SQLite, numpy cosine similarity instead
of Chroma, no reranker. Rejected: the tool layer is more honestly exercised against real
SQL, and the cross-encoder reranker is the single largest retrieval quality gain available
for ~90 MB and tens of milliseconds. Cutting it would weaken Lab 3's central claim.

**Node/TypeScript** — viable (Node 22 present) and would give a better UI story. Rejected:
`sentence-transformers` has no equivalent in the JS ecosystem, so local embeddings and
reranking would become hosted API calls, adding cost and removing the offline property.

**Docker for reproducibility** — Docker is installed. Rejected: nothing in the stack needs
a service, torch makes the image ~2.5 GB, and Windows volume mounts are a debugging risk. A
pinned `requirements.txt` provides sufficient reproducibility at this scale.

## Consequences

**Makes easy:** a fast start (no infrastructure), offline operation for everything except
generation, deterministic and unit-testable retrieval, and one process to trace and profile.

**Makes hard:** anything about scale. There is no independent deployment, no horizontal
scaling, and no multi-tenancy. Module boundaries are enforced by convention and code review
rather than by the network.

**Accepted:** ~200 MB of model downloads on first run (plus torch if the venv does not share
site-packages), and the residual risk that a `chromadb` wheel is unavailable for Python 3.13
— verified as the first step of Lab 2, with the 3.12 interpreter as the fallback.

## Revisit when

The corpus exceeds ~10⁴ documents (Chroma and in-process BM25 stop being adequate), more
than one user needs concurrent access, or the project moves from lab artifact toward
anything real.
