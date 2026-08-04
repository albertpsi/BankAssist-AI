# ADR-0008 — Reinstate `sentence-transformers` (CrossEncoder-only) for Lab 3 reranking

**Status:** Proposed — needs Gate 3 (scope) approval alongside the Lab 3 plan
**Date:** 2026-08-04
**Amends:** [ADR-0007](0007-pinecone-and-api-embeddings.md) §Consequences ("Lab 3" cost),
`docs/architecture/technology-stack.md` §2

## Context

ADR-0007 removed `sentence-transformers` from the stack because the Lab 2 brief forbade it
for embeddings, and flagged as an open item that Lab 2's brief-mandated cross-encoder
reranker (`ms-marco-MiniLM-L-6-v2`, a `sentence-transformers` `CrossEncoder`) had no home
left. The Lab 3 brief received now names exactly two reranker options —
`BAAI/bge-reranker-v2-m3` or `cross-encoder/ms-marco-MiniLM-L-6-v2` — and requires "top 10
→ top 5" reranking with before/after score logging. Neither model is reachable through the
`openai` SDK; both are Hugging Face checkpoints normally loaded via `sentence-transformers`
(`CrossEncoder`) or, for the BAAI model, `FlagEmbedding` (which itself depends on
`sentence-transformers`/`torch`).

Following the precedent already set at ADR-0007 ("the brief is what is graded; the brief
wins"), substituting an LLM-based reranker (e.g., asking `gpt-4.1-mini` to score/reorder
candidates) instead of one of the two named models would satisfy the *behavioural* shape
of FR-3.5 but not the lab's explicit model requirement, and would be a substitution made
unilaterally rather than approved. This ADR proposes bringing `sentence-transformers` back,
scoped narrowly.

## Decision

Add `sentence-transformers` back to the dependency set, **for reranking only**. Embeddings
stay exactly as ADR-0007 decided — OpenAI `text-embedding-3-small`, no local embedding
model. The reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2` (not the BAAI model — see
Alternatives), loaded once at process start and reused across requests, wrapped behind a
`Reranker` interface (`src/bankassist/rag/interfaces/reranker.py`) with an in-repo stub
double so no test loads the model or touches disk/network. This is the same "hosted/heavy
dependency behind a narrow interface with a test double" pattern ADR-0007 already
established for `VectorStore` and `Embedder`.

## Alternatives considered

**`BAAI/bge-reranker-v2-m3`.** The lab brief's other named option. Rejected for now:
it is a multilingual, larger checkpoint (~568M params vs. MiniLM's ~22M), pulls in
`FlagEmbedding` as an additional dependency on top of `sentence-transformers`/`torch`, and
the corpus is English-only banking policy text where the smaller cross-encoder is
well-proven. `ms-marco-MiniLM-L-6-v2` satisfies the brief (which offers either model) at a
fraction of the download size and CPU latency — material on a laptop with no GPU. Revisit
if reranking quality on the demo queries is visibly weak.

**LLM-based reranking via the existing `LLMClient` (no new dependency).** Would avoid
reintroducing `torch` entirely and stay inside the "OpenAI is the only provider" model
policy. Rejected: the lab brief names two specific open-weight cross-encoder models, not
"a reranker of your choice" — this is the same situation ADR-0007 already resolved in favor
of the brief over stack minimalism. Also rejected on cost/latency/determinism grounds: an
LLM call per candidate (or a listwise prompt over 10 candidates) is slower, non-free, and
harder to unit-test deterministically than a local cross-encoder's numeric score.

**A hosted rerank API (e.g., Cohere Rerank).** Rejected: adds a second LLM-adjacent
provider and a second API key requirement, which the amended model policy in CLAUDE.md §4
explicitly closed off ("OpenAI... using the API key already present... not a prerequisite
for any part of this project" implies no second provider account). Also not one of the two
models the brief names.

**Skip reranking, keep RRF as the final ranking.** Rejected outright — FR-3.5 and the
lab's required evidence (`docs/labs/` screenshot 7, "before ranking / after ranking")
explicitly require a reranking stage with visible reordering.

## Consequences

**Makes easy.** Lab 3 satisfies FR-3.5 and the brief's explicit model naming exactly. The
`Reranker` interface gives Lab 4+ (and any future swap to the BAAI model or a hosted API) a
single seam to change, per CLAUDE.md's "introduce an interface when there is a second
implementation" rule — the stub *is* that second implementation, satisfying the rule from
day one.

**Makes hard.** `torch` (CPU build) and its ~2.5 GB dependency tree — the exact cost
ADR-0007 was proud to avoid — comes back into the venv. First-request latency includes a
one-time model download (~90 MB for MiniLM) and load into memory; this is mitigated by
loading it once at app startup / first pipeline construction, not per-request, the same
lazy-singleton pattern `api/routes/rag.py::get_pipeline` already uses for the Pinecone
client.

**Accepted costs.**
- **Disk/install.** ~2.5 GB added to the venv (CPU-only `torch` wheel). No GPU is assumed
  or required.
- **Latency.** Reranking 10 candidates with a 22M-parameter cross-encoder on CPU is
  low-double-digit milliseconds per candidate — acceptable for a lab demo, logged per
  FR-3.5's "before/after" requirement so the real number is evidence, not an estimate.
- **Money.** None — the model runs locally, same as ADR-0007's rejected-then-reconsidered
  local option, but now scoped to reranking only, where the brief requires it.
- **Consistency.** The system is no longer "OpenAI-only for AI capability" in the strictest
  reading — cross-encoder reranking is a local, non-generative scoring model, which is a
  different category from "a second LLM provider," but this ADR records the nuance so it
  is not silently inconsistent with CLAUDE.md §4.

**Test obligation.** No unit test loads the real `CrossEncoder` model. A `StubReranker` in
`src/bankassist/rag/stages/` test doubles (mirroring `rag/stubs.py`) returns scripted
`RerankResult` objects so ordering/truncation logic is tested without the ~90 MB model
download in CI.

## Revisit when

CPU reranking latency is unacceptable for the demo; a future lab needs GPU-scale reranking
throughput; the BAAI model is needed for non-English corpus content; or the project moves
past teaching scope and a hosted rerank API's SLA/cost profile becomes preferable to
bundling `torch`.
