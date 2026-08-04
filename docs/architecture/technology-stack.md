# Technology Stack

**Status:** Approved; **amended 2026-08-03 for Lab 2 — the amendment is pending approval**
**Version:** 0.2
**Date:** 2026-08-03

> **Amendment (v0.2) — vector store and embeddings.** The Lab 2 brief mandates **Pinecone**
> and OpenAI **`text-embedding-3-small`**, and forbids `sentence-transformers`. Those rows of
> §2 are superseded; see [ADR-0007](../decisions/0007-pinecone-and-api-embeddings.md) and §3.8
> below. The reranking row is consequently an **open item for Lab 3**. Everything else in this
> document stands.

Selection principle: **the simplest stack that lets one engineer demonstrate all seven lab
capabilities in 2–3 days, on this machine, with evidence.** Anything that would be correct
for a production bank but costs a day of setup is documented as a scaling note in
[`architecture.md`](architecture.md#12-deferred-scaling-notes) rather than built.

---

## 1. Environment inspection

Detected on this machine before choosing anything:

| Tool | Version | Relevance |
|---|---|---|
| Python | **3.13.1** (default), **3.12** also installed | Primary runtime |
| pip | 25.2 | Dependency management |
| Node.js / npm | 22.12.0 / 11.4.1 | Available; not needed |
| .NET | 10.0.300 | Available; not needed |
| Java | OpenJDK 11 | Available; rules out nothing, motivates avoiding Elasticsearch |
| Docker | 29.6.1 | Available, but avoided — see §4 |
| Git | 2.42.0 | Required |
| GitHub CLI | 2.96.0, authenticated as `albertpsi` (scopes: `repo`, `workflow`, `gist`, `read:org`) | Lab 1 PR workflow works out of the box |
| `uv` | not installed | Using `venv` + pip |
| Ollama | not installed | Local generation not available |

**Already-installed Python packages that materially shape the choice:**

| Package | Version | Why it matters |
|---|---|---|
| `sentence-transformers` | 3.3.1 | Local embeddings **and** cross-encoder reranking — the two heaviest RAG components — with no API cost and no download beyond model weights |
| `torch` | installed | The expensive transitive dependency is already present |
| `fastapi` | 0.115.6 | API layer available immediately |
| `pydantic` | 2.10.4 | Typed boundaries |
| `numpy` | installed | Vector maths for the semantic cache |
| `litellm` | 1.55.8 | Provider abstraction — considered, see §3.2 |
| `openai` | 2.24.0 | Alternate provider SDK |
| `httpx`, `PyYAML`, `python-dotenv` | installed | Transport, golden dataset, config |

**Credentials present:** `OPENAI_API_KEY` is set. `ANTHROPIC_API_KEY` is **not** set
(`ANTHROPIC_BASE_URL` is, which belongs to the Claude Code session, not to an application
credential). The `ant` CLI is not installed, so there is no OAuth profile an SDK could pick
up either. **This is the one open decision in the stack — see §3.1.**

**Repository state:** `BankAssist-AI/` is a git repo with remote
`https://github.com/albertpsi/BankAssist-AI.git`, branch `main`, **zero commits**.

**Not installed yet** (added to `requirements.txt` as each lab needs them): ~~`uvicorn`,
`pydantic-settings`, `pytest`, `pytest-cov`, `ruff`~~ (Lab 1 — now installed); `pinecone`,
`streamlit`, `PyYAML` (Lab 2); `rank_bm25`, `tiktoken` (Lab 3).

**Credential note (v0.2 amendment):** `PINECONE_API_KEY` is **not** set on this machine.
It must be provisioned before Lab 2 can populate an index.

---

## 2. Selected stack

| Layer | Choice | Why this one |
|---|---|---|
| **Runtime** | Python 3.13 in a project venv (fall back to 3.12 if any wheel fails) | Matches the already-installed ML stack; the whole AI ecosystem is Python-first |
| **Dependencies** | `venv` + pinned `requirements.txt` | `uv` isn't installed; pip is sufficient at this dependency count |
| **API** | FastAPI + Uvicorn | Already installed; Pydantic-native; auto OpenAPI docs are free screenshot material |
| **UI** | Streamlit | Fastest route to a multi-tab demo (chat / traces / evaluation / cost). The UI is a screenshot source, not a product |
| **LLM** | `openai` SDK behind an `LLMClient` interface; Anthropic adapter optional, later | Uses the credential already present; no lab depends on a second one — see §3.1 |
| **Embeddings** | ~~`sentence-transformers` `all-MiniLM-L6-v2`~~ → **OpenAI `text-embedding-3-small` (1536-dim)** *(ADR-0007)* | Mandated by the Lab 2 brief; higher retrieval quality than MiniLM; no `torch` install. Costs a fraction of a cent per full ingestion |
| **Reranking** | ~~`sentence-transformers` `CrossEncoder`~~ → **open item for Lab 3** *(ADR-0007)* | `sentence-transformers` is no longer in the stack, so Lab 3 must choose a different reranking approach at its own approval gate |
| **Vector store** | ~~ChromaDB, persistent local client~~ → **Pinecone (serverless, namespace `bank-policies`)** *(ADR-0007)* | Mandated by the Lab 2 brief. Managed index, first-class console screenshot for lab evidence; costs a second credential and network dependence |
| **Keyword search** | `rank_bm25` (BM25Okapi) | Pure Python, no service, ~10 lines to index this corpus. Supplies the hybrid half that dense search misses |
| **Mock data** | SQLite (stdlib) + deterministic seed script | Real SQL exercises the tool layer honestly; ships with Python; zero setup |
| **Orchestration** | Hand-written supervisor + bounded tool loop | Every hop visible in the trace; no framework internals to explain second-hand ([ADR-0002](../decisions/0002-hand-written-orchestration.md)) |
| **Guardrails** | Custom layered engine (rules → heuristics → LLM classifier) | Full control of the verdict shape and its trace integration; no DSL ([ADR-0003](../decisions/0003-custom-guardrail-engine.md)) |
| **Tracing** | Custom span tracer → JSONL, OTel-shaped | No collector, no account, no second process. The file *is* the artifact ([ADR-0004](../decisions/0004-custom-tracing-and-evaluation.md)) |
| **Evaluation** | `pytest` + YAML golden dataset + deterministic scorers + LLM-as-judge → Markdown report | Metric definitions are transparent and quotable in the write-up |
| **Semantic cache** | MiniLM embeddings + cosine similarity + SQLite store | Reuses the embedding model already loaded; nothing new to install |
| **Testing** | `pytest`, `pytest-cov` | Standard |
| **Lint / format** | `ruff` | One tool for both; fast |
| **Config** | `pydantic-settings` + `python-dotenv` | Single typed settings module; no scattered `os.environ` |
| **VCS / CI** | Git + GitHub via `gh` | Already authenticated |

### Full dependency list

```
# Application
fastapi, uvicorn[standard], streamlit, pydantic, pydantic-settings, python-dotenv

# LLM
openai                         # initial provider (credential already present)
# anthropic                    # optional, only if an Anthropic adapter is added later

# RAG
pinecone                       # managed vector store          (Lab 2, ADR-0007)
# embeddings via the `openai` SDK above — text-embedding-3-small
rank-bm25                      # sparse retrieval              (Lab 3)
tiktoken                       # token budgeting for context assembly (Lab 3)
# reranking                    # open item for Lab 3 — see ADR-0007

# Data / eval
PyYAML, numpy

# Dev
pytest, pytest-cov, ruff
```

Roughly **13 direct dependencies**, of which 6 are already present. Everything else in the
system — SQLite, tracing, guardrails, orchestration, caching, cost accounting — is stdlib
or first-party code, which is exactly what makes it explainable in the lab write-up.

---

## 3. Decisions that needed real thought

### 3.1 LLM provider — decided: OpenAI

**Decision (approved 2026-08-03): use the `OPENAI_API_KEY` already present in the
environment.** No second API credential is a prerequisite for any lab.

The earlier draft recommended Anthropic because its explicit `cache_control` breakpoint and
`cache_read_input_tokens` field make provider prompt caching directly measurable. That is a
real advantage, but it is not worth making the whole project depend on acquiring another
credential — so Lab 7's measurable core was moved to levers that are provider-independent:

| Lab 7 lever | Provider-dependent? | Status |
|---|---|---|
| Semantic caching | No | Primary evidence |
| Model-tier routing | No | Primary evidence |
| Token accounting | No | Primary evidence |
| Latency measurement | No | Primary evidence |
| Estimated cost comparison | No | Primary evidence |
| Provider-native prompt caching | Yes | Best-effort; reported where a measurable signal exists, recorded as a finding where it does not |

This keeps Lab 7 fully quantitative on five of six levers, and the sixth becomes an honest
observation rather than a blocker.

**Portability is retained.** All model access goes through one `LLMClient` interface. An
`AnthropicClient` adapter can be added later without touching a single call site, if a key
becomes available and the prompt-caching evidence is wanted.

### 3.1a Model tiers — economical by default

No expensive frontier model is the default for anything.

| Tier | Setting | Used for | Policy |
|---|---|---|---|
| **Fast** | `LLM_MODEL_FAST` | Intent classification, query rewriting, guardrail classification, routine generation | The default for every operation |
| **Strong** | `LLM_MODEL_STRONG` | Selected LLM-as-judge evaluation cases only | **Optional.** If unset, the fast model is used everywhere and the system still works end to end |

**Model ids are configuration, never code.** They live in `.env`, are read once in
`config.py`, and are validated at startup so a typo or an unavailable model fails loudly
rather than at first use. Nothing hard-codes a model id at a call site.

**Prices are configuration too.** The price table is keyed by model id and ships with
documented defaults. Those defaults **must be verified against the provider's current
published pricing** before any cost figure is quoted in the submission — a cost table
generated from stale constants would be worse than no cost table.

Expected spend: with an economical model as the default and the judge tier optional, a full
evaluation run over the 20–25 case golden dataset should cost well under a dollar, and
total project spend across 2–3 days should be a small single-digit amount.

### 3.2 Why not LiteLLM, despite it being installed

`litellm` 1.55.8 is already present and would give provider abstraction for free. Rejected
because the normalizing layer sits exactly where Lab 7 needs precision: every token count,
latency figure, and cost calculation in the submission is read off the provider response,
and a translation layer in between means explaining *its* accounting behaviour rather than
ours. A ~120-line `LLMClient` with one adapter is more code but a better artifact — and it
is the same seam that lets every test run without an API key.

### 3.3 Why not an orchestration framework

LangGraph, CrewAI, and AutoGen would each supply the supervisor loop — and take the trace,
the guardrail hooks, and the cost accounting with them, behind abstractions the write-up
would then have to describe second-hand. Since Labs 5, 6, and 7 are all *about* those
cross-cutting concerns, owning ~200 lines of explicit orchestration is the cheaper trade.
Recorded as [ADR-0002](../decisions/0002-hand-written-orchestration.md).

### 3.4 Why not OpenTelemetry, LangSmith, or Phoenix

LangSmith and Phoenix are the natural production answers for AgentOps. Both add an account,
a second process, or a hosted dependency, and both would render the trace in *their* UI —
which makes the Lab 6 deliverable a screenshot of someone else's product. A custom span
tracer writing JSONL is ~150 lines, has no dependencies, and produces an artifact you can
open in a text editor and paste into the submission. The span model is deliberately
OTel-shaped so that "export to a real collector" is an exporter change, not a rewrite.

### 3.5 Why not RAGAS or DeepEval

Both are credible evaluation libraries. Both bring substantial dependency trees and define
metrics whose internals we would then be quoting rather than explaining. Since the lab
grades on *technical detail and learnings*, implementing groundedness and citation
correctness directly — deterministically where possible — is worth more than importing them.
Roughly 200 lines for the scorers and the report generator.

### 3.6 Why not Docker

Docker is installed and would make the environment reproducible. It is skipped because
nothing in the stack needs a service: Chroma, BM25, SQLite, embeddings, and reranking all
run in-process. Adding a container adds build time, image size (torch is ~2.5 GB), and a
Windows volume-mount debugging risk, in exchange for reproducibility that a pinned
`requirements.txt` already provides at this scale.

### 3.8 Pinecone and API embeddings (amendment, Lab 2)

The two rows above changed because the Lab 2 brief requires them, not because the original
reasoning was wrong — local Chroma + MiniLM remains the better engineering choice for a
system that must run offline at zero marginal cost. Demonstrating a managed vector database
and a hosted embedding model is part of what Lab 2 is graded on, so the requirement wins.

Both services sit behind first-party `VectorStore` and `Embedder` interfaces with in-repo
test doubles, so the test suite still runs offline, with no account and no cost. The full
reasoning, the alternatives, and the knock-on effects on Labs 3 and 7 are in
[ADR-0007](../decisions/0007-pinecone-and-api-embeddings.md).

### 3.7 Python 3.13 vs 3.12

3.13 is the default interpreter and already carries `torch` and `sentence-transformers`, so
it is the primary target. 3.12 is installed as a fallback if any wheel (most plausibly
`chromadb`) has no 3.13 build. Note that a plain `python -m venv .venv` does **not** inherit
the globally installed `torch` — if the ~2.5 GB re-download is unwelcome, create the venv
with `--system-site-packages`. This is verified as the first step of Lab 2, before any
application code is written.

---

## 4. Risks in the stack

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Provider prompt caching not measurably observable on OpenAI | Medium | One of six Lab 7 levers becomes qualitative | Accepted by design (§3.1); the other five levers carry the lab, and the absence of a signal is reported as a finding |
| Configured model id unavailable on the account | Medium | Startup failure | Settings validate the model id at startup and fail loudly with the configured value named |
| Price table drifts from published pricing | Medium | Cost figures in the submission are wrong | Table is configuration with a documented verify-before-quoting rule |
| ~~`chromadb` has no Python 3.13 wheel~~ / ~~first-run model downloads~~ / ~~cross-encoder too slow on CPU~~ | — | — | Retired by the v0.2 amendment: neither library is in the stack |
| Pinecone credential or network unavailable | Medium | No retrieval at all — the demo cannot run offline | Provision the key before Lab 2 M4; the test suite runs offline via the `VectorStore` double regardless |
| Lab 3 reranking has no chosen implementation | Medium | Lab 3 stalls at its design gate | Decide it at the Lab 3 approval gate; ADR-0007 records it as an open item rather than an assumption |
| Streamlit + FastAPI both needing to run | Low | Demo friction | Streamlit calls the API over HTTP; document the two-command startup |
| Scope creep across 7 labs | **High** | Project doesn't finish | Lab-by-lab exit criteria in the implementation plan; cut depth before cutting labs |
| Token spend during iteration | Low | Minor cost | Economical tier by default; stub the LLM in unit tests; caching from Lab 7 onward |

---

## 5. What would change for production

Recorded so the "recommendations for enterprise-scale adoption" section of the submission
has a foundation, and so the choices above read as choices rather than limits.

| This project | Production |
|---|---|
| Pinecone serverless, one namespace, one API key | Entitlement-aware retrieval: per-tenant namespaces or indexes, row-level access control, key rotation, and a private-network endpoint |
| `rank_bm25` in-process | OpenSearch / Elasticsearch |
| SQLite mock data | The real core banking system, behind an anti-corruption layer |
| Custom JSONL tracer | OpenTelemetry → collector → Tempo/Datadog; the span model is already shaped for this |
| Custom guardrail engine | Extracted into a shared policy service so every AI app inherits the same rules and audit trail |
| Hand-written supervisor | Durable workflow engine so a failure mid-dispute is resumable |
| Streamlit | Real frontend, with accessibility and design-system work |
| `customer_id` trusted from the request | OIDC/SSO, session binding, step-up auth before any dispute action |
| Evaluation run manually | Evaluation in CI on every PR, gating merges on regression thresholds |
| Single-process monolith | Services split along the module boundaries this design already enforces |
