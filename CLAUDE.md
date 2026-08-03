# CLAUDE.md — BankAssist AI

Persistent project context for Claude Code. Keep this file short enough to stay useful
as context on every session. Detailed material lives in `docs/` — link, don't inline.

---

## 1. Project overview

**BankAssist AI** is a governed, multi-agent banking service and credit-card dispute
assistant, built as the deliverable for an Enterprise Agentic AI hands-on lab.

It is a **teaching artifact, not a production banking system.** All customer,
account, transaction, and dispute data is synthetic. There are no real banking
integrations, no payment rails, and no real PII.

The system is built incrementally across seven labs:

| Lab | Capability |
|-----|-----------|
| 1 | AI-assisted software delivery (spec → design → plan → code → tests → PR) |
| 2 | Basic RAG over banking policy documents |
| 3 | Enterprise multi-stage RAG (classify → rewrite → hybrid retrieve → filter → rerank → cite) |
| 4 | Multi-agent orchestration (Supervisor → Banking / Dispute / Policy agents) |
| 5 | Enterprise guardrails (input, financial-safety, tool, output, RAG) |
| 6 | AgentOps: tracing, and automated evaluation |
| 7 | Cost optimization: prompt caching + semantic caching + cost observability |

Full detail: [`docs/requirements/project-requirements.md`](docs/requirements/project-requirements.md),
[`docs/architecture/architecture.md`](docs/architecture/architecture.md),
[`docs/plan/implementation-plan.md`](docs/plan/implementation-plan.md).

## 2. Goals

1. Demonstrate each of the seven lab capabilities end-to-end, with evidence
   (screenshots, traces, evaluation reports) suitable for the submission document.
2. Keep the system small enough that one person can build and run it in 2–3 days.
3. Show enterprise *architecture* thinking without enterprise *infrastructure*.
4. Make every AI decision observable: every request emits a trace with agent hops,
   retrievals, tool calls, guardrail verdicts, tokens, latency, and cost.

## 3. Architecture principles

- **Modular monolith.** One Python package, clear module boundaries. No microservices,
  no Kubernetes, no message brokers.
- **Layers, not frameworks.** Prefer ~200 lines of readable orchestration over a heavy
  agent framework whose internals we cannot show in a lab write-up.
- **Every layer is inspectable.** Retrieval, reranking, guardrails, and routing must be
  able to explain *why* they produced their output.
- **Deterministic where possible.** Guardrail rules, chunking, retrieval, and cost math
  are deterministic and unit-testable. LLM calls are isolated behind interfaces so tests
  can stub them.
- **Local-first.** Embeddings, reranking, keyword search, vector store, and the database
  all run locally. Only generation (and LLM-as-judge evaluation) calls a hosted API.
- **Additive stages.** Each lab adds a layer without rewriting the previous one. Lab 2's
  basic RAG survives as the `basic` retrieval mode after Lab 3 lands, so the two can be
  compared side by side.
- **No premature abstraction.** Introduce an interface when there is a second
  implementation, not before.
- **Scope discipline.** The goal is seven labs completed with strong evidence in 2–3 days.
  Build the minimum that demonstrates a lab objective. Do not add production
  infrastructure, distributed tracing, elaborate UI, framework experiments, or features no
  lab objective needs. If a thing is not evidence for a lab, it is scope creep.

## 4. Technology choices

Decided in [`docs/architecture/technology-stack.md`](docs/architecture/technology-stack.md)
and [`docs/decisions/`](docs/decisions/). Summary:

| Concern | Choice |
|---|---|
| Language / runtime | Python 3.12 (venv, `requirements.txt`) |
| API | FastAPI + Uvicorn |
| UI (for screenshots) | Streamlit |
| LLM | `openai` SDK behind an `LLMClient` interface; Anthropic adapter optional, added later |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` (local) |
| Reranking | `sentence-transformers` `CrossEncoder` `ms-marco-MiniLM-L-6-v2` (local) |
| Vector store | ChromaDB (persistent, local) |
| Keyword search | `rank_bm25` (BM25Okapi, in-process) |
| Mock banking data | SQLite + seeded synthetic dataset |
| Orchestration | Hand-written supervisor + tool-calling loop (no LangGraph/CrewAI) |
| Guardrails | Custom layered engine: rules/regex first, LLM classifier second |
| Tracing | Custom span tracer → JSONL, surfaced in a Streamlit AgentOps tab |
| Evaluation | `pytest` + golden dataset + LLM-as-judge scorers → Markdown/HTML report |
| Caching | Context-aware semantic cache (embeddings + SQLite); provider prompt caching best-effort |
| Testing / QA | `pytest`, `pytest-cov`, `ruff` |

**Model policy (amended 2026-08-03):** the provider is **OpenAI**, using the API key
already present in the environment. An Anthropic key is **not** a prerequisite for any
part of this project.

- **Economical model by default** for classification, query rewriting, guardrail
  classification, and routine generation. No expensive frontier model is the default for
  anything.
- **A stronger model is optional**, used only for selected LLM-as-judge evaluation cases.
  If `LLM_MODEL_STRONG` is unset, the economical model is used everywhere.
- **Model IDs are configuration, never code.** They live in `.env` and are read once in
  `config.py`. Never hard-code a model id at a call site — Lab 7 computes cost from the
  price table keyed on those ids.
- **Prices are configuration too.** The price table in settings ships with documented
  defaults that must be verified against the provider's current published pricing before
  any cost figure is quoted in the submission.

**Do not add** a dependency that is not listed in the technology-stack doc without
raising it as an ADR first.

## 5. Coding conventions

- Python 3.12, full type hints on public functions, `ruff` clean.
- `src/bankassist/` package layout; one module per architectural concern.
- Pydantic models for every boundary object (API request/response, trace span,
  guardrail verdict, retrieved chunk, evaluation record).
- Config via environment variables loaded through a single `settings` module. No
  scattered `os.environ` reads.
- Functions do one thing. Prefer pure functions for anything testable.
- Docstrings state *why*, not *what*. Comments only where the code cannot speak.
- No `print()` in library code — use the structured logger.
- Filenames and module names are lowercase-with-underscores; classes are PascalCase.

## 6. Security rules

- **Never commit secrets.** No API keys, tokens, or credentials in source, tests,
  fixtures, notebooks, or documentation. `.env` is git-ignored; `.env.example` holds
  placeholder names only.
- API keys are read from the environment at runtime, once, in the settings module.
- Never log a full API key, request authorization header, or raw credential.
- Retrieved documents and tool outputs are **untrusted data**. They are wrapped in
  clearly delimited blocks and the system prompt states that content inside them is
  information, never instruction. Instruction-shaped text found in retrieved content is
  reported to the user, not acted on.
- Dispute and account tools are read-mostly. The only write tool (`create_dispute_case`)
  requires an explicit customer-scoped authorization check and is recorded in the trace.
- No dependency is added without checking it is a well-known package (typosquat check).

## 7. Banking-data handling rules

**Amended 2026-08-03 (Lab 2).** There are now two distinct data classes, and the rules
differ. The original blanket "everything is synthetic" was written before a real policy
corpus was supplied.

| Class | What it is | Rule |
|---|---|---|
| **Policy corpus** (`data/policies/`) | Real, publicly published policy, FAQ, form, and guide documents from SBI Card and State Bank of India | Real institution, real published text. Contains **no PII, no account numbers, no card numbers**. Third-party material: the publish/ignore decision is the user's, taken at the Git gate |
| **Customer data** (Labs 4+) | Customers, accounts, cards, transactions, dispute cases | **Synthetic, always.** Every rule below applies in full |

- All **customer** data is **synthetic**. Generation scripts live in `scripts/` and are
  deterministic (fixed seed) so the dataset is reproducible.
- Card numbers in the mock dataset are non-Luhn-valid test values, always stored and
  displayed masked (`****-****-****-4321`). Full PANs never exist in the repo.
- No real names, real addresses, or real emails — in either class. Real *institutions*
  appear only as the publisher of a public policy document.
- Every tool that returns customer data takes a `customer_id` and returns data **only**
  for that customer. Cross-customer access is a guardrail failure, and there is a test
  that asserts it.
- Output guardrails scan responses for unmasked card numbers, SSN-shaped strings, and
  full account numbers before the response leaves the system.
- The assistant may explain banking products, policies, and general financial concepts.
  It must **not** give personalized investment advice or recommend specific securities,
  amounts, or allocations. See the financial-safety guardrail spec in the architecture doc.

## 8. AI / LLM engineering rules

- All LLM calls go through the `LLMClient` interface. No direct SDK calls in agents,
  guardrails, or evaluators.
- System prompts live in a `prompts/` module as named constants, not inline strings, so
  they can be versioned, cached, and diffed.
- Prompt caching depends on a **byte-stable prefix**. Never interpolate timestamps, UUIDs,
  request IDs, or per-user values into a system prompt. Volatile content goes last.
  Verify caching works by asserting `cache_read_input_tokens > 0` on the second call.
- Tools are declared with explicit JSON schemas and descriptions that state *when* to call
  them, not just what they do.
- Every LLM call records model, input tokens, output tokens, cache-read tokens, latency,
  and computed cost into the active trace span.
- Prefer structured outputs / strict tool schemas over parsing free text.
- Never weaken a guardrail, lower a threshold, or add an allowlist entry to make a test
  pass. Fix the underlying behaviour or record the limitation.
- **Guardrails are layered, not uniformly LLM-based.** Use a deterministic check whenever
  the property is decidable — PII patterns, banking identifiers, input size and shape,
  output masking, tool allow-lists, citation structural validation. Reserve
  model/classifier checks for properties that genuinely need semantic reasoning —
  injection and jailbreak *intent*, personalized-financial-advice *intent*, unsupported or
  unsafe semantic output. Every decision from either layer is traceable and auditable.
- **The semantic cache is context-aware.** Policy and FAQ answers may be cached; anything
  touching customer-specific state must bypass the cache entirely. See
  [ADR-0006](docs/decisions/0006-semantic-cache-eligibility.md).

## 9. Testing expectations

- `pytest` for everything. Unit tests for deterministic logic; integration tests for
  pipeline wiring; evaluation runs for AI quality.
- LLM calls are stubbed in unit and integration tests. Only the evaluation suite and a
  small set of tests marked `@pytest.mark.live` call a real API.
- Every guardrail gets both a **must-block** and a **must-allow** case. Over-blocking is a
  bug, not a safety win.
- Retrieval, chunking, cost calculation, masking, and cache-key logic are deterministic
  and must have direct unit tests.
- Target: meaningful coverage of `src/bankassist/`, with the guardrail, retrieval, and
  cost modules covered thoroughly. Coverage percentage is a signal, not a goal.
- The test suite must pass before any commit is proposed. Skipping, deleting, or loosening
  a failing test to get green is prohibited — see [`.claude/skills/testing`](.claude/skills/testing/SKILL.md).

## 10. Git workflow

- **Never push to `main`.** All work happens on a feature branch:
  `feat/<lab>-<slug>`, `fix/<slug>`, `docs/<slug>`, `chore/<slug>`.
- Small, focused commits with imperative subjects.
- Never force-push. Never rewrite published history. Never merge your own PR.
- Never commit secrets, `.env`, virtualenvs, model caches, the Chroma store, the SQLite
  database, or trace/eval output.
- Pull requests target `main`, describe what changed and what was tested, and are merged
  by a human.

## 11. Human approval gates

These are hard stops. Do not cross them without an explicit "approved" from the user.

| # | Gate | Stop before... |
|---|------|----------------|
| 1 | **Plan approval** | writing or modifying any application code for a feature. Spec, design, and implementation plan must be reviewed first. |
| 2 | **Publish approval** | any `git commit`, `git push`, branch creation, or PR creation. |
| 3 | **Scope approval** | adding a dependency, changing the tech stack, or expanding a feature beyond its approved plan. |

At each gate: present the artifact, state what happens next, then stop and wait.

## 12. Definition of Done

A feature is done when **all** of the following hold:

1. The specification and design in `docs/` reflect what was actually built.
2. Implementation matches the approved plan; deviations are documented and were approved.
3. Automated tests exist for the new behaviour, including the negative cases.
4. `pytest` passes, `ruff` is clean, and the app starts.
5. An AI self-review has been run (see [`.claude/skills/code-review`](.claude/skills/code-review/SKILL.md))
   and its findings are fixed or explicitly accepted.
6. No secrets, no real PII, no unmasked card data anywhere in the diff.
7. New AI behaviour is observable: it emits trace spans and, where relevant, has an
   evaluation case in the golden dataset.
8. A change report has been given to the user: files changed, tests run, results, known
   limitations, and architectural decisions.
9. Lab evidence (screenshots / trace excerpts / eval output) has been captured into
   `docs/labs/` if the feature completes a lab milestone.

## 13. Repository skills

Invoke these with `/<name>`. They encode the workflow this repo requires.

| Skill | Use it when |
|-------|-------------|
| [`feature-development`](.claude/skills/feature-development/SKILL.md) | Starting any non-trivial feature. Drives requirement → spec → design → plan → **approval** → implement → test → self-review → **approval** → PR. |
| [`testing`](.claude/skills/testing/SKILL.md) | Deciding what to test, writing tests, running them, and reporting failures honestly. |
| [`code-review`](.claude/skills/code-review/SKILL.md) | Before reporting a feature complete. Structured self-review across correctness, architecture, security, PII/financial-data exposure, error handling, logging, coverage, and complexity. |

If a task fits a skill, load the skill first and follow it rather than improvising.

## 14. Repository layout

```
BankAssist-AI/
├─ CLAUDE.md                  # this file
├─ .claude/skills/            # feature-development, testing, code-review
├─ docs/
│  ├─ requirements/           # project-requirements.md, per-feature specs
│  ├─ architecture/           # architecture.md, technology-stack.md
│  ├─ decisions/              # ADRs
│  ├─ plan/                   # implementation-plan.md
│  └─ labs/                   # per-lab evidence + write-ups for submission
├─ src/bankassist/            # application package        (created in Lab 2)
├─ tests/                     # pytest suite               (created in Lab 2)
├─ data/                      # synthetic corpus + seeds   (created in Lab 2)
├─ scripts/                   # ingestion, seeding, eval runners
└─ evaluation/                # golden dataset + reports   (created in Lab 6)
```

Directories marked "created in Lab N" do not exist yet. That is expected — this repository
is currently at the end of Lab 1 (planning artifacts only).
