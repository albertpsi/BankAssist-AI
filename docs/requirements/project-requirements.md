# BankAssist AI — Project Requirements

**Status:** Approved with amendments
**Version:** 0.2
**Date:** 2026-08-03

> **Amendment log — v0.2 (approved 2026-08-03).** Provider is OpenAI using the existing
> credential; no second API credential is a prerequisite (FR-7.1). Economical models are
> the default; a stronger model is optional and only for selected LLM-as-judge cases
> (NFR-12). Guardrails are explicitly split deterministic-vs-classifier (FR-5.17). The
> semantic cache gains an explicit eligibility/bypass decision (FR-7.3a). The golden
> dataset is 20–25 high-quality cases, not ~40, and evaluation separates retrieval quality
> from generation quality (FR-6.6, FR-6.7a). Observability stays deliberately simple
> (NFR-13).

---

## 1. Problem statement

Retail bank contact centres absorb a high volume of repetitive contacts that fall into
three broad shapes:

1. **Policy and product questions** — "What is the grace period on my card?", "How is APR
   calculated?", "What is your fraud liability policy?" The answers exist in published
   policy documents, but agents must locate and paraphrase them, and paraphrasing is where
   inaccuracy enters.
2. **Account and transaction lookups** — "What did I spend at that merchant on the 14th?"
   Simple retrieval, but it requires authenticated, correctly-scoped data access.
3. **Credit-card disputes** — a multi-step process: identify the transaction, check its
   eligibility against dispute policy, gather the reason, and open a case. This is the
   longest and most error-prone contact type.

A naive LLM assistant fails at all three in characteristic and *dangerous* ways: it
hallucinates policy it cannot cite, it has no access to customer data, it can be talked
into acting outside its remit by prompt injection, and — the failure mode that matters
most in a regulated industry — it will happily give personalized investment advice when
asked, which a bank is not licensed to do through this channel.

Solving this needs more than a good prompt. It needs grounded retrieval, specialized
agents with scoped tools, layered guardrails, and enough observability to *prove* the
system behaved correctly. That governed architecture is what this project builds.

## 2. Business objective

Deliver a working reference implementation of a **governed, multi-agent banking assistant**
that demonstrates, with evidence, the seven capabilities of the Enterprise Agentic AI
hands-on lab:

| Lab | Capability |
|-----|-----------|
| 1 | AI-assisted software delivery workflow |
| 2 | Basic RAG pipeline |
| 3 | Enterprise multi-stage RAG pipeline |
| 4 | Multi-agent orchestration |
| 5 | Enterprise guardrails architecture |
| 6 | AgentOps and automated evaluation |
| 7 | Cost optimization architecture |

Success is measured by the lab's own evaluation criteria: lab completion (50%), technical
implementation depth with diagrams, screenshots, code and configuration (35%), and
documented learnings and reflection per lab (15%). The artifacts this repository produces
must feed directly into that submission document.

**This is a lab project, not a production banking system.** It is explicitly not intended
to process real customer data, real money, or real disputes.

## 3. Target users

| User | Needs from the system |
|---|---|
| **Lab participant / engineer** (primary) | Build, run, inspect, and explain every layer; capture evidence for the submission. |
| **Lab assessor** | Read the write-up and see that each capability is genuinely implemented, not hand-waved. |
| **Simulated bank customer** (persona in demo) | Ask policy questions, look up transactions, and raise a dispute in natural language. |
| **Simulated risk / compliance reviewer** (persona) | Inspect a trace after the fact and see which guardrails fired, what was retrieved, and what it cost. |

## 4. Functional requirements

### FR-1 — Conversational interface
- **FR-1.1** Accept a natural-language question with an associated `customer_id` and
  `session_id`.
- **FR-1.2** Return a natural-language answer, plus structured metadata: agent(s) used,
  citations, guardrail verdicts, and a `trace_id`.
- **FR-1.3** Maintain multi-turn conversation state within a session.
- **FR-1.4** Expose both an HTTP API and a browser UI. The UI is the source of screenshots
  for the submission.

### FR-2 — Knowledge retrieval (Lab 2)
- **FR-2.1** Ingest a corpus of synthetic banking policy documents (Markdown) into a
  persistent vector store.
- **FR-2.2** Chunk documents with configurable size and overlap, preserving source
  document, section heading, and category metadata.
- **FR-2.3** Answer policy questions using retrieved context, grounded in that context.
- **FR-2.4** Return the source of every answer so it can be verified.
- **FR-2.5** State that it does not know when retrieval returns nothing relevant, rather
  than answering from parametric knowledge.

### FR-3 — Enterprise retrieval pipeline (Lab 3)
- **FR-3.1** Classify each query into a route (policy / account / dispute / general /
  out-of-domain) before retrieving.
- **FR-3.2** Rewrite the query for retrieval: resolve pronouns against conversation
  history, expand banking abbreviations, and optionally produce multiple query variants.
- **FR-3.3** Retrieve using **both** dense semantic search and BM25 keyword search, and
  fuse the result sets.
- **FR-3.4** Apply metadata filters (document category, product type, effective date).
- **FR-3.5** Rerank fused candidates with a cross-encoder and keep the top *k*.
- **FR-3.6** Build a token-budgeted context window from the reranked chunks, with
  per-chunk source identifiers.
- **FR-3.7** Produce answers with inline citations that resolve to real retrieved chunks.
- **FR-3.8** Retain the Lab 2 `basic` pipeline as a selectable mode so basic and enterprise
  retrieval can be compared on the same queries.

### FR-4 — Multi-agent orchestration (Lab 4)
- **FR-4.1** A **Supervisor** agent classifies intent and routes to a specialist, and can
  route a single query to more than one specialist and synthesize the results.
- **FR-4.2** A **Policy Agent** answers policy and product questions using the enterprise
  RAG pipeline.
- **FR-4.3** A **Banking Agent** answers account, balance, and transaction questions using
  mock data tools.
- **FR-4.4** A **Dispute Agent** handles the credit-card dispute flow.
- **FR-4.5** Mock tools are available with strict schemas:
  `get_customer_profile`, `get_transaction`, `get_customer_transactions`,
  `create_dispute_case`, and `get_dispute_status`.
- **FR-4.6** Every agent handoff is recorded in the trace with the reason for the handoff.
- **FR-4.7** Agent loops are bounded — a maximum number of turns and tool calls per
  request, after which the system returns a partial answer rather than spinning.

### FR-5 — Guardrails (Lab 5)

**Input guardrails**
- **FR-5.1** Detect and block prompt-injection attempts ("ignore previous instructions",
  role-override, delimiter injection).
- **FR-5.2** Detect and block jailbreak framings (fictional-persona, developer-mode,
  hypothetical-bypass patterns).
- **FR-5.3** Detect sensitive data in user input (full card numbers, SSN-shaped strings,
  CVV) and refuse to echo or store it, prompting the user not to share it.
- **FR-5.4** Detect out-of-domain requests and decline politely with a redirect.

**Financial-safety guardrails**
- **FR-5.5** *Allow*: general banking education, product explanation, policy explanation,
  and general financial concepts ("what is compound interest", "how do index funds work").
- **FR-5.6** *Restrict*: personalized investment recommendations, instructions to invest a
  specific amount in a specific security, and high-risk personalized financial advice.
  These receive a scoped refusal plus a suggestion to speak to a licensed advisor.
- **FR-5.7** The distinction is testable: a defined set of allow-examples must not be
  blocked, and a defined set of block-examples must not be allowed.

**Tool guardrails**
- **FR-5.8** Every data tool is scoped to the authenticated `customer_id`; cross-customer
  access is impossible by construction and proven by test.
- **FR-5.9** Write operations (`create_dispute_case`) require an explicit precondition
  check and are recorded in the trace.
- **FR-5.10** No agent may perform an unauthorized financial operation — there is no
  transfer, payment, or balance-mutation tool in the system at all.

**Output guardrails**
- **FR-5.11** Scan responses for PII leakage, unmasked card numbers, and full account
  numbers; redact or block.
- **FR-5.12** Scan for personalized financial advice that slipped past the input stage.
- **FR-5.13** Check grounding: flag claims not supported by retrieved context.
- **FR-5.14** Check citation correctness: every citation must resolve to a chunk that was
  actually retrieved for this request.

**RAG guardrails**
- **FR-5.15** Retrieved documents are treated as untrusted data. Instruction-shaped
  content inside a retrieved chunk must not alter system behaviour; it is surfaced to the
  user as a note, not obeyed.

- **FR-5.16** Every guardrail decision (`allow` / `block` / `redact`, with rule id and
  rationale) is recorded in the trace and is auditable after the fact.
- **FR-5.17** Guardrails are **layered by decidability**, not uniformly LLM-based:
  - *Deterministic checks* (regex, schema, structural) are used where the property is
    decidable: known PII patterns, sensitive banking identifiers, malformed or oversized
    input, output masking, tool allow-lists, and citation structural validation.
  - *Model/classifier checks* are used only where semantic reasoning is genuinely
    required: prompt-injection and jailbreak intent, personalized-financial-advice intent,
    and unsupported or unsafe semantic output.
  - A check must not use an LLM call where a deterministic rule would decide the same
    property.

### FR-6 — AgentOps and evaluation (Lab 6)
- **FR-6.1** Every request produces a single trace, identified by `trace_id`, containing
  a tree of spans.
- **FR-6.2** Span types captured: agent execution, agent handoff, LLM call, tool call,
  RAG retrieval (with the documents retrieved and their scores), guardrail intervention,
  and cache lookup.
- **FR-6.3** Each span records latency, and each LLM span records input tokens, output
  tokens, cache-read tokens, model, and computed cost.
- **FR-6.4** Errors are captured on the span where they occurred, with type and message.
- **FR-6.5** Traces are persisted and browsable in the UI — a trace list and a drill-down
  timeline view.
- **FR-6.6** A golden evaluation dataset of **20–25 high-quality cases** (quality over
  volume), covering: straightforward policy retrieval; exact banking terminology;
  multi-turn / query rewrite; retrieval failure (nothing relevant exists); generation and
  grounding; citations; dispute workflows; PII; prompt injection; financial advice; and
  out-of-domain requests.
- **FR-6.7** Automated evaluation computes: answer relevance, groundedness, context
  relevance, citation correctness, guardrail success rate, task completion, latency,
  token usage, and cost.
- **FR-6.7a** Evaluation **reports retrieval quality and generation quality separately**,
  so a failure can be attributed to the right stage. Retrieval metrics (expected-document
  recall, context relevance) are computed over the retrieved set; generation metrics
  (groundedness, answer relevance, citation correctness) are computed over the answer
  *given* that set.
- **FR-6.8** Evaluation produces a versioned report that can be compared across runs to
  detect regressions.

### FR-7 — Cost optimization (Lab 7)

The measurable core of this lab is **semantic caching, model-tier routing, token
accounting, latency measurement, and estimated cost comparison**. Provider-native prompt
caching is demonstrated *where measurable*, but the project must not depend on acquiring an
additional API credential to do so.

- **FR-7.1** Implement a semantic response cache: embed the incoming query, and serve a
  stored response when similarity to a previous query exceeds a threshold **and** the
  cached entry is still valid **and** the request is cache-eligible (FR-7.3a).
- **FR-7.2** Route by model tier: an economical model for classification, query rewriting,
  guardrail classification, and routine generation; a stronger model only where a lab
  objective needs it (selected LLM-as-judge cases).
- **FR-7.3** Never serve a cached response across customers.
- **FR-7.3a** The cache makes an **explicit eligibility/bypass decision** on every request,
  recorded in the trace with its reason. *Eligible:* policy and FAQ answers with no
  customer-specific content. *Bypassed:* anything touching transactions, customer profiles,
  dispute cases, or personalized banking state. The default is bypass — eligibility must be
  positively established, never assumed. See
  [ADR-0006](../decisions/0006-semantic-cache-eligibility.md).
- **FR-7.4** Record per-request token accounting (input, output, and cache-attributable
  tokens), latency, and estimated cost in the trace.
- **FR-7.5** Record cache hits, misses, and bypasses in the trace, distinguishing
  semantic-cache outcomes from any provider-native prompt-cache effect.
- **FR-7.6** Produce a measured before/after comparison of tokens, latency, and estimated
  cost over a fixed workload.
- **FR-7.7** *(best-effort)* Demonstrate provider-native prompt caching and report its
  effect where the provider exposes a measurable signal. If the signal is not observable,
  record that as a documented finding rather than omitting the topic.

### FR-8 — Data
- **FR-8.1** Provide a synthetic policy corpus of roughly 10–15 documents covering credit
  cards, disputes and chargebacks, accounts, fees, fraud, and general banking terms.
- **FR-8.2** Provide synthetic customers, accounts, cards, transactions, and dispute cases
  in SQLite, generated deterministically from a fixed seed.
- **FR-8.3** All data is fictional. No real names, institutions, or valid card numbers.

## 5. Non-functional requirements

| ID | Requirement |
|---|---|
| **NFR-1** | **Runs locally on Windows** with Python 3.12, a virtualenv, and one hosted LLM API key. No Docker, no cloud services, no external database required. |
| **NFR-2** | **Cold start** (ingestion + first query) completes in under 5 minutes on a laptop, including model downloads. |
| **NFR-3** | **Latency**: a policy question through the full enterprise pipeline returns in under ~10 s; a cached response in under ~1 s. These are demo targets, not SLAs. |
| **NFR-4** | **Determinism**: chunking, retrieval scoring, fusion, cost calculation, guardrail rules, and cache keys are deterministic and unit-testable. |
| **NFR-5** | **Observability**: no AI decision is unexplained. Every answer can be traced to its retrieved sources, agent path, and guardrail verdicts. |
| **NFR-6** | **Security**: no secrets in the repository; no real PII; card data always masked. |
| **NFR-7** | **Cost**: a full evaluation run over the golden dataset costs well under $5 at listed API prices. |
| **NFR-8** | **Buildability**: the whole system is achievable by one engineer in 2–3 focused days. |
| **NFR-9** | **Maintainability**: modular monolith, typed boundaries, lint-clean, tested. |
| **NFR-10** | **Offline capability**: embeddings, reranking, keyword search, vector store, and database all run locally; only generation requires network. |
| **NFR-11** | **Evidence**: every lab produces screenshots, diagrams, config/code excerpts, and a learnings section, captured under `docs/labs/`. |
| **NFR-12** | **Economical by default**: no expensive frontier model is the default for any operation. A stronger model is optional, configuration-driven, and used only for selected LLM-as-judge evaluation cases. |
| **NFR-13** | **Deliberately simple observability**: JSONL traces plus a simple Streamlit trace/evaluation view. No distributed tracing system, no hosted observability platform, no elaborate dashboard work. |
| **NFR-14** | **No second credential**: the project runs end-to-end on the OpenAI API key already present in the environment. No lab may become blocked on acquiring another provider credential. |

## 6. Assumptions

1. A single hosted LLM provider is available and its API key is supplied via environment
   variable. The provider is abstracted so it can be swapped.
2. Authentication is out of scope: `customer_id` is passed in the request and trusted.
   This is a stated simplification, and the guardrail work assumes it — scoping is enforced
   *given* an identity, not by establishing one.
3. The synthetic corpus is small enough that in-memory BM25 and a local Chroma collection
   are adequate. Scale strategies are discussed in the architecture doc but not built.
4. A single user runs the system at a time; no concurrency or multi-tenancy requirements.
5. Latency targets are demo-quality, not production SLOs.
6. `sentence-transformers` models can be downloaded once on first run and cached locally.
7. The GitHub remote (`albertpsi/BankAssist-AI`) is available and `gh` is authenticated for
   the Lab 1 PR workflow.

## 7. Constraints

| ID | Constraint |
|---|---|
| **C-1** | Total effort budget of 2–3 days. Scope is cut before quality. |
| **C-2** | No microservices, Kubernetes, Kafka, or equivalent distributed infrastructure. |
| **C-3** | No classical ML training. Pre-trained embedding and reranking models only. |
| **C-4** | No real banking integrations, payment processing, or money movement. |
| **C-5** | No complex authentication or authorization system. |
| **C-6** | No large frontend application. The UI is a thin, functional demo surface. |
| **C-7** | Dependency footprint stays small; every addition needs justification. |
| **C-8** | Must run on Windows without WSL, Docker, or a build toolchain. |
| **C-9** | Human approval gates as defined in `CLAUDE.md` are mandatory. |

## 8. Explicitly out of scope

- Real customer data, real accounts, real cards, real money, real disputes.
- Payment processing, funds transfer, card issuance, or any state-changing banking
  operation beyond creating a mock dispute record.
- Production authentication, OAuth, SSO, RBAC, or session security.
- Card-network chargeback integration (Visa/Mastercard dispute rails).
- Regulatory compliance certification — PCI-DSS, SOC 2, GDPR/CCPA implementation. The
  design *discusses* these considerations; it does not implement or claim them.
- Fine-tuning, model training, or classical ML models.
- Multi-language / i18n support.
- Voice, telephony, or IVR channels.
- Horizontal scaling, load balancing, high availability, disaster recovery.
- A production-grade frontend (routing, state management, design system, accessibility
  audit).
- Streaming token-by-token responses (nice to have; explicitly deferred).
- Real-time document ingestion or a document management system.
- Human-in-the-loop escalation to a live agent.

## 9. Acceptance criteria

### Lab 1 — AI-assisted delivery
- **AC-1.1** `CLAUDE.md`, three `.claude/skills/`, and the `docs/` tree exist and are
  coherent with one another.
- **AC-1.2** Requirements, architecture, technology-stack, ADR, and implementation-plan
  documents exist and cover their stated sections.
- **AC-1.3** At least one feature has gone through the full documented workflow —
  spec → design → plan → approval → code → tests → self-review → approval → PR — with the
  PR as evidence.

### Lab 2 — Basic RAG
- **AC-2.1** Ingestion produces a persistent vector store from the policy corpus, and the
  chunk count and configuration are reported.
- **AC-2.2** For a defined set of policy questions, the correct source document appears in
  the top-3 retrieved chunks.
- **AC-2.3** Answers cite their source document.
- **AC-2.4** A question with no supporting document yields an explicit "I don't have
  information on that" rather than a fabricated answer.

### Lab 3 — Enterprise RAG
- **AC-3.1** All six stages (classify, rewrite, hybrid retrieve, filter, rerank, build
  context) execute and are individually visible in the trace.
- **AC-3.2** A keyword-heavy query (e.g. an exact fee name) that basic semantic search
  misses is retrieved correctly by the hybrid pipeline. This comparison is recorded.
- **AC-3.3** Reranking measurably improves ordering on a defined query set, and the
  before/after ordering is shown.
- **AC-3.4** Every citation in an answer resolves to a chunk actually retrieved for that
  request; a test asserts this.
- **AC-3.5** Basic and enterprise modes can be run on the same query and compared.

### Lab 4 — Multi-agent
- **AC-4.1** Policy, account, and dispute queries each route to the intended specialist,
  demonstrated across a routing test set.
- **AC-4.2** A query spanning two domains triggers two specialists and a synthesized answer.
- **AC-4.3** An end-to-end dispute flow — identify transaction, check eligibility, collect
  reason, create case — completes and returns a case id.
- **AC-4.4** Handoffs and tool calls appear in the trace with their arguments and results.
- **AC-4.5** The Policy Agent demonstrably uses the Lab 3 pipeline.

### Lab 5 — Guardrails
- **AC-5.1** A prompt-injection suite is blocked; the trace shows which rule fired.
- **AC-5.2** A jailbreak suite is blocked.
- **AC-5.3** Sensitive input (card number, SSN-shaped string) is detected and not echoed.
- **AC-5.4** The financial-safety allow-set is answered and the block-set is refused —
  both directions tested, with over-blocking treated as failure.
- **AC-5.5** A tool called for customer A never returns customer B's data; test proves it.
- **AC-5.6** A policy document containing injected instructions does not change system
  behaviour.
- **AC-5.7** An output containing an unmasked card number is caught before delivery.
- **AC-5.8** Every guardrail decision is visible in the trace.

### Lab 6 — AgentOps and evaluation
- **AC-6.1** A single request produces one trace with a complete, correctly nested span
  tree.
- **AC-6.2** Spans carry latency, tokens, cost, and — for retrieval — the documents and
  scores.
- **AC-6.3** The UI lists traces and drills into a timeline for one.
- **AC-6.4** The golden dataset holds 20–25 cases spanning the eleven categories in FR-6.6,
  and the report shows retrieval and generation quality as distinct figures.
- **AC-6.5** An evaluation run produces a report with every metric in FR-6.7.
- **AC-6.6** Two runs can be compared and a deliberately introduced regression is detected.

### Lab 7 — Cost optimization
- **AC-7.1** A semantically similar (not identical) query hits the semantic cache; an
  unrelated query misses; the threshold boundary behaves as specified.
- **AC-7.2** Model-tier routing is demonstrated: auxiliary calls use the economical model,
  and the token/cost saving versus a single-tier baseline is measured.
- **AC-7.3** Customer-specific requests **bypass** the cache. The bypass decision and its
  reason appear in the trace, and a test proves no customer-specific answer is ever stored
  or served.
- **AC-7.4** Cache hits, misses, and bypasses appear in the trace and in a UI counter.
- **AC-7.6** *(best-effort)* Provider-native prompt caching is demonstrated where the
  provider exposes a measurable signal; if it does not, that is recorded as a finding.
- **AC-7.5** A before/after table over a fixed workload shows tokens, latency, and
  estimated cost, with the reduction quantified.

### Project-level
- **AC-P1** `python -m pytest` passes and `ruff check .` is clean.
- **AC-P2** No secrets, no real PII, and no unmasked card numbers anywhere in the
  repository.
- **AC-P3** Each lab has an evidence write-up in `docs/labs/` containing problem statement,
  objectives, assumptions, architecture approach, key design decisions, implementation
  strategy, validation approach, screenshots, code/config excerpts, observations,
  challenges, trade-offs, learnings, and enterprise-adoption recommendations.
