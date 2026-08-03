# BankAssist AI — Architecture

**Status:** Approved with amendments
**Version:** 0.2
**Date:** 2026-08-03

> **Amendment log — v0.2.** OpenAI is the initial provider (§9, §11 row 9); no second
> credential is a prerequisite. Economical models are the default tier. The guardrail
> layering is stated explicitly as deterministic-vs-classifier (§7.0). The semantic cache
> gains an explicit eligibility/bypass decision (§9.2, [ADR-0006](../decisions/0006-semantic-cache-eligibility.md)).
> Evaluation separates retrieval quality from generation quality and uses 20–25 cases (§8.2).
> Observability stays deliberately simple — JSONL plus a simple Streamlit view (§8).

Companion documents: [`technology-stack.md`](technology-stack.md),
[`../requirements/project-requirements.md`](../requirements/project-requirements.md),
[`../plan/implementation-plan.md`](../plan/implementation-plan.md),
[`../decisions/`](../decisions/).

---

## 1. Architectural stance

BankAssist AI is a **modular monolith**: one Python package, one process, strong internal
boundaries. Every layer that an enterprise deployment would run as a separate service
exists here as a module with an explicit interface — retrieval, orchestration, guardrails,
observability, caching. The seams are real; the deployment is not distributed.

This is deliberate. The lab's subject is *architecture*, and a distributed deployment would
consume the entire time budget in infrastructure while making every layer harder to
demonstrate. Each module's "how this scales in production" story is documented in §11
rather than built.

Three properties drive every decision below:

1. **Inspectability** — every stage explains its own output. If retrieval returns a chunk,
   you can see the score that put it there and the stage that let it through.
2. **Determinism at the edges** — chunking, fusion, guardrail rules, cost maths, and cache
   keys are pure and testable. Non-determinism is confined to the LLM boundary.
3. **Additive layering** — each lab adds a layer without demolishing the previous one.

---

## 2. High-level architecture

```mermaid
graph TB
    subgraph Clients
        UI[Streamlit UI]
        API_CLIENT[HTTP client / curl]
    end

    subgraph Application["FastAPI application"]
        ROUTE["/chat endpoint"]
    end

    subgraph Governance["Governance layer"]
        IG[Input guardrails]
        OG[Output guardrails]
    end

    subgraph Caching["Cost layer"]
        SC[Semantic response cache]
    end

    subgraph Orchestration["Orchestration layer"]
        SUP[Supervisor agent]
        PA[Policy agent]
        BA[Banking agent]
        DA[Dispute agent]
    end

    subgraph Knowledge["Knowledge layer"]
        RAG[Enterprise RAG pipeline]
        VS[(ChromaDB<br/>vector store)]
        BM[(BM25 index)]
    end

    subgraph Data["Mock data layer"]
        TOOLS[Banking tools]
        DB[(SQLite<br/>synthetic data)]
    end

    subgraph Platform["Cross-cutting"]
        LLM[LLMClient<br/>+ prompt caching]
        TR[Tracer]
    end

    UI --> ROUTE
    API_CLIENT --> ROUTE
    ROUTE --> IG
    IG -->|allow| SC
    IG -->|block| OG
    SC -->|miss| SUP
    SC -->|hit| OG
    SUP --> PA
    SUP --> BA
    SUP --> DA
    PA --> RAG
    RAG --> VS
    RAG --> BM
    BA --> TOOLS
    DA --> TOOLS
    TOOLS --> DB
    PA --> OG
    BA --> OG
    DA --> OG
    OG --> ROUTE

    PA -.-> LLM
    BA -.-> LLM
    DA -.-> LLM
    SUP -.-> LLM
    RAG -.-> LLM
    IG -.-> LLM
    OG -.-> LLM

    TR -.observes.-> Governance
    TR -.observes.-> Orchestration
    TR -.observes.-> Knowledge
    TR -.observes.-> Caching
```

The dotted lines matter as much as the solid ones: **every** component reaches the model
through a single `LLMClient`, and **every** component reports to a single `Tracer`. Those
two chokepoints are what make Labs 6 and 7 possible at all.

---

## 3. Component responsibilities

| Component | Responsibility | Deliberately not responsible for |
|---|---|---|
| **API layer** (`api/`) | HTTP surface, request/response schemas, session handling, error mapping. | Business logic, prompting, retrieval. |
| **UI** (`ui/`) | Chat surface, trace browser, evaluation and cost dashboards. Source of screenshots. | Any logic not also available via the API. |
| **Guardrail engine** (`guardrails/`) | Runs ordered checks over input and output; returns typed verdicts with rule ids. | Deciding *what* to answer; that is the agents' job. |
| **Semantic cache** (`caching/`) | Embed query → similarity lookup → serve or store. Cacheability policy. | Correctness of the cached answer; it stores what the pipeline produced. |
| **Supervisor** (`agents/supervisor.py`) | Intent classification, routing, fan-out to multiple specialists, synthesis, loop bounds. | Domain answers. |
| **Policy agent** (`agents/policy.py`) | Grounded policy/product answers with citations, via the RAG pipeline. | Customer-specific data. |
| **Banking agent** (`agents/banking.py`) | Account, balance, and transaction questions via scoped tools. | Policy interpretation; disputes. |
| **Dispute agent** (`agents/dispute.py`) | Dispute eligibility, reason collection, case creation, status. | Deciding dispute outcomes. |
| **Tools** (`tools/`) | Typed, schema'd, customer-scoped access to mock data. | Any write beyond `create_dispute_case`. |
| **RAG pipeline** (`rag/`) | Classify → rewrite → hybrid retrieve → filter → rerank → build context. | Generation; it returns context, the agent generates. |
| **Vector store / BM25** | Dense and sparse candidate generation. | Ranking quality; that is the reranker's job. |
| **LLMClient** (`llm/`) | Single provider boundary. Prompt caching, retries, token/cost accounting. | Prompt content. |
| **Tracer** (`tracing/`) | Span creation, nesting, timing, attributes, persistence. | Interpreting what it recorded. |
| **Evaluation** (`evaluation/`) | Golden dataset, scorers, report generation, regression comparison. | Fixing what it finds. |

---

## 4. Request data flow

The end-to-end path a single question takes:

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant API as FastAPI
    participant IG as Input guardrails
    participant SC as Semantic cache
    participant S as Supervisor
    participant A as Specialist agent
    participant R as RAG pipeline
    participant T as Tools / SQLite
    participant OG as Output guardrails
    participant TR as Tracer

    U->>API: question + customer_id + session_id
    API->>TR: start trace
    API->>IG: check(input)
    alt blocked
        IG-->>API: verdict=block(rule_id)
        API->>TR: guardrail span (block)
        API-->>U: scoped refusal + trace_id
    else allowed
        IG-->>API: verdict=allow
        API->>SC: lookup(embed(query))
        alt cache hit and cacheable
            SC-->>API: cached answer
            API->>TR: cache span (hit)
        else miss
            API->>S: route(query, history)
            S->>TR: agent span + routing decision
            S->>A: delegate(intent)
            opt policy intent
                A->>R: retrieve(query)
                R->>TR: retrieval span (docs + scores)
                R-->>A: ranked context + citations
            end
            opt account / dispute intent
                A->>T: tool call (customer-scoped)
                T->>TR: tool span (args + result)
                T-->>A: rows
            end
            A-->>S: draft answer + citations
            S-->>API: synthesized answer
            API->>SC: store if cacheable
        end
        API->>OG: check(answer, context)
        OG->>TR: guardrail span (verdicts)
        alt violation
            OG-->>API: redact or block
        end
        API->>TR: close trace (tokens, cost, latency)
        API-->>U: answer + citations + trace_id
    end
```

Two things worth noting. First, the **input guardrail runs before the cache** — a blocked
query must never be answered from cache. Second, the **output guardrail runs on the cached
path too**, so a cached response cannot bypass grounding and PII checks.

---

## 5. RAG architecture

### 5.1 Two pipelines, one interface

Lab 2's basic pipeline is not thrown away when Lab 3 lands. Both implement a
`RetrievalPipeline` protocol and are selectable by config, which is what makes the
Lab 2 vs Lab 3 comparison in the write-up possible.

```mermaid
graph LR
    subgraph Basic["Lab 2 — basic"]
        B1[Query] --> B2[Embed] --> B3[Vector search top-k] --> B4[Context] --> B5[LLM] --> B6[Answer]
    end
```

```mermaid
graph TB
    Q[User query + history] --> CL[1 . Query classification<br/>policy / account / dispute /<br/>general / out-of-domain]
    CL --> RW[2 . Query rewrite<br/>pronoun resolution,<br/>abbreviation expansion,<br/>multi-query variants]
    RW --> HS[3a . Dense retrieval<br/>MiniLM embeddings<br/>ChromaDB top-20]
    RW --> KS[3b . Sparse retrieval<br/>BM25Okapi top-20]
    HS --> FU[4 . Fusion<br/>Reciprocal Rank Fusion]
    KS --> FU
    FU --> MF[5 . Metadata filtering<br/>category, product,<br/>effective date]
    MF --> RR[6 . Cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2<br/>top-5]
    RR --> CB[7 . Context builder<br/>token budget, dedup,<br/>source ids attached]
    CB --> GR[8 . RAG guardrail<br/>untrusted-content wrapping]
    GR --> GEN[9 . Grounded generation<br/>with inline citations]
    GEN --> CC[10 . Citation validation<br/>every id must exist<br/>in retrieved set]
    CC --> A[Answer + citations]
```

### 5.2 Design notes

- **Chunking**: heading-aware Markdown splitting, ~500 tokens with ~80 overlap. Each chunk
  carries `doc_id`, `doc_title`, `section`, `category`, `effective_date`, `chunk_index`.
  Metadata is what makes filtering (FR-3.4) and citation (FR-3.7) possible, so it is
  captured at ingestion, not reconstructed later.
- **Why hybrid**: dense retrieval is strong on paraphrase and weak on exact tokens. A user
  asking about the *"Foreign Transaction Fee"* by its exact name is the canonical case
  where BM25 wins outright. Fusing both is the cheapest large accuracy gain available.
- **Fusion**: Reciprocal Rank Fusion (`score = Σ 1/(k + rank)`, k=60). Rank-based, so it
  needs no score normalization between two incomparable scoring systems. Deterministic and
  trivially unit-testable.
- **Reranking**: the cross-encoder scores query and chunk *jointly*, which bi-encoder
  embeddings cannot. It is the single biggest precision win in the pipeline, and it runs
  locally on CPU in tens of milliseconds for 20 candidates.
- **Citations**: generation is asked to emit `[doc_id#chunk_index]` markers. A deterministic
  post-check resolves each against the retrieved set; unresolvable markers are a guardrail
  failure, not a formatting quirk. This turns "does it hallucinate citations?" from a vibe
  into an assertion.

---

## 6. Agent architecture

```mermaid
graph TB
    U[User query] --> SUP{Supervisor agent<br/>intent classification<br/>+ routing}

    SUP -->|policy / product /<br/>terms / fees| PA[Policy agent]
    SUP -->|balance / transactions /<br/>account details| BA[Banking agent]
    SUP -->|dispute / chargeback /<br/>unauthorized charge| DA[Dispute agent]
    SUP -->|multi-domain| BOTH[Fan-out to 2+ agents]

    PA --> RAGP[Enterprise RAG pipeline]

    BA --> T1[get_customer_profile]
    BA --> T2[get_customer_transactions]
    BA --> T3[get_transaction]

    DA --> T3
    DA --> T4[create_dispute_case]
    DA --> T5[get_dispute_status]
    DA -.->|dispute policy lookup| RAGP

    BOTH --> SYN[Synthesis]
    PA --> SYN
    BA --> SYN
    DA --> SYN
    SYN --> OUT[Final answer]

    T1 --> DB[(SQLite)]
    T2 --> DB
    T3 --> DB
    T4 --> DB
    T5 --> DB
```

### 6.1 Orchestration model

Routing is **LLM-driven with a deterministic fallback**. The supervisor makes one
low-cost classification call (economical tier) that returns a structured intent; if the call
fails or returns an unknown intent, a keyword-based fallback classifier decides. That
fallback is not a nicety — it means routing is testable without a live model and the demo
never dies on a transient API error.

Specialist agents run a bounded tool-calling loop:

```
for turn in range(MAX_TURNS):          # MAX_TURNS = 5
    response = llm.call(messages, tools=agent_tools)
    if not response.tool_calls:
        return response.text
    for call in response.tool_calls:   # MAX_TOOL_CALLS budget enforced
        result = execute(call, customer_id=ctx.customer_id)
        messages.append(tool_result(result))
return partial_answer(messages)        # bounded, never spins
```

The `customer_id` is **injected by the runtime, not supplied by the model**. This is the
central tool-guardrail decision: the model chooses *which* tool and *what* transaction id,
but it cannot choose *whose* data. Cross-customer access is therefore not a policy the
model must obey — it is structurally impossible.

### 6.2 Why hand-written rather than a framework

An orchestration framework would supply the loop above and take the trace, the guardrail
hooks, and the cost accounting with it — behind abstractions the lab write-up would then
have to explain second-hand. Roughly 200 lines of explicit supervisor code keeps every hop
visible, keeps the trace exact, and removes a large dependency. Recorded as
[ADR-0002](../decisions/0002-hand-written-orchestration.md).

---

## 7. Guardrail architecture

Guardrails are a **pipeline of ordered checks**, cheapest and most certain first. Each
returns a typed `GuardrailVerdict { rule_id, decision, severity, rationale, matched_span }`.
The engine short-circuits on the first `block`. Every verdict — from either layer — is
written to the trace, so any decision can be audited after the fact.

### 7.0 Which layer decides what

The governing rule: **a check must not use an LLM where a deterministic rule decides the
same property.** Deterministic checks are free, instant, reproducible, and cannot be
argued out of firing; classifier checks cost money and latency and are themselves
attackable. Running the deterministic layer first means the fallible layer only ever sees
what survived the certain one.

| Property | Layer | Why |
|---|---|---|
| Known PII patterns (PAN, SSN-shaped, CVV) | **Deterministic** — regex | Decidable by pattern; a regex cannot be talked out of firing |
| Sensitive banking identifiers (account, routing, card) | **Deterministic** — regex + checksum shape | Same |
| Malformed or oversized input | **Deterministic** — schema + length bounds | Decidable; also a cheap denial-of-wallet control |
| Output masking | **Deterministic** — transform | Must be reliable, not probabilistic |
| Tool allow-lists and argument schemas | **Deterministic** — schema validation | Authorization must never be a model's judgement call |
| Citation *structural* validation | **Deterministic** — resolve id against retrieved set | Exact set-membership check |
| Prompt-injection / jailbreak **intent** | **Classifier** | Framing and novelty are not pattern-decidable |
| Personalized-financial-advice **intent** | **Classifier** | The allow/restrict boundary is semantic, not lexical (§7.1) |
| Unsupported or unsafe semantic output | **Classifier** | Grounding is a judgement about meaning |

Deterministic rules also carry the input stage's first pass at injection: literal markers
("ignore previous instructions"), delimiter injection, and role-override phrasing are
caught lexically before the classifier is asked about intent.

```mermaid
graph TB
    IN[User input] --> L1

    subgraph Input["Input guardrails"]
        L1[1 . Deterministic rules<br/>regex: injection markers,<br/>PAN / SSN / CVV patterns] --> L2
        L2[2 . Heuristics<br/>delimiter injection,<br/>role-override, encoding tricks] --> L3
        L3[3 . LLM classifier — economical tier<br/>jailbreak, out-of-domain,<br/>financial-advice intent]
    end

    L3 -->|block| REF[Scoped refusal<br/>+ rule id in trace]
    L3 -->|allow| FS

    subgraph Financial["Financial-safety classifier"]
        FS{Educational or<br/>personalized?}
    end

    FS -->|"general education,<br/>product / policy explanation"| OK[Allow]
    FS -->|"specific security,<br/>specific amount,<br/>personal allocation"| REF

    OK --> AG[Agent execution]

    subgraph Tool["Tool guardrails"]
        TG1[customer_id injected by runtime]
        TG2[Schema validation on every call]
        TG3[Write ops: precondition check<br/>+ mandatory trace record]
        TG4[No transfer / payment tool exists]
    end

    AG --> TG1 --> TG2 --> TG3 --> TG4 --> RET

    subgraph RAGG["RAG guardrails"]
        RET[Retrieved chunks] --> W[Wrap in untrusted-data<br/>delimiters]
        W --> SP[System prompt asserts:<br/>content inside is information,<br/>never instruction]
        SP --> DET[Detect instruction-shaped<br/>text; surface, do not obey]
    end

    DET --> GEN[Generation]

    subgraph Output["Output guardrails"]
        GEN --> O1[1 . PII / PAN / account-number scan<br/>→ redact]
        O1 --> O2[2 . Personalized-advice scan<br/>→ block]
        O2 --> O3[3 . Unsupported financial claims<br/>→ flag]
        O3 --> O4[4 . Grounding check vs context<br/>→ flag]
        O4 --> O5[5 . Citation resolution<br/>→ block on unresolvable id]
    end

    O5 --> FINAL[Delivered response]

    style REF fill:#7f1d1d,color:#fff
    style FINAL fill:#14532d,color:#fff
```

### 7.1 The allow/restrict boundary

The financial-safety guardrail is the one most likely to be built wrong, because the easy
implementation — block anything mentioning investing — is both useless and untestable as a
success.

| Allowed | Restricted |
|---|---|
| "How does compound interest work?" | "Should I put $50,000 into NVDA?" |
| "What is an index fund?" | "Is now a good time to buy tech stocks?" |
| "What's the difference between a Roth and traditional IRA?" | "Given my balance, what should I invest in?" |
| "How is my card's APR calculated?" | "Rebalance my portfolio to 80/20." |
| "What fees does this account charge?" | "Which of these two funds will perform better?" |

The distinguishing signal is **personalization and directive specificity**, not topic. The
test suite encodes both columns, and a failure in the left column is treated as severely as
a failure in the right — over-blocking is a defect (FR-5.7, AC-5.4).

### 7.2 Why layered rather than a single model call

A single classifier prompt judging everything at once is the most attackable design — the
guardrail itself becomes a prompt — costs a full call on every request, and produces one
opaque verdict instead of the per-rule attribution the trace requires. The layering in
§7.0 is what makes each decision cheap, attributable, and auditable.

---

## 8. Observability architecture

```mermaid
graph TB
    subgraph Trace["Trace — one per request"]
        ROOT["root span<br/>trace_id, customer_id, session_id<br/>total latency / tokens / cost"]
        ROOT --> G1[guardrail.input<br/>rules run, verdict, rule_id]
        ROOT --> C1[cache.lookup<br/>hit / miss, similarity]
        ROOT --> AG[agent.supervisor<br/>intent, confidence, route]
        AG --> HO[agent.handoff<br/>from, to, reason]
        HO --> AG2[agent.policy]
        AG2 --> RT[rag.retrieval<br/>stage, query, doc_ids,<br/>scores, latency per stage]
        AG2 --> LM[llm.call<br/>model, input/output/cache tokens,<br/>latency, cost]
        AG --> AG3[agent.banking]
        AG3 --> TL[tool.call<br/>name, args, result summary,<br/>error, latency]
        ROOT --> G2[guardrail.output<br/>per-check verdicts]
    end

    Trace --> JL[(traces/*.jsonl)]
    JL --> UI[Streamlit AgentOps tab<br/>trace list → timeline drill-down]
    JL --> EV[Evaluation harness]
```

### 8.1 Span contract

Every span carries `span_id`, `parent_span_id`, `trace_id`, `type`, `name`,
`start_ts`, `duration_ms`, `status`, and a typed `attributes` payload. LLM spans
additionally carry `model`, `input_tokens`, `output_tokens`, `cache_read_tokens`, and
`cost_usd`. This is an OpenTelemetry-shaped model without the OpenTelemetry dependency —
JSONL on disk is enough for a single-process lab and is trivially inspectable, which is the
point.

### 8.2 Evaluation

```mermaid
graph LR
    GD["Golden dataset<br/>20–25 curated YAML cases<br/>11 categories"] --> RUN[Eval runner]
    RUN --> SYS[BankAssist pipeline]
    SYS --> TRC[Traces]
    TRC --> SPLIT{Attribute the outcome}

    SPLIT --> RQ
    SPLIT --> GQ

    subgraph RQ["Retrieval quality — scored over the retrieved set"]
        R1["expected-document recall<br/>(deterministic)"]
        R2["rank of first relevant doc<br/>(deterministic)"]
        R3["context relevance<br/>(judge)"]
    end

    subgraph GQ["Generation quality — scored over the answer GIVEN that set"]
        G1["citation correctness<br/>(deterministic)"]
        G2["guardrail success<br/>(deterministic)"]
        G3["groundedness, answer relevance,<br/>task completion (judge)"]
    end

    RQ --> REP["Markdown report + JSON metrics<br/>retrieval and generation reported separately"]
    GQ --> REP
    TRC --> OPS["latency · tokens · cost<br/>(deterministic, from spans)"]
    OPS --> REP
    REP --> CMP[Run-over-run comparison<br/>regression detection]
```

Two properties of this design matter more than the metric list:

**Retrieval and generation are scored separately** (FR-6.7a). A bad answer has two very
different causes — the right documents were not retrieved, or they were retrieved and the
model failed to use them — and a single blended score hides which. Retrieval metrics are
computed over the retrieved set; generation metrics are computed over the answer *given*
that set, so a generation failure is not blamed on retrieval or vice versa.

**Deterministic scorers are preferred wherever the question has a factual answer.**
Citation correctness, guardrail success, and expected-document recall are exact checks, not
judgements — using an LLM judge for them would add cost and variance for no information.
The judge is reserved for the four metrics that genuinely require reading comprehension,
and only there is the optional stronger model used.

The dataset is **20–25 curated cases, not volume**, covering: straightforward policy
retrieval, exact banking terminology, multi-turn / query rewrite, retrieval failure,
generation and grounding, citations, dispute workflows, PII, prompt injection, financial
advice, and out-of-domain. A case that does not distinguish a working system from a broken
one does not belong in the set.

Observability stays deliberately small: **JSONL traces plus a simple Streamlit view.** No
collector, no hosted platform, no distributed tracing — those are scaling notes (§12), not
lab work.

---

## 9. Cost optimization architecture

```mermaid
graph TB
    Q[Incoming query] --> ELIG{"1 . Cache eligibility<br/>default = BYPASS"}

    ELIG -->|"customer-specific:<br/>transactions, profile,<br/>disputes, banking state"| BYPASS["BYPASS<br/>never looked up,<br/>never stored"]
    ELIG -->|"policy / FAQ,<br/>no customer content"| SEM{"2 . Semantic cache<br/>cosine similarity ≥ threshold?"}

    SEM -->|hit| SERVE["Serve cached answer<br/>0 generation cost"]
    SEM -->|miss| ROUTE
    BYPASS --> ROUTE

    ROUTE{"3 . Model-tier routing"}
    ROUTE -->|"classification, rewrite,<br/>guardrail classifier,<br/>routine generation"| FAST["Economical model<br/>LLM_MODEL_FAST"]
    ROUTE -->|"selected LLM-as-judge<br/>cases only (optional)"| STRONG["Stronger model<br/>LLM_MODEL_STRONG"]

    FAST --> PP
    STRONG --> PP

    subgraph PP["4 . Stable prompt prefix (best-effort provider caching)"]
        P1["tools — deterministic order"]
        P2["system prompt — frozen:<br/>no timestamps, UUIDs,<br/>per-user values"]
        P3["conversation history"]
        P4["current question — volatile, last"]
        P1 --> P2 --> P3 --> P4
    end

    PP --> RESP[Response]
    RESP --> STORE{"Eligible?"}
    STORE -->|yes| PUT["Store in semantic cache"]
    STORE -->|no| SKIP["Not stored"]
    PUT --> OUT[Deliver]
    SKIP --> OUT
    SERVE --> OUT

    RESP --> ACC["5 . Accounting<br/>tokens · latency ·<br/>estimated cost → trace span"]

    style SERVE fill:#14532d,color:#fff
    style BYPASS fill:#7f1d1d,color:#fff
```

### 9.1 The measurable levers

The core of Lab 7 is deliberately **provider-independent**, so no lab outcome depends on
acquiring a second API credential.

| Lever | Mechanism | Measurable as |
|---|---|---|
| **Semantic caching** | Embed query → cosine similarity against stored eligible queries → serve on threshold. | Hit rate; generation calls avoided; latency and cost delta on a fixed workload. |
| **Model-tier routing** | Economical model for classification, rewriting, guardrail classification, and routine generation. Stronger model only for selected LLM-as-judge cases. | Per-call-type token and cost breakdown vs. a single-tier baseline. |
| **Token accounting** | Every LLM span records input and output tokens against the model that produced them. | Tokens per request type, aggregated from real spans. |
| **Latency measurement** | Every span is timed; the root span totals the request. | p50/p95 per request type, cached vs. uncached. |
| **Estimated cost comparison** | Pure function over the configurable price table. | Before/after table over a fixed 30-query workload. |
| **Provider prompt caching** *(best-effort)* | Keep the prefix byte-stable and let the provider cache it. | Reported **only where the provider exposes a measurable signal**; otherwise recorded as a documented finding. |

### 9.2 Cache eligibility — an explicit, defaulted-to-bypass decision

The semantic cache is the one component that can cause a *privacy* failure rather than
merely a wrong answer: serving customer A's cached transaction summary to customer B is a
data breach, not a cache miss. The design therefore makes eligibility an **explicit
decision on every request, defaulting to bypass**, rather than an exclusion rule that has
to remember every unsafe case.

| | Eligible | Bypass (default) |
|---|---|---|
| **Content** | Policy, product, FAQ, general banking education | Transactions, customer profiles, dispute cases, balances, any personalized banking state |
| **Signal** | Route is `policy` or `general` **and** no customer-scoped tool was invoked | Anything else, including unknown or unclassified routes |
| **Behaviour** | Looked up before generation; stored after | Never looked up, never stored |

Rules that follow from it:

- **Positive establishment.** Eligibility must be affirmatively determined. An unrecognized
  route is a bypass, not a cache.
- **Two-sided enforcement.** The check runs at lookup *and* at store. A request that turns
  out to have touched customer data mid-flight is not written to the cache even though it
  was eligible on entry.
- **No cross-customer key.** Eligible entries contain no customer identity by construction,
  because customer-scoped requests never reach the cache at all.
- **Output guardrails still run on hits.** A cached response is not a trusted response.
- **Entries expire**, so a policy update does not serve stale answers indefinitely.
- **The decision is traced** with its reason, so any cache hit can be audited afterwards.
- **Threshold failure mode is documented**: too low, and a near-miss question gets a
  confidently wrong answer — worse than a miss.

Recorded as [ADR-0006](../decisions/0006-semantic-cache-eligibility.md).

### 9.3 Prompt-prefix hygiene

Provider prompt caching is a **prefix match** — one changed byte invalidates everything
after it. The codebase rule holds regardless of provider, because it costs nothing and is
good hygiene anyway: no timestamps, request ids, UUIDs, per-user values, or
non-deterministically-serialized JSON in the system prompt or tool definitions; tools
serialized in sorted order; volatile content last.

Where the provider reports a cache signal, it is asserted and measured. Where it does not,
the prefix discipline still stands and the absence of a measurable signal is reported as a
finding rather than quietly dropped.

### 9.4 Cost accounting

Model prices live in one configurable table in `config.py`, keyed by model id. Every LLM
span computes:

```
cost = (billable_input_tokens × input_price_per_mtok
      + output_tokens         × output_price_per_mtok) / 1_000_000
```

`billable_input_tokens` is `input_tokens` minus any provider-reported cached-prefix tokens,
which are re-added at the provider's discounted rate when — and only when — the provider
reports them. Where no cache signal is available the term is simply zero, and the figure
remains a correct uncached cost.

Because it is one pure function over a configurable table, it is unit-testable to the cent.
The table ships with documented defaults that **must be verified against the provider's
current published pricing** before any cost figure is quoted in the submission; the Lab 7
before/after table is then generated from real recorded spans rather than estimated.

---

## 10. Security considerations

| Concern | Approach in this project | Production would additionally need |
|---|---|---|
| **Secrets** | Environment variables via a single settings module; `.env` git-ignored; `.env.example` holds names only. | Vault / KMS, rotation, no long-lived keys. |
| **Authentication** | Out of scope. `customer_id` is passed and trusted (stated assumption). | OIDC/SSO, session binding, step-up auth for disputes. |
| **Authorization** | Runtime-injected `customer_id` on every tool; the model cannot select whose data it reads. | Policy engine, per-field entitlements, audit of every access. |
| **Prompt injection (user)** | Layered input guardrails; rules then classifier. | Continuous red-teaming, injection corpus in CI. |
| **Prompt injection (retrieved content)** | Retrieved chunks wrapped in untrusted-data delimiters; system prompt asserts they are information, never instruction; instruction-shaped content is surfaced, not obeyed. | Content provenance, ingestion-time scanning, signed documents. |
| **PII exposure** | Card numbers masked at the data layer; output scanning for PAN/SSN/account patterns; no PII in traces or logs. | DLP, tokenization, field-level encryption, retention policy. |
| **Data minimization** | Tools return only the fields the agent needs; no bulk export tool exists. | Purpose limitation enforcement, consent records. |
| **Unauthorized financial operations** | Structurally prevented — no transfer, payment, or balance-mutation tool exists in the system. | Dual control, transaction limits, out-of-band confirmation. |
| **Supply chain** | Small, pinned, well-known dependency set; each addition needs an ADR. | SBOM, dependency scanning, signed builds. |
| **Auditability** | Every request produces a persisted trace with guardrail verdicts and tool calls. | Immutable audit log, retention, tamper evidence. |
| **Denial of wallet** | Bounded agent loops (max turns, max tool calls); cheap-model routing; caching. | Per-tenant quotas, rate limiting, spend alerts. |

Two honest disclaimers, stated because a governance project that overstates its own
guarantees has failed at its own subject:

1. **Guardrails are defence in depth, not proof.** A layered system with deterministic
   rules and a classifier raises the cost of an attack; it does not make injection
   impossible. The evaluation suite measures the guardrail success rate precisely so the
   residual failure rate is a published number rather than an assumption.
2. **No compliance claim is made.** This design discusses PCI-DSS, GDPR, and audit
   considerations. It does not implement them and must not be represented as compliant.

---

## 11. Key architectural trade-offs

| # | Decision | Chosen | Rejected alternative | Rationale | Cost of the choice |
|---|---|---|---|---|---|
| 1 | Deployment shape | Modular monolith | Microservices | Fits the time budget; keeps every layer demonstrable in one trace. | No independent scaling or deployment; boundaries enforced by convention and review, not by network. |
| 2 | Orchestration | Hand-written supervisor | LangGraph / CrewAI / AutoGen | Every hop visible and explainable; exact trace control; no large dependency. | We reimplement routing, loop bounds, and state; no ecosystem tooling. |
| 3 | Embeddings & reranking | Local `sentence-transformers` | Hosted embedding/rerank APIs | Zero marginal cost, offline, deterministic, fast on CPU for this corpus size. | Lower ceiling on retrieval quality than a frontier embedding model; ~500 MB of model download. |
| 4 | Vector store | ChromaDB (persistent, local) | pgvector / Pinecone / Qdrant / FAISS | Zero infrastructure, persists to disk, metadata filtering built in. | Not a scale story; single-process only. |
| 5 | Keyword search | `rank_bm25` in-process | Elasticsearch / OpenSearch | Two dependencies-free lines vs. a JVM service. | Index rebuilt at startup; fine at this corpus size, wrong at 10⁶ docs. |
| 6 | Guardrails | Custom layered engine | NeMo Guardrails / Guardrails AI | Full control over verdict shape and trace integration; no DSL to learn or explain. | We own the rule corpus; no community-maintained attack patterns. |
| 7 | Tracing | Custom JSONL tracer | OpenTelemetry + Jaeger / LangSmith / Phoenix | No account, no collector, no extra process; the file *is* the artifact. | No distributed tracing, no production-grade UI; we build the viewer. |
| 8 | Evaluation | Custom scorers + LLM-as-judge | RAGAS / DeepEval | Transparent metric definitions we can show in the write-up; no heavy dependency tree. | Metric implementations are ours to validate; no published benchmarks. |
| 9 | LLM provider | OpenAI (existing credential), behind `LLMClient` | Anthropic primary; direct SDK calls everywhere | No second credential becomes a project prerequisite. Lab 7's measurable core moves to semantic caching, tier routing, token accounting, latency, and cost — all provider-independent. | Provider-native prompt caching is less observable, so that part of Lab 7 is best-effort. Anthropic remains addable behind the same interface. |
| 13 | Model tier | Economical model as the default everywhere; stronger model optional, judge-only | Frontier model by default | Cost discipline; the labs grade architecture, not model horsepower. | Some tasks may need prompt iteration to work well on a small model. |
| 10 | Mock data store | SQLite | JSON files / Postgres | Real SQL exercises the tool layer honestly; zero setup; ships with Python. | None material at this scale. |
| 11 | UI | Streamlit | React SPA / plain HTML | Fastest path to a screenshot-worthy multi-tab demo (chat, traces, evals, cost). | Not a production UI; limited interaction model. |
| 12 | RAG mode retention | Keep basic *and* enterprise | Replace basic with enterprise | The Lab 2→3 comparison is a required deliverable, not an afterthought. | Slight extra code and a config switch. |

---

## 12. Deferred scaling notes

Recorded so the write-up's "recommendations for enterprise-scale adoption" section has a
foundation, and so the choices above are legibly *choices* rather than limits:

- **Retrieval at scale**: replace Chroma with pgvector or a managed vector DB; move BM25 to
  OpenSearch; precompute embeddings in a batch pipeline; add an ingestion service with
  document-level ACLs so retrieval is entitlement-aware.
- **Orchestration at scale**: the supervisor becomes a service; agent invocations become
  durable workflow steps so a failure mid-dispute is resumable rather than lost.
- **Guardrails at scale**: extract the engine into a shared policy service so every AI
  application in the estate inherits the same rules and the same audit trail; maintain the
  injection corpus as a continuously red-teamed asset.
- **Observability at scale**: emit OpenTelemetry spans to a real collector; the span model
  here was chosen to be shaped compatibly, so this is an exporter change rather than a
  rewrite.
- **Evaluation at scale**: run the golden set in CI on every PR, gate merges on regression
  thresholds, and grow the dataset from production traces that were flagged by guardrails.
- **Cost at scale**: per-tenant budgets and quotas, cache warming for known-hot policy
  questions, and a longer-TTL prompt cache for high-traffic prefixes.
