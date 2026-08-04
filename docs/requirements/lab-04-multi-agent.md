# Lab 4 — Multi-Agent Orchestration with LangGraph

**Status:** Draft — awaiting Approval Gate 1 (amended 2026-08-04: security foundation)
**Depends on:** Lab 3 (`EnterpriseRagPipeline`, tracing, config)
**Supersedes orchestration approach in:** [ADR-0002](../decisions/0002-hand-written-orchestration.md)
(see [ADR-0009](../decisions/0009-langgraph-agent-orchestration.md))
**Security foundation:** [ADR-0010](../decisions/0010-local-jwt-auth-and-rbac.md) (local
JWT auth + centralized RBAC — Lab 4 scope; guardrails proper are Lab 5)

## 1. Problem

BankAssist today answers one-shot policy questions only (Lab 2/3). There is no way to
ask about a customer's own accounts/transactions, no multi-turn state, and no path to a
state-changing action (filing a dispute) with human oversight. Lab 4 introduces routed,
multi-agent behaviour: a Supervisor that classifies intent and hands off to a Policy,
Banking, or Dispute specialist, with a human-approved write path for dispute creation.

## 2. Objective

A LangGraph `StateGraph` orchestrates four agents (Supervisor, Policy, Banking, Dispute)
over shared, checkpointed conversation state. Read tools execute directly; the one
state-changing tool (`create_dispute`) pauses the graph via `interrupt()` until a human
approves or rejects it. Every request emits a real-execution-derived event stream that
the Streamlit "Agentic Assistant" page renders as a live workflow graph and timeline —
answering "what executed?", not "how well?" (that is Lab 6).

## 3. Functional requirements

**Orchestration**
- FR-1 A `bankassist.agents.graph` module builds a LangGraph `StateGraph` with nodes:
  supervisor, policy_agent, banking_agent, dispute_agent, prepare_dispute (interrupt),
  create_dispute, clarification, unsupported.
- FR-2 Supervisor emits a structured `SupervisorDecision {route, confidence, reason}`
  via the existing `LLMClient` interface (structured output / tool schema), routing to
  one of `POLICY | BANKING | DISPUTE | CLARIFICATION | UNSUPPORTED`. `reason` is a short
  operational string; no chain-of-thought is stored or returned.
- FR-3 Supervisor performs no retrieval, no tool calls, no DB access — routing only.
- FR-4 Conditional edges route on `state.route` exactly as decided by the Supervisor.

**Agents**
- FR-5 Policy Agent answers via the existing `EnterpriseRagPipeline` — no duplicate
  retrieval/classification/rerank logic is written for Lab 4.
- FR-6 Banking Agent answers using only the scoped tools in FR-9; no direct SQLite
  access from agent code.
- FR-7 Dispute Agent composes Banking tools + `EnterpriseRagPipeline` (dispute/KYC
  policy) + Dispute tools to run: identify transaction → check eligibility → prepare
  action → (interrupt) → create dispute.
- FR-8 Multi-turn reference resolution: "the ₹4,500 one" resolves against transactions
  already surfaced in `state.messages` / `state.selected_transaction_id` for the same
  `session_id` — no long-term memory beyond the checkpointed thread.

**Tools**
- FR-9 Exactly five scoped tools, each typed-in/typed-out, single-responsibility,
  independently unit-tested: `get_customer_accounts`, `get_recent_transactions`,
  `get_transaction_details`, `check_dispute_eligibility`, `create_dispute`.
- FR-10 Every tool that reads or writes customer data takes `customer_id` and enforces
  ownership; a cross-customer request returns a typed rejection, never the other
  customer's data. No generic SQL tool exists.
- FR-11 Every tool call emits one `ExecutionEvent` pair (`TOOL_STARTED` /
  `TOOL_COMPLETED` or `FAILED`).

**State & checkpointing**
- FR-12 `BankAssistState` (typed, e.g. `TypedDict`/pydantic) holds the fields listed in
  the Lab 4 brief (messages, customer_id, session_id, request_id, current_agent, route,
  route_confidence, tool_calls, tool_results, retrieved_sources,
  selected_transaction_id, dispute_eligibility, pending_action, approval_required,
  approval_status, execution_events, final_answer). No hidden reasoning or
  chain-of-thought field exists.
- FR-13 LangGraph's in-memory checkpointer persists state per thread; `session_id` is
  used as the LangGraph `thread_id`. Restarting the process loses state — acceptable
  per the lab brief (session-scoped only, no durable memory).

**Human-in-the-loop**
- FR-14 `create_dispute` is only reachable after a genuine `interrupt()` raised from a
  `prepare_dispute` node; resuming uses `Command(resume=...)`, not a string match on the
  next user message.
- FR-15 Rejecting the approval cancels the pending action and never calls
  `create_dispute`. Approving resumes the graph exactly once; a second resume against an
  already-resolved interrupt is rejected (no duplicate dispute creation).

**Execution events / visualization**
- FR-16 `ExecutionEvent {event_type, node_id, node_type, label, status, timestamp,
  summary}` is emitted at each meaningful transition (see event/status enums in the
  Lab 4 brief §14) by graph nodes/tools, not reconstructed after the fact from the
  final answer text.
- FR-17 `POST /api/v1/agent/chat` returns `execution_events` alongside the answer; the
  Streamlit "Agentic Assistant" page renders the workflow graph and a compact timeline
  purely from that list — no keyword matching on the user's question or the answer to
  decide what to highlight.
- FR-18 The graph visual distinguishes NOT_TRIGGERED / RUNNING / COMPLETED /
  WAITING_FOR_APPROVAL / FAILED per node for the current request.

**API**
- FR-19 `POST /api/v1/agent/chat` request: `{message, customer_id, session_id}`.
  Response: `{answer, agent, session_id, status, approval_required, sources,
  execution_events}`. No chain-of-thought, hidden prompts, or raw LangGraph state is
  exposed.
- FR-20 A second endpoint (or the same one keyed by a resume flag) resumes an
  interrupted thread with an approve/reject decision.

## 3a. Functional requirements — security foundation (ADR-0010)

- FR-21 New `users` table (`id, username, password_hash, customer_id nullable, role,
  is_active, created_at`) seeded deterministically with at least one `CUSTOMER` per
  seeded customer (`CUST001`, `CUST002`), one `SUPPORT_AGENT`, one `ADMIN`. Passwords
  are bcrypt-hashed; plaintext is never stored or logged.
- FR-22 `POST /api/v1/auth/login {username, password}` → on success, a signed JWT with
  claims `sub, role, customer_id, exp`; on failure, a generic invalid-credentials error
  (no username/password enumeration hints).
- FR-23 `POST /api/v1/agent/chat` (and the resume endpoint) require a valid, unexpired
  JWT (`Authorization: Bearer …`); missing/invalid/expired tokens are rejected before
  any agent/graph code runs.
- FR-24 A `SecurityContext {user_id, role, customer_id, session_id, request_id}` is
  built once per request from the validated JWT only. `customer_id` and `role` in this
  context are never taken from the request body, from a tool argument, or from
  anything the LLM generates.
- FR-25 Every scoped tool from FR-9 executes using `SecurityContext.customer_id`, not a
  caller/LLM-supplied `customer_id` argument, even if the model's tool call includes
  one — a supplied value is either ignored or, if it conflicts with the context, treated
  as a rejected request (fail closed, not silently corrected).
- FR-26 A central permission set (`VIEW_OWN_ACCOUNTS, VIEW_OWN_TRANSACTIONS,
  VIEW_CUSTOMER_DATA, CHECK_DISPUTE_ELIGIBILITY, CREATE_OWN_DISPUTE,
  INVESTIGATE_DISPUTE, ADMIN_ACCESS`) is defined in one module; each tool declares the
  permission it requires.
- FR-27 A tool dispatcher enforces: LLM requests tool → dispatcher resolves required
  permission → `authorize(context, permission, resource)` → ownership check → execution.
  A denied authorization short-circuits before any DB read/write and is recorded as an
  `ExecutionEvent` (`status=FAILED`, reason = permission denial, no customer data in the
  summary).
- FR-28 `BankAssistState` never contains a password hash, raw JWT, or signing secret.
  Only the minimum identity needed downstream (e.g. `customer_id`, `session_id`) is
  threaded into graph state; `SecurityContext` itself lives outside the checkpointed
  state.
- FR-29 Streamlit gains a login screen (username/password against `/api/v1/auth/login`)
  before the Agentic Assistant page is reachable; after login it shows
  `Logged in as / Role / Customer` and does not render the raw JWT.
- FR-30 `SUPPORT_AGENT` maps to `VIEW_CUSTOMER_DATA` + `INVESTIGATE_DISPUTE` (can look
  across customers for investigation) but not `ADMIN_ACCESS`; `CUSTOMER` maps to
  `VIEW_OWN_ACCOUNTS, VIEW_OWN_TRANSACTIONS, CHECK_DISPUTE_ELIGIBILITY,
  CREATE_OWN_DISPUTE` only; `ADMIN` has `ADMIN_ACCESS` plus all of the above. Exact
  matrix finalized in design, not renegotiated per request.
- FR-31 `create_dispute` still requires both: (a) RBAC permission `CREATE_OWN_DISPUTE`
  for the acting `SecurityContext`, and (b) the existing LangGraph HITL
  interrupt/approval from FR-14/FR-15. Neither substitutes for the other.

## 4. Non-functional requirements

- NFR-1 Labs 1–3 behaviour, routes, and tests are unchanged (`/rag/query`,
  `/api/v1/rag/query`, basic + enterprise pipelines, existing Streamlit RAG page).
- NFR-2 All LLM calls continue to go through `LLMClient`; supervisor/agent calls use the
  fast tier by default (ADR-0005 / CLAUDE.md §4).
- NFR-3 All customer data (`CUST001`, `CUST002`, …) is synthetic, deterministic seed
  data (CLAUDE.md §7); no real PII.
- NFR-4 Deterministic pieces (customer scoping, eligibility rules, event ordering) are
  unit-tested without a live LLM; all LLM calls are stubbed in unit/integration tests
  (CLAUDE.md §9).
- NFR-5 No dependency beyond `langgraph` (+ its direct requirement, `langgraph-checkpoint`
  if separately packaged) is added without being listed in the technology-stack doc /
  called out at the approval gate.
- NFR-6 LangGraph owns orchestration only; SQL, retrieval, business validation,
  customer scoping, and dispute rules stay in `bankassist` service/tool code, not graph
  node bodies (core rule from the Lab 4 brief).
- NFR-7 Authentication stays local and demo-friendly (ADR-0010): no external IdP, no
  OAuth, no MFA, no refresh-token infrastructure. JWT secret is configuration
  (`settings`), never hard-coded.
- NFR-8 No security-relevant decision (identity, role, customer scope) is ever made by
  the LLM; RBAC and ownership checks are deterministic code, unit-testable without a
  live model.

## 5. Assumptions

- Synthetic SQLite dataset (`customers`, `accounts`, `cards`, `transactions`,
  `disputes`) is new in Lab 4 and lives under `data/` with a deterministic seed script
  in `scripts/`, per CLAUDE.md's "created in Lab N" convention.
- "Human approval" in the API/Streamlit context is the same operator driving the demo
  UI — there is no separate reviewer identity/auth model in this lab.
- The installed LangGraph version will be pinned after a real `pip install` during
  implementation; the design below targets the current stable `langgraph` public API
  (`StateGraph`, `add_conditional_edges`, `interrupt`, `Command(resume=...)`,
  `MemorySaver`) and will be re-verified against the actually installed version before
  code is written (Lab 4 brief §30.8).

## 6. Out of scope

- Guardrails beyond what already exists (Lab 5).
- Tracing/eval dashboards, cost/latency/token analytics, LLM-as-judge (Lab 6/7) — the
  execution-event model is designed to be *extended* by Lab 6, not to include it now.
- Durable/external checkpoint store (Redis/Postgres) — in-memory only.
- Long-term/cross-session memory.
- Real payment rails, real account data, authentication/authorization beyond
  `customer_id` scoping.
- A general-purpose SQL or database tool.
- New frontend framework for visualization (React, etc.) — Streamlit + a lightweight
  graph rendering approach only.

## 6a. Out of scope — security (Lab 5 boundary)

Explicitly deferred to Lab 5, per the amendment: prompt-injection/jailbreak detection,
PII/sensitive-data output scanning, tool-argument guardrails beyond RBAC/ownership,
financial-advice guardrails, RAG/context guardrails, an attack corpus, and
must-block/must-allow guardrail evaluation. Lab 4 delivers only the identity/RBAC
foundation those guardrails will sit on top of.

## 7. Acceptance criteria

- AC-1 A POLICY question routes Supervisor → Policy Agent → EnterpriseRagPipeline →
  response; Banking/Dispute agents show NOT_TRIGGERED in the event stream.
- AC-2 A BANKING question routes to Banking Agent and calls exactly the scoped tool(s)
  needed; no other customer's data is ever in the tool result.
- AC-3 A DISPUTE workflow: recent transactions → transaction resolved from a follow-up
  turn referencing an amount → eligibility checked → policy evidence retrieved →
  interrupt raised → graph state shows `WAITING_FOR_APPROVAL` → approval resumes exactly
  once → `create_dispute` runs → a dispute reference is returned.
- AC-4 Rejecting the approval cancels the action; `create_dispute` never runs; the event
  stream shows `APPROVAL_REJECTED` and no `TOOL_STARTED(create_dispute)`.
- AC-5 CUST001 requesting a transaction/dispute belonging to CUST002 is safely rejected
  with no CUST002 data returned, and this is asserted by a test.
- AC-6 Resuming an already-resolved interrupt a second time is rejected, not repeated.
- AC-7 `execution_events` returned by the API are ordered, typed, and sufficient to
  reconstruct the active path — verified by a test that derives the path from events and
  compares it to the known expected path for each demo scenario.
- AC-8 Existing Lab 1–3 test suite still passes unmodified.
- AC-9 Valid credentials authenticate; invalid credentials fail with a generic error;
  an expired or tampered JWT is rejected.
- AC-10 An unauthenticated request to `/api/v1/agent/chat` is rejected before any graph
  execution.
- AC-11 A `CUSTOMER` can view their own accounts/transactions; the same request for
  another customer's data is rejected with no data returned (restates AC-5 in RBAC
  terms) and cannot create a dispute on another customer's transaction.
- AC-12 A tool call whose LLM-generated arguments include a `customer_id` different
  from `SecurityContext.customer_id` is rejected, not silently corrected — proving the
  context, not the model, is authoritative.
- AC-13 A `CUSTOMER` attempting a `SUPPORT_AGENT`/`ADMIN`-only permission is denied by
  `authorize(...)`; a `SUPPORT_AGENT` gets exactly the permissions in FR-30's matrix.
- AC-14 `create_dispute` is blocked by RBAC denial alone (no HITL reached) when the
  actor lacks `CREATE_OWN_DISPUTE`, and separately still requires HITL approval when
  the actor has permission — the two checks are independent and both required.
