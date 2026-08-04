# Lab 5 — Guardrails, Security & Financial Safety

## Problem

Lab 4 built a working multi-agent banking assistant with real authentication, RBAC,
customer-ownership enforcement, and a human-approval gate on the one mutating tool.
What it does not have is a layered defense around that architecture: no input
guardrail stops a prompt-injection or system-prompt-extraction attempt before it
reaches the supervisor, no output guardrail catches an accidental secret or unmasked
identifier before it leaves the system, and while routing/authorization/ownership are
each individually correct, nothing in the graph makes it *visible* that "the
supervisor routed here" and "this action is authorized" are different, independently
verified facts.

## Objective

Add an explicit, layered guardrail boundary around the existing Lab 4 architecture —
input, authentication, authorization, ownership, tool, financial-mutation, RAG, and
output — without rewriting any Lab 1–4 module. Every blocked request must be traceable
to WHAT was blocked, WHERE, WHY, and WHICH rule fired. AI-semantic checks (prompt
injection, jailbreak, system-prompt extraction, output safety) are handled by NeMo
Guardrails (ADR-0011); every other control — authentication, RBAC, ownership, tool
authorization, HITL approval, replay protection, secret redaction, PII/financial-data
masking — remains deterministic application code, testable without a live model.

## Functional requirements

- **FR-1** Every chat request passes through deterministic input-shape validation
  (empty/oversized) before any LLM or NeMo call.
- **FR-2** Every chat request that passes FR-1 passes through a NeMo input rail
  classifying prompt-injection/jailbreak, system-prompt-extraction, and
  role/tool-manipulation intent. A blocked request never reaches the LangGraph
  supervisor.
- **FR-3** Authenticated identity continues to come only from the validated JWT
  (`SecurityContext`, ADR-0010) — never from the LLM, and never from a request-body
  `customer_id` that disagrees with it.
- **FR-4** Every sensitive tool invocation (banking read, dispute read/write) is
  independently authorization-checked at the graph-node boundary, in addition to the
  check already inside the scoped tool (`scoped_tools.py`) — supervisor routing is
  never treated as authorization.
- **FR-5** Cross-customer access attempts (accounts, transactions, disputes) are
  rejected deterministically at the data/tool boundary, and the rejection is recorded
  as an `OWNERSHIP_CHECK_BLOCKED` execution event.
- **FR-6** `create_dispute` executes only when all of: authenticated, authorized,
  customer owns the resource, transaction is eligible, human approval was granted, and
  that specific approval has not already been consumed. Each condition is
  independently testable and independently enforced (`tool_authorization.py`).
- **FR-7** An already-resolved approval cannot be replayed to re-trigger a mutation.
- **FR-8** Every final agent answer passes through a NeMo output rail (semantic
  safety) followed by deterministic secret redaction and PII/financial-data masking
  before being returned to the caller.
- **FR-9** Secrets (JWT, bearer tokens, API keys, password hashes) are never present
  in execution-event summaries, logs, or API responses.
- **FR-10** Every guardrail decision (allow or block) is emitted as an
  `ExecutionEvent`, extending the Lab 4 event model, and is renderable by the existing
  Streamlit workflow graph without keyword-matching the final answer.
- **FR-11** Retrieved RAG content remains untrusted data; the existing `<document>`
  delimiting (`rag/prompts.py`) is preserved, and injected instruction-shaped text in
  a retrieved chunk cannot trigger a tool call or change authorization/approval state.

## Non-functional requirements

- **NFR-1** All guardrail decisions are deterministic in tests — no test asserts on a
  live model's judgement; NeMo is exercised through an injectable fake LLM.
- **NFR-2** The guardrail layer adds at most two extra LLM calls per guarded turn (one
  input rail, one output rail) — no larger Colang dialogue system.
- **NFR-3** No Lab 1–4 module is rewritten; the guardrail layer is additive
  (`src/bankassist/guardrails/`) plus thin call-sites in `graph.py`/`agent.py`.
- **NFR-4** Full existing regression suite remains green.

## Assumptions

- The seeded dataset is synthetic; card numbers are already stored pre-masked
  (`****-****-****-4321`). Masking/redaction demonstrate the *pattern*, not protection
  of real customer data.
- `SUPPORT_AGENT`'s `VIEW_CUSTOMER_DATA`/`INVESTIGATE_DISPUTE` permissions remain
  declared but unexercised by any tool (a pre-existing Lab 4 gap, not introduced by
  Lab 5) — documented as a known limitation rather than used to invent a new tool.

## Out of scope

- Lab 6 observability/evaluation platform, LLM-as-judge, cost dashboards.
- A large Colang dialogue system or multiple hand-written rails per direction.
- A generic SQL tool, arbitrary code execution tool, or any new mutating tool beyond
  `create_dispute`.
- Production key rotation, MFA, or a real identity provider.

## Acceptance criteria

AC-1 through AC-20 as specified in the approved Lab 5 plan; implemented in
`tests/security/test_adversarial.py`, cross-referencing `tests/integration/` and
`tests/unit/` where a scenario is already covered by an existing regression test
(noted inline in the test file).
