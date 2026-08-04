# ADR-0010 — Local JWT authentication + centralized RBAC as the Lab 4 security foundation

**Status:** Proposed
**Date:** 2026-08-04
**Relates to:** [ADR-0009](0009-langgraph-agent-orchestration.md) (LangGraph orchestration),
Lab 5 (guardrails, not yet implemented)

## Context

Lab 4 introduces per-customer banking data and a state-changing tool (`create_dispute`).
Until now every request was answered with no notion of "who is asking" — the API took a
free-text `customer_id` at face value. That is adequate for Lab 1–3's ungated policy
Q&A, but not for customer-scoped data or a write path, and Lab 5's guardrails need a
trustworthy identity to gate *against* rather than inventing one themselves.

## Decision

Add a minimal, local, demo-appropriate authentication and authorization layer:

- **Authentication:** `POST /api/v1/auth/login` against a new `users` table (bcrypt
  password hashes, never plaintext), issuing a short-lived signed JWT
  (`sub`, `role`, `customer_id`, `exp`).
- **SecurityContext:** every authenticated request builds a `SecurityContext(user_id,
  role, customer_id, session_id, request_id)` from the *validated JWT only* — never from
  request body fields, and never from LLM tool-call arguments.
- **RBAC:** a small, centralized `authorize(context, permission, resource)` component;
  permissions are declared per tool, checked by a tool dispatcher before execution, not
  scattered across agent prompts or agent code.
- **Customer isolation, restated:** tools that take a `customer_id` use
  `SecurityContext.customer_id`, never a value the LLM supplies as a tool argument.

## Alternatives considered

**External IdP (Cognito / Auth0 / Entra ID / generic OAuth).** Rejected for this lab:
adds a hosted dependency, a signup/account flow, and infrastructure the lab specifically
asks to avoid ("keep authentication local and demo-friendly"). Nothing about the lab's
teaching goal — establishing a trustworthy `SecurityContext` boundary for RBAC and tool
authorization — requires a real identity provider; a local, deterministic user table
demonstrates the same boundary with none of the account-management overhead.

**No authentication, keep trusting request-supplied `customer_id`.** Rejected: this is
exactly the gap the amendment closes, and it would leave nothing for Lab 5's guardrails
to gate against — Lab 5 is specified as building *on* this foundation, not creating it.

**Session cookies instead of JWT.** Considered; JWT was chosen because the API is
stateless FastAPI called from a separate Streamlit process (no shared session store),
and a signed, self-contained token is the simplest thing that lets both processes verify
identity without adding a session-store dependency (Redis, etc.) that CLAUDE.md's
scope-discipline principle says to avoid unless a lab objective needs it.

**Full refresh-token / MFA flow.** Explicitly out of scope per the amendment; short-lived
access token only, re-login on expiry. A lab teaching artifact does not need refresh-token
rotation to demonstrate the RBAC/isolation boundary.

## New dependencies

- `pyjwt` — JWT encode/verify. Small, no transitive dependency tree, widely used.
- `bcrypt` — password hashing, used directly (`passlib`'s bcrypt wrapper turned out
  to be incompatible with `bcrypt` 4.1+, which dropped the `__about__` module passlib
  reads for version detection — discovered during implementation; `bcrypt` alone is
  fewer moving parts anyway). Never store plaintext.

Both are flagged as a scope-approval item alongside `langgraph` (ADR-0009) at the
implementation-plan approval gate, per CLAUDE.md §4/§11 gate 3.

## Consequences

**Makes easy:** every tool call has a trustworthy, LLM-proof identity to authorize
against; Lab 5 gets a real boundary to add guardrails around rather than inventing one
under time pressure; customer isolation tests can assert against a real `SecurityContext`
instead of a caller-supplied string.

**Makes hard / accepted limits:** this is not production auth — no MFA, no refresh
tokens, no revocation list, one signing secret from config. That is intentional and
matches the lab's "teaching artifact, not a production banking system" framing
(CLAUDE.md §1). Documented here so it is not mistaken for a production-ready auth
system later.

**Placement relative to LangGraph state (ADR-0009):** `SecurityContext` is built once
per request, outside `BankAssistState`, and only the minimum needed (customer_id, and
enough to log which role acted) is threaded into state for tool execution. No password
hash, JWT, or raw token ever enters `BankAssistState` — it is not something a
checkpointer should ever persist.

## Revisit when

A future lab needs multi-user concurrent sessions with real account lifecycle
(signup, password reset, MFA) — at that point a real IdP integration is justified and
this ADR should be revisited, not extended in place.
