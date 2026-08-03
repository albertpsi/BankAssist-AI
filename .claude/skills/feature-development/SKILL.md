---
name: feature-development
description: Spec-first, human-in-the-loop workflow for any non-trivial feature in BankAssist AI. Drives requirement → specification → design → implementation plan → HUMAN APPROVAL → implementation → tests → self-review → HUMAN APPROVAL → git/PR. Use whenever the user asks to add, change, or extend application behaviour, or names a lab milestone. Do not write application code before the plan is approved.
---

# Feature Development Workflow

This repository is spec-first and human-in-the-loop. The point is not ceremony — it is
that a human decides *what* gets built and *when it ships*, and the model does the work
in between.

## The two hard stops

```
Requirement → Spec → Design → Plan → ⛔ APPROVAL GATE 1 ⛔ → Implement → Test
  → Self-review → Report → ⛔ APPROVAL GATE 2 ⛔ → Branch → Commit → Push → PR
```

**Gate 1 — before any application code.** Never create, edit, or delete a file under
`src/`, `tests/`, `scripts/`, or `data/` until the user has approved the plan.
**Gate 2 — before anything leaves the machine.** Never run `git commit`, `git push`,
`git branch`, `git checkout -b`, or `gh pr create` until the user has approved the change
report.

Writing or updating documents under `docs/` is *not* gated — that is the deliverable of
the first half.

---

## Phase 1 — Understand the requirement

Before writing anything, resolve:

- What user-visible behaviour changes?
- Which lab does this belong to, and does it depend on an earlier lab being finished?
- What is explicitly **out** of scope for this change?
- Which existing modules does it touch?
- What could it break — a guardrail, a trace contract, a cached prompt prefix, an eval case?

Read the relevant parts of `CLAUDE.md`, `docs/architecture/architecture.md`, and
`docs/plan/implementation-plan.md` before proposing anything. If two readings of the
request would produce materially different work, ask **one** batched clarifying question.
Otherwise make the routine judgement call, state your assumption, and continue.

## Phase 2 — Specification

Write or update `docs/requirements/<feature-slug>.md`:

- **Problem** — one paragraph. What is wrong or missing today.
- **Objective** — what "solved" looks like.
- **Functional requirements** — numbered `FR-n`, each independently verifiable.
- **Non-functional requirements** — latency, cost, determinism, observability.
- **Assumptions** — including anything about the synthetic dataset.
- **Out of scope** — be specific; this is what stops scope creep later.
- **Acceptance criteria** — numbered `AC-n`, phrased so a test can assert them.

If the feature already has coverage in `docs/requirements/project-requirements.md`,
extend that file instead of creating a near-duplicate.

## Phase 3 — Design

Propose the design in the spec file or a sibling design note. Cover:

- Component/module placement — which package, which existing seams it plugs into.
- Data flow, with a Mermaid diagram when there is more than one hop.
- Interfaces and Pydantic models being added or changed.
- **Guardrail impact** — does this add an input surface, a tool, or an output path?
- **Observability impact** — which trace spans and attributes it emits.
- **Cost impact** — new LLM calls, their model tier, and whether they can be cached.
- Alternatives considered, and why they were rejected.
- Trade-offs you are knowingly accepting.

Anything that changes the tech stack, adds a dependency, or alters a cross-cutting
contract (trace schema, guardrail verdict shape, `LLMClient`) needs an ADR in
`docs/decisions/` — number it sequentially, follow the existing template.

## Phase 4 — Implementation plan

A concrete, ordered list. Each step names files and is small enough to verify:

```
1. src/bankassist/retrieval/rerank.py — add CrossEncoderReranker (new file)
2. src/bankassist/retrieval/pipeline.py — insert rerank stage after hybrid merge
3. src/bankassist/config.py — add RERANK_MODEL, RERANK_TOP_N
4. tests/retrieval/test_rerank.py — ordering, top-n truncation, empty-input cases
5. tests/retrieval/test_pipeline.py — extend: assert rerank stage appears in trace
6. docs/architecture/architecture.md — update the retrieval diagram
```

Also state: which tests will be written, what evidence will be captured for the lab
write-up, and the estimated blast radius.

## Phase 5 — ⛔ Approval gate 1

Present, in one message:

1. The specification summary (FRs and ACs).
2. The design, including diagrams.
3. The implementation plan.
4. Risks, open questions, and anything you had to assume.

Then say plainly that you are waiting for approval before writing code — and **stop**.
Do not "get a head start". Do not create placeholder files. If the user replies with
changes, revise and re-present; the gate reopens only on an explicit go-ahead.

## Phase 6 — Implementation

Once approved:

- Follow the plan. If reality forces a deviation, say so in the change report — do not
  silently redesign.
- Match the surrounding code's style, naming, and comment density.
- Keep the diff scoped to the plan. Unrelated cleanups you notice go in the report as
  suggestions, not into this diff.
- Do not add error handling for conditions that cannot occur, or abstractions with one
  implementation.
- Every new LLM call, retrieval, tool invocation, and guardrail decision must emit a trace
  span. This is not optional — Lab 6 depends on it.

## Phase 7 — Tests

Load the [`testing`](../testing/SKILL.md) skill and follow it. In brief: write the
must-pass and the must-fail cases, stub LLM calls, run the suite, and report real results.

## Phase 8 — Quality gates

Run, and report actual output for each:

```bash
python -m pytest -q
```
```bash
python -m ruff check .
```

Plus a smoke check that the app still starts (API and/or Streamlit) when the change could
affect startup. A failing gate is a blocker, not a footnote.

## Phase 9 — Self-review

Load the [`code-review`](../code-review/SKILL.md) skill and run the full checklist against
your own diff. Fix what it finds. Anything you decide not to fix becomes an explicit,
justified entry in the "known limitations" section of the report.

## Phase 10 — Change report

Report to the user:

| Section | Content |
|---|---|
| **Files changed** | Path + one line on what changed in each. |
| **Tests executed** | Exact commands run. |
| **Test results** | Real pass/fail counts and coverage delta. If something failed, say so and show the output. |
| **Self-review findings** | What the review caught, and what you did about each. |
| **Known limitations** | Honest. Including anything deferred or stubbed. |
| **Architectural decisions** | Choices made during implementation and why; link any new ADR. |
| **Deviations from plan** | Anything that differs from what was approved at Gate 1. |
| **Lab evidence** | Screenshots / trace excerpts / eval output captured to `docs/labs/`. |

## Phase 11 — ⛔ Approval gate 2

State that you are waiting for approval before any Git operation — and **stop**.

## Phase 12 — Git and PR

Only after approval, and only in this order:

```bash
git checkout -b feat/<lab>-<slug>
```
```bash
git add <specific paths>
```
```bash
git commit -m "<imperative subject>"
```
```bash
git push -u origin feat/<lab>-<slug>
```
```bash
gh pr create --base main --title "<title>" --body "<what changed, what was tested>"
```

Stage specific paths, never `git add -A` — that is how `.env` files and local databases
get committed. Verify `git status` is clean of secrets and generated artifacts before
committing.

**Absolutely never:** push to `main`, force-push, rewrite published history, merge the PR,
or bypass a failing test or hook.

---

## Anti-patterns

| Don't | Instead |
|---|---|
| Start coding "just to explore the shape" before Gate 1 | Write the design; that *is* the exploration |
| Present a plan and immediately begin implementing it | Present, then stop and wait |
| Widen scope because an adjacent problem was obvious | Finish the approved scope; list the adjacent problem in the report |
| Report "tests pass" without running them | Run them; paste the real counts |
| Delete or `xfail` a failing test to get green | Fix the code, or report the failure |
| Commit because the work is "obviously done" | Gate 2 exists for exactly that feeling |
| Skip the lab evidence because the code works | The evidence *is* the deliverable for this project |
