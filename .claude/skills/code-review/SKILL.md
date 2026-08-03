---
name: code-review
description: Structured self-review of a change before reporting it complete, covering correctness, architecture, maintainability, security, secrets, PII and financial-data exposure, error handling, logging, test coverage, and unnecessary complexity. Use after implementing and testing a feature, and before the Git approval gate.
---

# Self-Review

Review the diff you just produced as if someone else wrote it and you have to sign off on
it. Read the actual changed files — do not review from memory of what you intended.

Start with:

```bash
git status
```
```bash
git diff
```

(or `git diff --stat` first, then read each changed file in full).

Work the checklist below. For each finding, record: **file:line**, what is wrong, the
concrete failure it causes, and the fix. Fix what you can. Anything you consciously leave
becomes a stated entry under "known limitations" — not silence.

---

## 1. Correctness

- Does the code do what the approved spec's acceptance criteria say?
- Off-by-one, boundary, and empty-collection handling — first item, last item, zero items.
- `None` / missing-key paths. Does anything assume a dict key or optional field exists?
- Are async/sync boundaries right? No blocking call inside an async handler.
- Are floats compared with tolerance rather than `==`? (Relevant to score fusion and cost.)
- Is anything mutating a shared or default-argument object?
- Does the change silently alter existing behaviour that other modules depend on?

## 2. Architecture

- Is this in the right module? Does it respect the layering in `docs/architecture/architecture.md`?
- Does it introduce a circular import or a dependency from a lower layer onto a higher one?
- Are boundaries typed with Pydantic models rather than bare dicts?
- Does it bypass an existing seam — e.g. calling an SDK directly instead of via `LLMClient`,
  or reading `os.environ` instead of the settings module?
- Are prompts in the `prompts/` module rather than inline strings?
- Would a reader learning the system from this file understand where they are?

## 3. Maintainability

- Names say what the thing is. No `data`, `result2`, `tmp`, `handle_stuff`.
- Functions do one thing; anything past ~40 lines or 3 nesting levels deserves scrutiny.
- Comments explain *why*, not *what*. Delete comments that narrate the next line.
- No dead code, commented-out blocks, leftover debug scaffolding, or `TODO` without an issue.
- Duplication: is this the third copy of the same logic? (Two is fine; three is a function.)
- Would this be obvious to the reader in three months, or does it depend on context only
  you currently hold?

## 4. Security

- **Secrets.** Grep the diff for anything key-shaped:
  ```bash
  git diff | grep -inE "sk-|api[_-]?key|secret|token|password|bearer|ghp_|xox"
  ```
  Any hit that is not a variable name or placeholder is a blocker.
- Is `.env` (or any local config with real values) staged? Check `git status` before commit.
- Are credentials read once through settings and never logged?
- **Untrusted content.** Retrieved documents and tool outputs are data. Are they clearly
  delimited in the prompt, and does the system prompt state they are never instructions?
- **Injection.** Is any user-controlled string interpolated into a SQL query, a shell
  command, a file path, or a prompt template without escaping/parameterization?
- **Path traversal.** Any file operation on a model- or user-supplied path must resolve to
  a canonical path inside an allowed root.
- Are new dependencies well-known packages, pinned, and listed in `requirements.txt`?
- Does any tool gain a capability beyond what its lab scope requires?

## 5. PII and financial-data exposure

This is the project's subject matter — review it hardest.

- Are card numbers stored, logged, traced, or returned **masked**? Full PANs must not exist
  anywhere, including test fixtures.
- Do trace spans, log lines, error messages, or exception payloads carry customer data?
  Traces are written to disk and screenshotted for the submission document.
- Does every customer-data tool take a `customer_id` and scope its query to it? Is there a
  test proving customer A cannot retrieve customer B's records?
- Does the output guardrail run on **this** response path, or did the change add a route
  that bypasses it?
- Is any real-looking name, address, email, SSN, or institution in fixtures or docs?
  Everything must be synthetic and obviously so.
- **Financial safety:** can this change produce personalized investment advice — a specific
  security, a specific amount, an allocation? If it opens that surface, it needs a
  guardrail case and a test.

## 6. Error handling

- Are failure modes handled at the boundary they occur, or swallowed?
- No bare `except:` and no `except Exception: pass`. Every caught exception is either
  handled meaningfully or re-raised with context.
- Do external calls (LLM, vector store, DB) have sensible timeouts and a defined behaviour
  on failure? Does the user get a useful message rather than a stack trace?
- Are error messages free of internal detail that shouldn't reach a user, while logs keep
  enough to debug?
- Is a partial failure (one agent fails, others succeed) handled coherently?
- Is there error handling for a condition that **cannot happen**? Delete it — it adds noise
  and hides the paths that matter.

## 7. Logging and observability

- Does every new LLM call, tool invocation, retrieval, agent hop, and guardrail decision
  emit a trace span with the required attributes (trace id, span type, latency, tokens,
  cost, verdict)? Lab 6 depends on this; a missing span is a missing deliverable.
- Correct levels: `DEBUG` for developer detail, `INFO` for lifecycle, `WARNING` for
  degraded, `ERROR` for failures. Nothing important logged at `DEBUG`.
- No `print()` in library code.
- No secret, credential, full prompt with customer data, or unmasked PAN in any log line.
- Is anything logged in a hot loop that will drown the output?

## 8. Test coverage

- Does every acceptance criterion have a test that would fail if the criterion were broken?
- Negative cases present: invalid input, empty input, permission denied, guardrail block.
- For guardrails: **both** a must-block and a must-allow case.
- Are tests asserting on structure and routing rather than on model prose?
- Was any existing test weakened, skipped, deleted, or made `xfail` in this diff? If so,
  that is a finding — check the [`testing`](../testing/SKILL.md) skill's rule.
- Do tests actually run without network access and without an API key?
- Is there a new eval case in the golden dataset if this changes AI behaviour?

## 9. Unnecessary complexity

- Is there an abstraction with exactly one implementation? Inline it.
- A config flag nobody sets? A hook nobody calls? Remove it.
- Generality added for a hypothetical future requirement? Delete it.
- A dependency added for something 15 lines of stdlib would do?
- Could this be a pure function instead of a class holding state?
- Is the control flow deeper than it needs to be — can a guard clause flatten it?

---

## Reporting

Group findings by severity and be concrete:

```
BLOCKER
  src/bankassist/agents/dispute.py:88
  get_customer_transactions() ignores the customer_id argument and returns all rows.
  Failure: asking about customer C-002's disputes returns C-001's transactions.
  Fix: add WHERE customer_id = ? and add tests/unit/test_tool_scoping.py.

MAJOR
  src/bankassist/rag/pipeline.py:41
  Retrieval failures are caught with `except Exception: return []`, so an unreachable
  vector store looks identical to "no relevant documents" and the model answers unsourced.
  Fix: re-raise as RetrievalError; surface as a degraded response, not an empty context.

MINOR
  src/bankassist/tracing/span.py:23
  Span duration computed with time.time(); use time.perf_counter() for monotonicity.

NIT
  src/bankassist/config.py:12 — `RERANK_TOP_N` would read better as `RERANK_TOP_K`
  to match the rest of the retrieval config.
```

If the review found nothing of substance, say so plainly — do not manufacture findings to
look thorough. A clean review of a small, focused diff is a normal outcome.

Then fix the blockers and majors, re-run the tests, and carry the remainder into the change
report's "known limitations" section.
