---
name: testing
description: How to identify, write, run, and report tests in BankAssist AI — unit, integration, guardrail, and AI-evaluation tests. Use when adding tests, when a test fails, or when deciding what a change needs covered. Enforces the rule that tests are never weakened, skipped, or deleted to obtain a passing result.
---

# Testing

## The one rule that matters

**Never weaken a test to make it pass.**

Loosening an assertion, lowering a guardrail threshold, adding an allowlist entry, marking
a test `xfail`/`skip`, deleting a case, or stubbing out the thing under test — all of these
convert a real defect into a hidden one. In a project whose subject is *governance*, a
green suite that lies is worse than a red one that tells the truth.

If a test fails: fix the code, or report the failure with its output and stop. Changing a
test is legitimate only when the **requirement** changed — and then you say so explicitly
in the change report and point at the spec that changed.

---

## 1. Identify what to test

Work from the acceptance criteria in the spec, then add what the criteria miss.

For any change, ask:

- What is the **happy path**, and what does correct output look like precisely?
- What are the **boundaries**? Empty input, single item, oversized input, unicode,
  missing optional fields, `None`.
- What must be **refused**? Every guardrail needs a must-block case.
- What must **not** be refused? Every guardrail needs a must-allow case — over-blocking
  is a defect. A guardrail that blocks "explain how APR is calculated" has failed.
- What is **deterministic** and therefore directly testable without an LLM?
- What **breaks silently**? Cache key collisions, prompt-prefix drift, trace spans not
  emitted, cross-customer data leakage.

### Coverage map for this project

| Area | Test kind | Must cover |
|---|---|---|
| Chunking | unit | boundaries, overlap, metadata preserved, empty doc |
| Embedding / vector store | integration | round-trip: ingest → query → expected doc ranks top |
| BM25 keyword search | unit | exact-term match beats semantic-only, tokenization |
| Hybrid fusion + reranking | unit | ordering, score fusion math, top-n truncation, ties |
| Metadata filtering | unit | filter narrows results; empty result set handled |
| Citations | unit | every cited id exists in retrieved context; no fabricated ids |
| Supervisor routing | unit (stubbed LLM) | each intent routes to the intended agent; ambiguous input |
| Agent tools | unit | schema validation, `customer_id` scoping, error paths |
| **Cross-customer access** | unit | a tool called with customer A never returns customer B's data |
| Input guardrails | unit | prompt injection blocked; jailbreak blocked; benign question allowed |
| Financial-safety guardrail | unit | "explain index funds" allowed; "put $50k into NVDA" blocked |
| PII guardrails | unit | unmasked PAN / SSN-shaped strings caught in input and output |
| RAG guardrail | unit | instruction-shaped text inside a retrieved doc does not change behaviour |
| Output grounding | unit + eval | unsupported claim flagged |
| Tracing | unit | every LLM/tool/retrieval/guardrail call produces a span with required attributes |
| Cost calculation | unit | token counts × price table = expected cost, exactly |
| Prompt caching | integration | second identical call reports `cache_read_input_tokens > 0` |
| Semantic cache | unit | near-duplicate query hits; unrelated query misses; threshold boundary |
| API layer | integration | status codes, response schema, error shape |

## 2. Test structure

```
tests/
├─ conftest.py            # shared fixtures: stub LLM, temp Chroma, seeded SQLite
├─ unit/                  # fast, no I/O, no network
├─ integration/           # real vector store + real SQLite, stubbed LLM
├─ guardrails/            # must-block / must-allow pairs
└─ live/                  # marked @pytest.mark.live — real API, not run by default
```

Conventions:

- `test_<unit>_<condition>_<expected>` — e.g. `test_rerank_empty_input_returns_empty`.
- Arrange / Act / Assert, visually separated.
- One behaviour per test. A test that asserts five unrelated things tells you nothing
  useful when it fails.
- Parametrize instead of copy-pasting near-identical cases.
- Fixtures over setup boilerplate. Deterministic seeds everywhere.

## 3. Stubbing the LLM

Unit and integration tests must not call a hosted API. They are slow, non-deterministic,
cost money, and fail in CI without a key.

Every LLM call in this repo goes through the `LLMClient` interface, so tests inject a
`StubLLMClient` that returns scripted responses (and records the calls it received, so you
can assert on the prompt shape and the tools offered).

- Assert on **structure and routing**, not on model prose. `assert "APR" in response.text`
  is a flaky test; `assert routed_agent == "policy"` is not.
- Model quality is measured by the **evaluation suite** (Lab 6), not by unit tests.
- Tests that genuinely need a live model go in `tests/live/`, are marked
  `@pytest.mark.live`, and are excluded from the default run.

## 4. Running tests

```bash
python -m pytest -q
```

With coverage:

```bash
python -m pytest --cov=src/bankassist --cov-report=term-missing
```

A single area:

```bash
python -m pytest tests/guardrails -v
```

Including live tests (requires an API key, costs money):

```bash
python -m pytest -m live
```

Lint must pass too:

```bash
python -m ruff check .
```

## 5. When a test fails

Work the problem in this order:

1. **Read the actual failure.** Assertion message, diff, traceback. Do not guess.
2. **Reproduce it in isolation** — run that one test with `-vv`.
3. **Decide which side is wrong**: the code, or the expectation.
   - Code is wrong → fix the code.
   - Expectation is wrong *because the requirement changed* → update the spec first, then
     the test, and call it out in the change report.
   - Expectation is wrong *because the test was written badly* (wrong fixture, bad
     assumption) → fix the test and explain why in the report.
4. **Never** take a fourth option that involves making the assertion less strict.
5. Re-run the full suite, not just the one test — confirm you did not trade one failure
   for another.

### Flaky tests

A flaky test is a bug in the test or a real race in the code. Do not add retries to hide
it. Common causes here: unseeded randomness, wall-clock time in a cached prefix, shared
Chroma/SQLite state between tests, and asserting on model prose.

## 6. Evaluation tests (Lab 6)

The evaluation suite is separate from `pytest` correctness tests. It measures AI quality
against a golden dataset and produces a report, not a pass/fail exit code — except for
regression thresholds, which *are* enforced.

- Golden dataset: `evaluation/golden/*.yaml` — question, expected behaviour, expected
  source documents, expected refusal/allow verdict.
- Metrics: answer relevance, groundedness, context relevance, citation correctness,
  guardrail success rate, task completion, latency, tokens, cost.
- Scorers: deterministic where possible (citation correctness, guardrail success are exact
  checks); LLM-as-judge only where a human judgement is genuinely required.
- Run: `python scripts/run_eval.py` → writes `evaluation/reports/<timestamp>.md`.
- **Regression rule:** if a metric drops below its recorded threshold, that is a failure to
  report — not a threshold to lower.

## 7. Reporting results

Always report what actually happened:

```
Command:  python -m pytest -q
Result:   118 passed, 2 failed, 4 skipped in 21.3s
Failures: tests/guardrails/test_financial_safety.py::test_allows_general_education
          — over-blocking: "explain how index funds work" was refused
Coverage: src/bankassist 84% (guardrails 96%, retrieval 89%, agents 71%)
```

Never write "all tests pass" without having run them in this session. If you could not run
them, say that instead — that is a legitimate report; a fabricated one is not.
