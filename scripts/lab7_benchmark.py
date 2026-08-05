"""Lab 7 before/after benchmark: cache disabled vs enabled (ADR-0013).

Runs the *real* policy-node code path (`agents.graph.build_graph`'s policy node,
through `SemanticCache.lookup`/`store`) a fixed number of times with a repeated
question, once with the semantic cache disabled and once enabled, and reports
the measured wall-clock latency difference.

This deliberately uses `StubLLMClient` and a fake enterprise pipeline — the
same test doubles the automated test suite uses — rather than live OpenAI
calls, so the benchmark is free to run and reproducible in CI. It measures the
*actual* effect of the caching code path on latency (a real, not extrapolated,
number): the LLM/RAG latency itself is a fixed stub delay standing in for a
real generation call, so the relative latency drop this script reports is
real, even though the absolute stub latency is not. Estimated *dollar* costs
still come from the documented demo assumptions in
`docs/plan/lab-07-cost-optimization-plan.md` §7, applied to the exact hit/miss
counts this script measures — never invented separately.

Usage:
    python scripts/lab7_benchmark.py [--requests 20] [--redis-url redis://localhost:6379/0]

With no reachable Redis, `--redis-url` is ignored and the run falls back to an
in-process `fakeredis` instance so the script still runs end to end — this is
called out in the printed report, not hidden.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from bankassist.agents.graph import build_graph
from bankassist.caching.semantic_cache import SemanticCache
from bankassist.config import Settings
from bankassist.guardrails.models import GuardrailCategory, GuardrailResult
from bankassist.llm.stub import StubLLMClient
from bankassist.security.context import SecurityContext
from bankassist.tools import banking_data

POLICY_ROUTE = '{"route": "POLICY", "confidence": 0.9, "reason": "policy question"}'
QUESTION = "What is the dispute window for a credit card transaction?"

# Demo cost assumptions (Lab 7 plan §7) — applied to real measured hit counts,
# not invented independently of them.
_ASSUMED_GENERATION_COST_USD = 0.0015


@dataclass
class _FakeResult:
    generated_answer: str = "You have 30 days to dispute a card transaction."
    citations: list[str] = field(default_factory=lambda: ["dispute-policy.md"])


class _FakePipeline:
    """Stands in for `EnterpriseRagPipeline` — deterministic, no network call.

    Records how many times it actually ran, so a semantic-cache hit is
    verified by *this pipeline never being invoked*, not just by a faster
    clock.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def answer(self, question: str) -> _FakeResult:  # noqa: ARG002
        self.call_count += 1
        time.sleep(0.05)  # stands in for real RAG + LLM generation latency
        return _FakeResult()


class _AllowAllNemo:
    """Minimal standalone double — always allows. Not the shared test fixture
    (`tests/support/fake_nemo.py`) since this script must run outside pytest."""

    def check_input(self, message: str) -> GuardrailResult:  # noqa: ARG002
        return GuardrailResult.allow("nemo.input_rail.self_check", GuardrailCategory.INPUT)

    def check_output(self, answer: str) -> GuardrailResult:  # noqa: ARG002
        return GuardrailResult.allow("nemo.output_rail.self_check", GuardrailCategory.OUTPUT)


def _redis_client(redis_url: str):
    try:
        import redis

        client = redis.Redis.from_url(redis_url, socket_connect_timeout=0.5)
        client.ping()
        return client, "real Redis"
    except Exception:
        import fakeredis

        return fakeredis.FakeStrictRedis(), "fakeredis (no reachable Redis at --redis-url)"


def _run(*, requests: int, semantic_cache: SemanticCache | None, db_path: Path) -> dict:
    llm = StubLLMClient([POLICY_ROUTE] * requests)
    pipeline = _FakePipeline()
    graph = build_graph(
        llm=llm,
        enterprise_pipeline=pipeline,
        db_path=db_path,
        nemo=_AllowAllNemo(),
        semantic_cache=semantic_cache,
    )

    context = SecurityContext(
        user_id="bench",
        role="CUSTOMER",
        customer_id="CUST-BENCH",
        session_id="bench",
        request_id="r1",
    )
    latencies_ms = []
    for i in range(requests):
        config = {"configurable": {"thread_id": f"bench-{i}", "security_context": context}}
        started = time.perf_counter()
        graph.invoke({"messages": [{"role": "user", "content": QUESTION}]}, config)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        "latencies_ms": latencies_ms,
        "pipeline_calls": pipeline.call_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    args = parser.parse_args()

    db_path = Path(".lab7_benchmark.db")
    if db_path.exists():
        db_path.unlink()
    banking_data.session(db_path).__enter__()  # creates the schema

    settings = Settings(openai_api_key="sk-bench-not-real", redis_enabled=True, _env_file=None)

    print(f"Lab 7 benchmark — {args.requests} identical policy requests\n")

    # --- Baseline: no semantic cache at all (Labs 1-6 behavior) ---
    baseline = _run(requests=args.requests, semantic_cache=None, db_path=db_path)

    # --- Optimized: semantic cache enabled ---
    client, client_kind = _redis_client(args.redis_url)
    client.flushdb()
    embed = lambda text: [float(len(text) % 7), 1.0, 0.0]  # noqa: E731 - tiny deterministic stub
    cache = SemanticCache(client, settings, embed, redisearch_available=False)
    optimized = _run(requests=args.requests, semantic_cache=cache, db_path=db_path)

    db_path.unlink(missing_ok=True)

    def _summary(label: str, result: dict) -> None:
        lat = result["latencies_ms"]
        print(f"{label}:")
        print(f"  pipeline (RAG+LLM) invocations: {result['pipeline_calls']} / {args.requests}")
        print(f"  mean latency:   {statistics.mean(lat):.1f} ms")
        print(f"  median latency: {statistics.median(lat):.1f} ms")
        print(f"  p95 latency:    {sorted(lat)[int(len(lat) * 0.95) - 1]:.1f} ms\n")

    print(f"Redis backend used for the optimized run: {client_kind}\n")
    _summary("BEFORE (no semantic cache)", baseline)
    _summary("AFTER  (semantic cache enabled)", optimized)

    calls_saved = baseline["pipeline_calls"] - optimized["pipeline_calls"]
    latency_saved_ms = statistics.mean(baseline["latencies_ms"]) - statistics.mean(
        optimized["latencies_ms"]
    )
    print(f"Measured pipeline calls avoided: {calls_saved} / {args.requests}")
    print(f"Measured mean latency reduction: {latency_saved_ms:.1f} ms")
    print(
        f"Estimated cost saved (demo assumption, ${_ASSUMED_GENERATION_COST_USD}/generation "
        f"call, applied to the {calls_saved} calls actually avoided above): "
        f"${calls_saved * _ASSUMED_GENERATION_COST_USD:.4f}"
    )


if __name__ == "__main__":
    main()
