"""Lab 7 (ADR-0013): `call_tool`'s optional tool-cache wiring.

`ToolCache` enforces its own allowlist, so these tests exercise `call_tool`
with both an allowlisted deterministic label and a customer-scoped one, to
prove the dispatcher never bypasses that allowlist for either.
"""

from __future__ import annotations

import fakeredis

from bankassist.caching.tool_cache import TOOL_CACHE_ALLOWLIST, ToolCache
from bankassist.config import Settings
from bankassist.tools.dispatcher import call_tool


def _tool_cache() -> ToolCache:
    settings = Settings(openai_api_key="sk-test-not-used", redis_enabled=True, _env_file=None)
    return ToolCache(fakeredis.FakeStrictRedis(), settings)


def test_second_call_with_same_args_is_served_from_cache_without_calling_fn():
    cache = _tool_cache()
    label = next(iter(TOOL_CACHE_ALLOWLIST))
    calls = []

    def fn():
        calls.append(1)
        return {"text": "answer"}

    args = {"doc_id": "faq-1"}
    result_1, events_1 = call_tool(
        node_id="n1", label=label, fn=fn, tool_cache=cache, cache_args=args
    )
    result_2, events_2 = call_tool(
        node_id="n1", label=label, fn=fn, tool_cache=cache, cache_args=args
    )

    assert result_1 == result_2 == {"text": "answer"}
    assert len(calls) == 1  # fn only ran once; the second call was a cache hit
    # events[0] is the TOOL_STARTED marker; the cache decision is events[1]
    assert "miss" in events_1[1].summary
    assert "hit" in events_2[1].summary


def test_customer_scoped_label_is_never_cached_even_when_a_cache_is_passed():
    """Passing a `tool_cache` for a scoped tool must be inert, not a leak."""
    cache = _tool_cache()
    calls = []

    def fn():
        calls.append(1)
        return {"balance": 1000}

    args = {"customer_id": "cust-1"}
    call_tool(node_id="n1", label="get_customer_accounts", fn=fn, tool_cache=cache, cache_args=args)
    call_tool(node_id="n1", label="get_customer_accounts", fn=fn, tool_cache=cache, cache_args=args)

    assert len(calls) == 2  # fn ran both times — never served from cache


def test_no_cache_args_means_no_caching_at_all():
    """The dispatcher's default call shape (no `tool_cache`/`cache_args`) is
    unchanged from Labs 1-6 — this is the regression case."""
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result, events = call_tool(node_id="n1", label="any_tool", fn=fn)

    assert result == "ok"
    assert len(calls) == 1
    assert all(e.node_type != "cache" for e in events)
