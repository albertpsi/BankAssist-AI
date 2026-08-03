# ADR-0005 — OpenAI as the initial provider behind an LLMClient abstraction

**Status:** Accepted (amended 2026-08-03 — supersedes the "Anthropic primary" proposal)
**Date:** 2026-08-03

## Context

Lab 7 requires demonstrating cost optimization with measurable before/after figures. The
original draft of this ADR recommended Anthropic as the primary provider, because its
explicit `cache_control` breakpoint and `cache_read_input_tokens` usage field make
provider-native prompt caching directly observable — turning that part of Lab 7 from an
assertion into a measurement.

Environment inspection found `OPENAI_API_KEY` set and `ANTHROPIC_API_KEY` **not** set, with
no `ant` CLI and therefore no OAuth profile an SDK could resolve.

At the planning approval gate the trade-off was resolved in favour of **not making a second
API credential a prerequisite for the project**.

## Decision

**Use the existing OpenAI credential as the initial application provider.** Route all model
access through a single `LLMClient` interface so an `AnthropicClient` adapter can be added
later without touching a call site.

Lab 7's measurable core moves to levers that are provider-independent:

| Lever | Provider-dependent? | Role in Lab 7 |
|---|---|---|
| Semantic caching | No | Primary evidence |
| Model-tier routing | No | Primary evidence |
| Token accounting | No | Primary evidence |
| Latency measurement | No | Primary evidence |
| Estimated cost comparison | No | Primary evidence |
| Provider-native prompt caching | Yes | **Best-effort.** Demonstrated and reported where the provider exposes a measurable signal; recorded as a documented finding where it does not |

**Model tiering (see also NFR-12):** an economical model (`LLM_MODEL_FAST`) is the default
for classification, query rewriting, guardrail classification, and routine generation. A
stronger model (`LLM_MODEL_STRONG`) is **optional** and used only for selected LLM-as-judge
evaluation cases; if unset, the fast model is used everywhere and the system still works end
to end. Model ids and the price table are configuration, read once in `config.py`, validated
at startup, and never hard-coded at a call site.

## Alternatives considered

**Anthropic primary** — better prompt-caching observability, and still the better answer if
that single lever were the deciding factor. Rejected: it makes the project depend on
acquiring a credential that does not exist, to strengthen one of six Lab 7 levers. The other
five are unaffected by provider, so the marginal evidence does not justify the dependency.

**Both providers from the start** — build the Anthropic adapter now and switch if a key
appears. Rejected as scope creep: a second adapter with no key to test it against is
untested code.

**LiteLLM** (installed, v1.55.8) — provider abstraction for free. Rejected: a normalizing
layer sits exactly where Lab 7 wants precise token and cache accounting, and the write-up
would end up explaining LiteLLM's passthrough behaviour rather than the caching architecture.
A ~120-line interface with one adapter is more code but a better artifact.

**Direct SDK calls at each site** — simplest to write. Rejected: makes the provider
unswappable, scatters token and cost accounting, and makes LLM stubbing in tests impossible
without monkey-patching. The single chokepoint is what makes Labs 6 and 7 tractable at all.

**A frontier model as the default tier** — Rejected: the labs grade architecture, not model
horsepower, and an expensive default would make iteration costly for no marginal evidence.

## Consequences

**Makes easy:** the project runs today with zero new credentials. Provider swap is a config
change plus one adapter. Every LLM call is stubbed in tests through one seam. Token, latency,
and cost accounting happen in exactly one place.

**Makes hard:** provider-native prompt caching is less directly observable, so that specific
Lab 7 sub-claim is best-effort rather than guaranteed. Provider-specific features need
adapter-level handling rather than direct use.

**Accepted:** if the prompt-cache signal turns out not to be measurable, Lab 7 reports that
as a finding — with the prefix-stability discipline still applied, since it costs nothing and
is good hygiene regardless.

## Revisit when

An Anthropic key becomes available and the prompt-caching evidence is wanted; a third
provider is needed; streaming is added (currently deferred); or a provider capability is
needed that the thin interface cannot express without leaking.
