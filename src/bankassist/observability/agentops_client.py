"""AgentOps SDK lifecycle: init once at process startup, no-op otherwise.

Deliberately conservative: disabled unless ``Settings.agentops_enabled`` is
true *and* a non-blank ``AGENTOPS_API_KEY`` is configured — the same "off
unless explicitly configured" posture as Pinecone (``config.py``). A failure
to reach AgentOps is logged and swallowed; it must never take down the
banking application it is observing (Lab 6 requirements §21).
"""

from __future__ import annotations

import atexit

from bankassist.config import Settings
from bankassist.logging_config import get_logger

logger = get_logger(__name__)

_initialized = False


def init_agentops(settings: Settings) -> bool:
    """Initialize the AgentOps SDK once. Returns whether it is now active.

    Safe to call multiple times (e.g. once per app factory in tests) — only
    the first call with a valid configuration does anything.
    """
    global _initialized
    if _initialized:
        return True
    if not settings.agentops_enabled:
        return False
    if not settings.has_agentops_credential():
        logger.warning(
            "AGENTOPS_ENABLED is true but AGENTOPS_API_KEY is not set; "
            "observability stays disabled."
        )
        return False

    _exclude_broken_langgraph_node_instrumentation()

    try:
        import agentops

        agentops.init(
            api_key=settings.agentops_api_key.get_secret_value(),
            default_tags=[settings.agentops_project, settings.agentops_environment],
            auto_start_session=True,
            # Custom BankAssist attributes are sanitized ourselves
            # (observability/redaction.py) before being attached; ambient
            # process/environment metadata is not something this banking
            # teaching app needs to send off-box.
            env_data_opt_out=True,
            fail_safe=True,
            log_level="WARNING",
        )
    except Exception:
        logger.exception("AgentOps initialization failed; continuing without it.")
        return False

    _initialized = True
    atexit.register(_shutdown_quietly)
    logger.info(
        "AgentOps initialized",
        extra={"project": settings.agentops_project, "environment": settings.agentops_environment},
    )
    return True


def _exclude_broken_langgraph_node_instrumentation() -> None:
    """Work around a confirmed bug in ``agentops==0.4.21``'s LangGraph auto-
    instrumentation, verified against a running instance of this app before
    this workaround was added.

    ``agentops.instrumentation.agentic.langgraph.instrumentation
    .LanggraphInstrumentor._wrap_add_node`` replaces every graph node
    function with a wrapper accepting only ``(state)``, silently dropping
    ``config``. Every BankAssist node accepts ``(state, config)`` — the
    standard, documented way a LangGraph node reads
    ``config["configurable"]`` (this project's ``SecurityContext``, ADR-0010)
    — so every single graph invocation raised ``TypeError: ...got an
    unexpected keyword argument 'config'`` with AgentOps enabled. This is not
    specific to BankAssist: any LangGraph node using ``config`` breaks the
    same way.

    A first attempt called ``instrumentor.uninstrument()`` *after*
    ``agentops.init()`` — that did not hold: AgentOps installs a global
    ``builtins.__import__`` hook that re-scans and silently re-instruments
    LangGraph on the *next* relevant import, regardless of an explicit
    uninstall. The fix instead removes ``"langgraph"`` from AgentOps'
    ``AGENTIC_LIBRARIES``/``TARGET_PACKAGES`` registries *before*
    ``agentops.init()`` runs, so LangGraph is never targeted for
    instrumentation in the first place — verified by reproducing the
    original `TypeError` against a real `graph.invoke()` call, then
    confirming it no longer occurs with this workaround applied.

    OpenAI auto-instrumentation and every custom span in
    ``observability/decorators.py`` (which wrap functions directly, not via
    LangGraph's ``add_node``) are unaffected. Revisit this workaround once a
    newer ``agentops`` release fixes the upstream bug (see ADR-0012's
    "Revisit when").
    """
    try:
        from agentops.instrumentation import AGENTIC_LIBRARIES, TARGET_PACKAGES

        AGENTIC_LIBRARIES.pop("langgraph", None)
        TARGET_PACKAGES.discard("langgraph")
        logger.warning(
            "Excluded LangGraph from AgentOps' auto-instrumentation: it is "
            "incompatible with LangGraph nodes that accept a `config` "
            "parameter (this project's HITL/security-context pattern). "
            "OpenAI call tracing and BankAssist's own custom spans are "
            "unaffected. See observability/agentops_client.py."
        )
    except Exception:
        # Best-effort workaround reaching into agentops' internal module
        # state — if that structure changes in a future release, the
        # `TypeError` this exists to prevent may resurface. Logged at
        # WARNING (not silently swallowed at DEBUG) because that is a
        # real regression, not routine degraded telemetry.
        logger.warning(
            "Could not apply the LangGraph instrumentation workaround; graph "
            "invocations may fail while AgentOps is enabled. See "
            "observability/agentops_client.py.",
            exc_info=True,
        )


def is_enabled() -> bool:
    """True once ``init_agentops`` has successfully activated the SDK."""
    return _initialized


def shutdown_agentops() -> None:
    """Flush and end the active AgentOps trace. Never raises."""
    global _initialized
    if not _initialized:
        return
    _shutdown_quietly()
    _initialized = False


def _shutdown_quietly() -> None:
    try:
        import agentops

        agentops.end_trace()
    except Exception:
        logger.debug("AgentOps shutdown raised; ignoring.", exc_info=True)


def reset_for_tests() -> None:
    """Test-only hook: force ``is_enabled()`` back to False between tests."""
    global _initialized
    _initialized = False
