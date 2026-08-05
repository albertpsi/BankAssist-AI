"""FastAPI application factory.

A factory rather than a module-level app, so tests build isolated instances with
their own settings instead of mutating a global.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from bankassist.api.routes import agent, auth, health, rag
from bankassist.api.schemas import ErrorDetail, ErrorResponse
from bankassist.config import Settings, get_settings
from bankassist.context import get_trace_id, trace_context
from bankassist.errors import BankAssistError
from bankassist.logging_config import configure_logging, get_logger
from bankassist.observability import init_agentops
from bankassist.tracing.span import SpanType
from bankassist.tracing.tracer import build_tracer

logger = get_logger(__name__)

TRACE_HEADER = "X-Trace-Id"


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the one error envelope every failure path returns.

    ``headers`` carries through any protocol headers the original exception set —
    ``Allow`` on a 405, ``WWW-Authenticate`` on a 401. Replacing the body with our
    envelope must not strip headers the HTTP spec requires or clients depend on.
    """
    trace_id = get_trace_id()
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details or {}),
        trace_id=trace_id,
    )
    merged = dict(headers or {})
    if trace_id:
        merged[TRACE_HEADER] = trace_id
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        headers=merged or None,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    app = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        description="A governed, multi-agent banking assistant. Lab project; synthetic data only.",
    )
    app.state.settings = resolved
    app.state.tracer = build_tracer(enabled=resolved.tracing_enabled)
    # AgentOps (Lab 6, ADR-0012): no-ops unless AGENTOPS_ENABLED and a key are
    # both configured — never blocks startup, and inert for every test.
    init_agentops(resolved)

    app.include_router(health.router)
    app.include_router(rag.router)
    # FR-L3-12.1 names `/api/v1/rag/query`; Lab 2's unversioned `/rag/query` stays
    # working unchanged (NFR-L3-4) — the same route is mounted a second time
    # under the version prefix rather than replaced (lab-03 spec §6.2).
    app.include_router(rag.router, prefix="/api/v1")
    # Lab 4: local auth + multi-agent chat, both versioned-only (no Lab 2/3-style
    # unversioned alias — there is no pre-Lab-4 contract to preserve here).
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(agent.router, prefix="/api/v1")

    @app.middleware("http")
    async def trace_and_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Bind a trace id to the request, then time and log it."""
        incoming = request.headers.get(TRACE_HEADER)
        with trace_context(incoming) as trace_id:
            tracer = request.app.state.tracer
            started = time.perf_counter()
            try:
                with tracer.span(SpanType.REQUEST, f"{request.method} {request.url.path}") as span:
                    response = await call_next(request)
                    span.set_attribute("status_code", response.status_code)
            except Exception as exc:
                # Deliberately handled here rather than via app.exception_handler(
                # Exception): that runs in Starlette's outermost middleware, by
                # which point this trace context has unwound — so the envelope and
                # the response header would carry no trace id, on exactly the
                # requests where correlation matters most.
                logger.exception("unhandled exception", extra={"error_type": type(exc).__name__})
                response = _error_response(500, "internal_error", "An internal error occurred.")

            duration_ms = (time.perf_counter() - started) * 1000.0
            response.headers[TRACE_HEADER] = trace_id
            logger.info(
                "request complete",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            return response

    @app.exception_handler(BankAssistError)
    async def handle_known_error(_: Request, exc: BankAssistError) -> JSONResponse:
        logger.warning(
            "handled application error",
            extra={"error_code": exc.code, "error_message": exc.message},
        )
        return _error_response(exc.http_status, exc.code, exc.message, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(exc.status_code, "http_error", str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Project each error down to JSON-safe fields; pydantic's raw `ctx` can
        # carry exception objects that JSONResponse cannot serialize.
        errors = [
            {"loc": [str(part) for part in err.get("loc", ())], "msg": err.get("msg", "")}
            for err in exc.errors()
        ]
        return _error_response(
            422, "validation_error", "Request validation failed.", {"errors": errors}
        )

    return app
