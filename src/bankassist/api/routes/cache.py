"""Cache statistics endpoint (Lab 7, ADR-0013, FR-7.x)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from bankassist.api.schemas import CacheStatsResponse
from bankassist.caching.stats import get_stats
from bankassist.config import Settings

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/stats", response_model=CacheStatsResponse, summary="Lab 7 cache hit/miss statistics")
def cache_stats(request: Request) -> CacheStatsResponse:
    """Aggregate semantic/embedding/tool cache counters, and estimated savings.

    Numbers are exact counts of what actually happened in this Redis instance —
    not the extrapolated demo estimates in the Lab 7 documentation, which are
    clearly labelled as assumptions there.
    """
    settings: Settings = request.app.state.settings
    client = getattr(request.app.state, "redis_client", None)
    stats = get_stats(client, settings)
    return CacheStatsResponse(**stats.model_dump())
