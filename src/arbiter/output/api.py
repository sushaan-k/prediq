"""FastAPI REST API for programmatic access to arbiter analytics.

Exposes endpoints for querying divergences, violations, liquidity,
and market quality. Designed for integration with trading systems,
dashboards, and research tools.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Query

from arbiter.exceptions import InsufficientDataError
from arbiter.models import (
    Divergence,
    LiquidityProfile,
    MarketQuality,
    MultiOutcomeViolation,
    ProbabilityViolation,
)

logger = logging.getLogger(__name__)

# In-memory store for the latest analytics results.
# In production, this would be backed by the storage layer.
_state: dict[str, Any] = {
    "divergences": [],
    "binary_violations": [],
    "multi_violations": [],
    "liquidity": {},
    "quality": {},
}

# The Arbiter instance used by the API server. Populated at startup.
_arbiter_instance: Any = None

# Background tasks that must survive garbage collection.
_background_tasks: set[asyncio.Task[None]] = set()

_REFRESH_INTERVAL_SECONDS = 30.0


def update_state(
    divergences: list[Divergence] | None = None,
    binary_violations: list[ProbabilityViolation] | None = None,
    multi_violations: list[MultiOutcomeViolation] | None = None,
    liquidity: dict[str, LiquidityProfile] | None = None,
    quality: dict[str, MarketQuality] | None = None,
) -> None:
    """Update the API state with fresh analytics results.

    Called by the orchestrator after each analytics cycle.

    Args:
        divergences: Latest cross-exchange divergences.
        binary_violations: Latest binary probability violations.
        multi_violations: Latest multi-outcome violations.
        liquidity: Market ID -> LiquidityProfile mapping.
        quality: Exchange name -> MarketQuality mapping.
    """
    if divergences is not None:
        _state["divergences"] = divergences
    if binary_violations is not None:
        _state["binary_violations"] = binary_violations
    if multi_violations is not None:
        _state["multi_violations"] = multi_violations
    if liquidity is not None:
        _state["liquidity"] = liquidity
    if quality is not None:
        _state["quality"] = quality


async def _populate_state() -> None:
    """Run one analytics cycle and populate ``_state``."""
    global _arbiter_instance
    arbiter = _arbiter_instance
    if arbiter is None:
        return

    divergences: list[Divergence] = []
    binary_v: list[ProbabilityViolation] = []
    multi_v: list[MultiOutcomeViolation] = []
    liquidity: dict[str, LiquidityProfile] = {}
    quality: dict[str, MarketQuality] = {}

    try:
        divergences = await arbiter.divergences(min_spread=0.0)
    except Exception:
        logger.exception("Failed to refresh divergences")

    try:
        binary_v, multi_v = await arbiter.violations()
    except Exception:
        logger.exception("Failed to refresh violations")

    for exchange_name, markets in arbiter._market_cache.items():
        for market in markets:
            try:
                profile = await arbiter.liquidity(exchange_name, market.id)
            except Exception:
                logger.debug(
                    "Failed to compute liquidity for %s/%s",
                    exchange_name,
                    market.id,
                    exc_info=True,
                )
            else:
                liquidity[market.id] = profile

        try:
            quality[exchange_name] = await arbiter.quality(
                exchange=exchange_name,
                category="all",
            )
        except InsufficientDataError:
            logger.debug(
                "Skipping quality for %s due to insufficient data",
                exchange_name,
            )
        except Exception:
            logger.debug(
                "Failed to compute quality for %s",
                exchange_name,
                exc_info=True,
            )

    update_state(
        divergences=divergences,
        binary_violations=binary_v,
        multi_violations=multi_v,
        liquidity=liquidity,
        quality=quality,
    )
    logger.info(
        "API state populated: %d divergences, %d binary violations, "
        "%d multi violations, %d liquidity profiles, %d quality scores",
        len(divergences),
        len(binary_v),
        len(multi_v),
        len(liquidity),
        len(quality),
    )


async def _refresh_state_loop() -> None:
    """Continuously refresh API state until cancelled."""
    while True:
        await _populate_state()
        await asyncio.sleep(_REFRESH_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(application: FastAPI):  # type: ignore[no-untyped-def]
    """Application lifespan: create Arbiter, run initial scan, then clean up."""
    global _arbiter_instance
    refresh_task: asyncio.Task[None] | None = None
    try:
        from arbiter.engine import Arbiter
        from arbiter.exchanges.manifold import ManifoldExchange
        from arbiter.exchanges.polymarket import PolymarketExchange

        _arbiter_instance = Arbiter(
            exchanges=[PolymarketExchange(), ManifoldExchange()]
        )
        # Run initial population in a background task so startup isn't blocked
        # if the network is slow; errors are logged, not raised.
        refresh_task = asyncio.create_task(_refresh_state_loop())
        _background_tasks.add(refresh_task)
        refresh_task.add_done_callback(_background_tasks.discard)
    except Exception:
        logger.exception("Failed to initialize Arbiter for API server")
    yield
    # Shutdown
    if refresh_task is not None:
        refresh_task.cancel()
        with suppress(asyncio.CancelledError):
            await refresh_task
    if _arbiter_instance is not None:
        await _arbiter_instance.close()
        _arbiter_instance = None


app = FastAPI(
    title="arbiter",
    description="Cross-exchange prediction market analytics API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "arbiter"}


@app.get("/api/v1/divergences", response_model=list[Divergence])
async def get_divergences(
    min_spread: float = Query(0.0, ge=0.0, description="Minimum spread filter"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
) -> list[Divergence]:
    """Get current cross-exchange price divergences.

    Args:
        min_spread: Only return divergences with spread >= this value.
        limit: Maximum number of results.

    Returns:
        List of Divergence objects sorted by spread descending.
    """
    divs: list[Divergence] = _state["divergences"]
    filtered = [d for d in divs if d.spread >= min_spread]
    return filtered[:limit]


@app.get("/api/v1/violations/binary", response_model=list[ProbabilityViolation])
async def get_binary_violations(
    limit: int = Query(50, ge=1, le=500),
) -> list[ProbabilityViolation]:
    """Get current binary market probability violations.

    Returns:
        List of ProbabilityViolation objects.
    """
    violations: list[ProbabilityViolation] = _state["binary_violations"]
    return violations[:limit]


@app.get(
    "/api/v1/violations/multi",
    response_model=list[MultiOutcomeViolation],
)
async def get_multi_violations(
    limit: int = Query(50, ge=1, le=500),
) -> list[MultiOutcomeViolation]:
    """Get current multi-outcome probability violations.

    Returns:
        List of MultiOutcomeViolation objects.
    """
    violations: list[MultiOutcomeViolation] = _state["multi_violations"]
    return violations[:limit]


@app.get("/api/v1/liquidity/{market_id}", response_model=LiquidityProfile | None)
async def get_liquidity(market_id: str) -> LiquidityProfile | None:
    """Get liquidity profile for a specific market.

    Args:
        market_id: Exchange-native market identifier.

    Returns:
        LiquidityProfile if available, None otherwise.
    """
    profiles: dict[str, LiquidityProfile] = _state["liquidity"]
    return profiles.get(market_id)


@app.get("/api/v1/quality/{exchange}", response_model=MarketQuality | None)
async def get_quality(exchange: str) -> MarketQuality | None:
    """Get market quality metrics for an exchange.

    Args:
        exchange: Exchange name (e.g. 'polymarket', 'kalshi').

    Returns:
        MarketQuality metrics if available, None otherwise.
    """
    scores: dict[str, MarketQuality] = _state["quality"]
    return scores.get(exchange.lower())
