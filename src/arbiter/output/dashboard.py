"""Web dashboard for real-time prediction market analytics.

Provides a simple FastAPI-based dashboard that serves a live view
of divergences, violations, and market quality scores.
For v1, this is a JSON API that can be consumed by any frontend.
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Any

from fastapi import FastAPI

from arbiter.models import Divergence, LiquidityProfile, MarketQuality

logger = logging.getLogger(__name__)


def create_dashboard_app() -> FastAPI:
    """Create a FastAPI application for the dashboard.

    Returns:
        Configured FastAPI application.
    """
    dashboard = FastAPI(
        title="arbiter dashboard",
        description="Real-time prediction market analytics dashboard",
        version="0.1.0",
    )

    _dashboard_state: dict[str, Any] = {
        "divergences": [],
        "quality_scores": {},
        "liquidity_profiles": {},
        "last_updated": None,
    }

    @dashboard.get("/dashboard/summary")
    async def dashboard_summary() -> dict[str, Any]:
        """Get a summary of current analytics state.

        Returns:
            Summary with counts and top divergences.
        """
        divs: list[Divergence] = _dashboard_state["divergences"]
        quality: dict[str, MarketQuality] = _dashboard_state["quality_scores"]

        return {
            "total_divergences": len(divs),
            "top_divergences": [
                {
                    "event": d.event,
                    "spread": d.spread,
                    "spread_pct": d.spread_pct,
                    "exchanges": f"{d.exchange_a.value} vs {d.exchange_b.value}",
                }
                for d in sorted(divs, key=lambda x: x.spread, reverse=True)[:10]
            ],
            "exchanges_scored": list(quality.keys()),
            "last_updated": _dashboard_state["last_updated"],
        }

    @dashboard.get("/dashboard/leaderboard")
    async def exchange_leaderboard() -> list[dict[str, Any]]:
        """Get exchange accuracy leaderboard.

        Returns:
            List of exchanges ranked by Brier score (lower is better).
        """
        quality: dict[str, MarketQuality] = _dashboard_state["quality_scores"]
        ranked = sorted(quality.values(), key=lambda q: q.brier_score)
        return [
            {
                "exchange": q.exchange.value,
                "brier_score": q.brier_score,
                "calibration_error": q.calibration_error,
                "sample_size": q.sample_size,
            }
            for q in ranked
        ]

    def update_dashboard(
        divergences: list[Divergence] | None = None,
        quality_scores: dict[str, MarketQuality] | None = None,
        liquidity_profiles: dict[str, LiquidityProfile] | None = None,
    ) -> None:
        """Update the dashboard state with fresh data.

        Args:
            divergences: Latest divergences.
            quality_scores: Exchange quality scores.
            liquidity_profiles: Market liquidity profiles.
        """
        from datetime import datetime

        if divergences is not None:
            _dashboard_state["divergences"] = divergences
        if quality_scores is not None:
            _dashboard_state["quality_scores"] = quality_scores
        if liquidity_profiles is not None:
            _dashboard_state["liquidity_profiles"] = liquidity_profiles
        _dashboard_state["last_updated"] = datetime.now(UTC).isoformat()

    dashboard.update_state = update_dashboard  # type: ignore[attr-defined]
    return dashboard
