"""Tests for arbiter.output.dashboard -- FastAPI dashboard endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from arbiter.models import (
    Divergence,
    ExchangeName,
    LiquidityProfile,
    MarketQuality,
)
from arbiter.output.dashboard import create_dashboard_app


@pytest.fixture()
def dashboard_app():
    return create_dashboard_app()


@pytest.fixture()
def client(dashboard_app) -> TestClient:
    return TestClient(dashboard_app)


class TestDashboardSummary:
    def test_empty_summary(self, client: TestClient) -> None:
        resp = client.get("/dashboard/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_divergences"] == 0
        assert data["top_divergences"] == []
        assert data["exchanges_scored"] == []
        assert data["last_updated"] is None

    def test_summary_with_data(self, client: TestClient, dashboard_app) -> None:
        divs = [
            Divergence(
                event=f"Event {i}",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.50 + 0.01 * (i + 1),
                price_b=0.50,
                spread=0.01 * (i + 1),
                spread_pct=0.02 * (i + 1),
            )
            for i in range(3)
        ]
        quality = {
            "polymarket": MarketQuality(
                exchange=ExchangeName.POLYMARKET,
                brier_score=0.18,
                calibration_error=0.03,
                manipulation_score=0.10,
                volume_accuracy_correlation=0.65,
                sample_size=100,
            )
        }
        dashboard_app.update_state(divergences=divs, quality_scores=quality)

        resp = client.get("/dashboard/summary")
        data = resp.json()
        assert data["total_divergences"] == 3
        assert len(data["top_divergences"]) == 3
        assert data["exchanges_scored"] == ["polymarket"]
        assert data["last_updated"] is not None
        # Top divergences should be sorted by spread descending
        spreads = [d["spread"] for d in data["top_divergences"]]
        assert spreads == sorted(spreads, reverse=True)


class TestExchangeLeaderboard:
    def test_empty_leaderboard(self, client: TestClient) -> None:
        resp = client.get("/dashboard/leaderboard")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_leaderboard_sorted_by_brier(
        self, client: TestClient, dashboard_app
    ) -> None:
        quality = {
            "polymarket": MarketQuality(
                exchange=ExchangeName.POLYMARKET,
                brier_score=0.25,
                calibration_error=0.05,
                manipulation_score=0.10,
                volume_accuracy_correlation=0.60,
                sample_size=80,
            ),
            "kalshi": MarketQuality(
                exchange=ExchangeName.KALSHI,
                brier_score=0.15,
                calibration_error=0.03,
                manipulation_score=0.08,
                volume_accuracy_correlation=0.70,
                sample_size=120,
            ),
        }
        dashboard_app.update_state(quality_scores=quality)

        resp = client.get("/dashboard/leaderboard")
        data = resp.json()
        assert len(data) == 2
        # Sorted by brier score (lower first)
        assert data[0]["exchange"] == "kalshi"
        assert data[1]["exchange"] == "polymarket"


class TestUpdateDashboard:
    def test_update_with_liquidity(self, dashboard_app) -> None:
        lp = LiquidityProfile(
            market="Test",
            market_id="liq-1",
            exchange=ExchangeName.POLYMARKET,
        )
        dashboard_app.update_state(liquidity_profiles={"liq-1": lp})
        # No exception means success

    def test_partial_update(self, dashboard_app) -> None:
        dashboard_app.update_state(divergences=[])
        # No exception means success

    def test_update_sets_timestamp(self, client: TestClient, dashboard_app) -> None:
        dashboard_app.update_state(divergences=[])
        resp = client.get("/dashboard/summary")
        data = resp.json()
        assert data["last_updated"] is not None
