"""Tests for arbiter.output.api -- FastAPI REST API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import arbiter.output.api as api
from arbiter.models import (
    Divergence,
    ExchangeName,
    LiquidityProfile,
    MarketQuality,
    MultiOutcomeViolation,
    ProbabilityViolation,
)
from arbiter.output.api import _populate_state, _state, app, update_state


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Reset API state before each test."""
    api._arbiter_instance = None
    _state["divergences"] = []
    _state["binary_violations"] = []
    _state["multi_violations"] = []
    _state["liquidity"] = {}
    _state["quality"] = {}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestHealthCheck:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "arbiter"


class TestDivergenceEndpoint:
    def test_empty_divergences(self, client: TestClient) -> None:
        resp = client.get("/api/v1/divergences")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_divergences(self, client: TestClient) -> None:
        div = Divergence(
            event="Test Event",
            outcome="Yes",
            exchange_a=ExchangeName.POLYMARKET,
            exchange_b=ExchangeName.KALSHI,
            price_a=0.55,
            price_b=0.48,
            spread=0.07,
            spread_pct=0.136,
        )
        update_state(divergences=[div])

        resp = client.get("/api/v1/divergences")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["spread"] == pytest.approx(0.07)

    def test_min_spread_filter(self, client: TestClient) -> None:
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
            for i in range(5)
        ]
        update_state(divergences=divs)

        resp = client.get("/api/v1/divergences?min_spread=0.04")
        data = resp.json()
        assert all(d["spread"] >= 0.04 for d in data)

    def test_limit(self, client: TestClient) -> None:
        divs = [
            Divergence(
                event=f"Event {i}",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.55,
                price_b=0.50,
                spread=0.05,
                spread_pct=0.10,
            )
            for i in range(10)
        ]
        update_state(divergences=divs)

        resp = client.get("/api/v1/divergences?limit=3")
        assert len(resp.json()) == 3


class TestBinaryViolationsEndpoint:
    def test_empty_violations(self, client: TestClient) -> None:
        resp = client.get("/api/v1/violations/binary")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_binary_violations(self, client: TestClient) -> None:
        v = ProbabilityViolation(
            market="Test market",
            market_id="t-1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.55,
            no_price=0.52,
            price_sum=1.07,
            implied_arb=0.07,
        )
        update_state(binary_violations=[v])

        resp = client.get("/api/v1/violations/binary")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["implied_arb"] == pytest.approx(0.07)

    def test_limit(self, client: TestClient) -> None:
        violations = [
            ProbabilityViolation(
                market=f"Market {i}",
                market_id=f"m-{i}",
                exchange=ExchangeName.POLYMARKET,
                yes_price=0.55,
                no_price=0.52,
                price_sum=1.07,
                implied_arb=0.07,
            )
            for i in range(5)
        ]
        update_state(binary_violations=violations)

        resp = client.get("/api/v1/violations/binary?limit=2")
        assert len(resp.json()) == 2


class TestMultiViolationsEndpoint:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get("/api/v1/violations/multi")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_multi_violations(self, client: TestClient) -> None:
        v = MultiOutcomeViolation(
            market="Election",
            market_id="e-1",
            exchange=ExchangeName.KALSHI,
            outcomes={"A": 0.40, "B": 0.35, "C": 0.30},
            price_sum=1.05,
            deviation=0.05,
        )
        update_state(multi_violations=[v])

        resp = client.get("/api/v1/violations/multi")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["deviation"] == pytest.approx(0.05)


class TestLiquidityEndpoint:
    def test_missing_market(self, client: TestClient) -> None:
        resp = client.get("/api/v1/liquidity/nonexistent")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_get_liquidity(self, client: TestClient) -> None:
        lp = LiquidityProfile(
            market="Test",
            market_id="liq-1",
            exchange=ExchangeName.POLYMARKET,
            best_bid=0.72,
            best_ask=0.74,
            spread=0.02,
        )
        update_state(liquidity={"liq-1": lp})

        resp = client.get("/api/v1/liquidity/liq-1")
        data = resp.json()
        assert data["best_bid"] == pytest.approx(0.72)
        assert data["spread"] == pytest.approx(0.02)


class TestQualityEndpoint:
    def test_missing_exchange(self, client: TestClient) -> None:
        resp = client.get("/api/v1/quality/nonexistent")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_get_quality(self, client: TestClient) -> None:
        mq = MarketQuality(
            exchange=ExchangeName.POLYMARKET,
            brier_score=0.18,
            calibration_error=0.03,
            manipulation_score=0.10,
            volume_accuracy_correlation=0.65,
            sample_size=100,
        )
        update_state(quality={"polymarket": mq})

        resp = client.get("/api/v1/quality/polymarket")
        data = resp.json()
        assert data["brier_score"] == pytest.approx(0.18)
        assert data["sample_size"] == 100

    def test_case_insensitive(self, client: TestClient) -> None:
        mq = MarketQuality(
            exchange=ExchangeName.KALSHI,
            brier_score=0.20,
            calibration_error=0.04,
            manipulation_score=0.05,
            volume_accuracy_correlation=0.50,
            sample_size=50,
        )
        update_state(quality={"kalshi": mq})

        resp = client.get("/api/v1/quality/KALSHI")
        data = resp.json()
        assert data is not None
        assert data["brier_score"] == pytest.approx(0.20)


class TestUpdateState:
    def test_partial_update(self) -> None:
        div = Divergence(
            event="E",
            outcome="Yes",
            exchange_a=ExchangeName.POLYMARKET,
            exchange_b=ExchangeName.KALSHI,
            price_a=0.55,
            price_b=0.50,
            spread=0.05,
            spread_pct=0.10,
        )
        update_state(divergences=[div])
        assert len(_state["divergences"]) == 1
        assert _state["binary_violations"] == []

    def test_none_does_not_overwrite(self) -> None:
        div = Divergence(
            event="E",
            outcome="Yes",
            exchange_a=ExchangeName.POLYMARKET,
            exchange_b=ExchangeName.KALSHI,
            price_a=0.55,
            price_b=0.50,
            spread=0.05,
            spread_pct=0.10,
        )
        update_state(divergences=[div])
        update_state(binary_violations=[])
        assert len(_state["divergences"]) == 1


class TestPopulateState:
    @pytest.mark.asyncio
    async def test_populates_liquidity_and_quality(self) -> None:
        from arbiter.models import Market, MarketStatus, Outcome

        market = Market(
            id="liq-1",
            exchange=ExchangeName.POLYMARKET,
            title="Liquidity test",
            status=MarketStatus.ACTIVE,
            outcomes=[
                Outcome(name="Yes", price=0.55, volume=0.0),
                Outcome(name="No", price=0.45, volume=0.0),
            ],
        )
        lp = LiquidityProfile(
            market="Liquidity test",
            market_id="liq-1",
            exchange=ExchangeName.POLYMARKET,
            best_bid=0.54,
            best_ask=0.56,
            spread=0.02,
        )
        mq = MarketQuality(
            exchange=ExchangeName.POLYMARKET,
            brier_score=0.18,
            calibration_error=0.03,
            manipulation_score=0.10,
            volume_accuracy_correlation=0.65,
            sample_size=10,
        )

        mock_arb = AsyncMock()
        mock_arb.divergences = AsyncMock(return_value=[])
        mock_arb.violations = AsyncMock(return_value=([], []))
        mock_arb.liquidity = AsyncMock(return_value=lp)
        mock_arb.quality = AsyncMock(return_value=mq)
        mock_arb._market_cache = {"polymarket": [market]}

        api._arbiter_instance = mock_arb
        try:
            await _populate_state()

            assert _state["liquidity"]["liq-1"].best_bid == pytest.approx(0.54)
            assert _state["quality"]["polymarket"].sample_size == 10
        finally:
            api._arbiter_instance = None
