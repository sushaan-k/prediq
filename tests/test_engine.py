"""Tests for arbiter.engine -- the main Arbiter orchestrator."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from arbiter.engine import Arbiter
from arbiter.exceptions import ConfigError
from arbiter.exchanges.base import BaseExchange
from arbiter.models import (
    ContractType,
    ExchangeName,
    Market,
    MarketStatus,
    OrderBook,
    Outcome,
)


def _make_mock_exchange(
    name: ExchangeName,
    markets: list[Market],
) -> BaseExchange:
    """Create a mock exchange connector.

    Args:
        name: Exchange name.
        markets: Markets to return from fetch_markets.

    Returns:
        A mock BaseExchange instance.
    """
    mock = AsyncMock(spec=BaseExchange)
    mock.name = name
    mock.fetch_markets = AsyncMock(return_value=markets)
    mock.fetch_market = AsyncMock(
        side_effect=lambda mid: next((m for m in markets if m.id == mid), markets[0])
    )
    mock.fetch_order_book = AsyncMock(return_value=OrderBook())
    mock.close = AsyncMock()
    return mock


@pytest.fixture()
def polymarket_markets(now: datetime) -> list[Market]:
    return [
        Market(
            id="pm-btc",
            exchange=ExchangeName.POLYMARKET,
            title="Will Bitcoin hit $200K by Dec 2026?",
            category="Crypto",
            contract_type=ContractType.BINARY,
            status=MarketStatus.ACTIVE,
            outcomes=[
                Outcome(name="Yes", price=0.35, volume=100_000.0),
                Outcome(name="No", price=0.65, volume=80_000.0),
            ],
            volume_total=500_000.0,
            fetched_at=now,
        ),
        Market(
            id="pm-rain",
            exchange=ExchangeName.POLYMARKET,
            title="Will it rain in NYC tomorrow?",
            category="Weather",
            contract_type=ContractType.BINARY,
            status=MarketStatus.ACTIVE,
            outcomes=[
                Outcome(name="Yes", price=0.70, volume=20_000.0),
                Outcome(name="No", price=0.30, volume=15_000.0),
            ],
            volume_total=50_000.0,
            fetched_at=now,
        ),
    ]


@pytest.fixture()
def kalshi_markets(now: datetime) -> list[Market]:
    return [
        Market(
            id="kx-btc",
            exchange=ExchangeName.KALSHI,
            title="Bitcoin to reach $200,000 by end of 2026?",
            category="Crypto",
            contract_type=ContractType.BINARY,
            status=MarketStatus.ACTIVE,
            outcomes=[
                Outcome(name="Yes", price=0.30, volume=50_000.0),
                Outcome(name="No", price=0.70, volume=45_000.0),
            ],
            volume_total=200_000.0,
            fetched_at=now,
        ),
    ]


class TestArbiter:
    """Tests for the Arbiter orchestrator."""

    @pytest.mark.asyncio
    async def test_fetch_all_markets(
        self,
        polymarket_markets: list[Market],
        kalshi_markets: list[Market],
    ) -> None:
        pm_mock = _make_mock_exchange(ExchangeName.POLYMARKET, polymarket_markets)
        kx_mock = _make_mock_exchange(ExchangeName.KALSHI, kalshi_markets)

        arb = Arbiter(exchanges=[pm_mock, kx_mock])
        result = await arb.fetch_all_markets()

        assert "polymarket" in result
        assert "kalshi" in result
        assert len(result["polymarket"]) == 2
        assert len(result["kalshi"]) == 1

        await arb.close()

    @pytest.mark.asyncio
    async def test_fetch_no_exchanges_raises(self) -> None:
        arb = Arbiter(exchanges=[])
        with pytest.raises(ConfigError):
            await arb.fetch_all_markets()

    @pytest.mark.asyncio
    async def test_add_exchange(self, kalshi_markets: list[Market]) -> None:
        arb = Arbiter()
        mock = _make_mock_exchange(ExchangeName.KALSHI, kalshi_markets)
        arb.add_exchange(mock)
        assert len(arb._exchanges) == 1
        await arb.close()

    @pytest.mark.asyncio
    async def test_match_markets(
        self,
        polymarket_markets: list[Market],
        kalshi_markets: list[Market],
    ) -> None:
        pm_mock = _make_mock_exchange(ExchangeName.POLYMARKET, polymarket_markets)
        kx_mock = _make_mock_exchange(ExchangeName.KALSHI, kalshi_markets)

        arb = Arbiter(
            exchanges=[pm_mock, kx_mock],
            similarity_threshold=0.3,
        )
        await arb.fetch_all_markets()
        pairs = arb.match_markets()

        # The BTC markets should match
        assert len(pairs) >= 1
        await arb.close()

    @pytest.mark.asyncio
    async def test_divergences(
        self,
        polymarket_markets: list[Market],
        kalshi_markets: list[Market],
    ) -> None:
        pm_mock = _make_mock_exchange(ExchangeName.POLYMARKET, polymarket_markets)
        kx_mock = _make_mock_exchange(ExchangeName.KALSHI, kalshi_markets)

        arb = Arbiter(
            exchanges=[pm_mock, kx_mock],
            similarity_threshold=0.3,
        )
        divs = await arb.divergences(min_spread=0.01)

        # Should detect BTC price divergence (0.35 vs 0.30 = 0.05 spread)
        btc_divs = [
            d for d in divs if "bitcoin" in d.event.lower() or "btc" in d.event.lower()
        ]
        assert len(btc_divs) >= 1
        assert btc_divs[0].spread == pytest.approx(0.05)
        await arb.close()

    @pytest.mark.asyncio
    async def test_violations(
        self,
        polymarket_markets: list[Market],
    ) -> None:
        pm_mock = _make_mock_exchange(ExchangeName.POLYMARKET, polymarket_markets)

        arb = Arbiter(exchanges=[pm_mock])
        _binary_v, _multi_v = await arb.violations()

        # YES + NO = 0.35 + 0.65 = 1.0 (exact), no violation expected
        # with default tolerance
        await arb.close()

    @pytest.mark.asyncio
    async def test_liquidity(
        self,
        polymarket_markets: list[Market],
    ) -> None:
        pm_mock = _make_mock_exchange(ExchangeName.POLYMARKET, polymarket_markets)

        arb = Arbiter(exchanges=[pm_mock])
        profile = await arb.liquidity("polymarket", "pm-btc")
        assert profile.market_id == "pm-btc"
        await arb.close()

    @pytest.mark.asyncio
    async def test_get_exchange_not_found(self) -> None:
        arb = Arbiter(exchanges=[])
        with pytest.raises(ConfigError):
            arb._get_exchange("nonexistent")

    @pytest.mark.asyncio
    async def test_context_manager(
        self,
        polymarket_markets: list[Market],
    ) -> None:
        pm_mock = _make_mock_exchange(ExchangeName.POLYMARKET, polymarket_markets)

        async with Arbiter(exchanges=[pm_mock]) as arb:
            result = await arb.fetch_all_markets()
            assert len(result) == 1

        pm_mock.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_exchange_fetch_failure_handled(self) -> None:
        failing_mock = AsyncMock(spec=BaseExchange)
        failing_mock.name = ExchangeName.POLYMARKET
        failing_mock.fetch_markets = AsyncMock(side_effect=RuntimeError("API down"))
        failing_mock.close = AsyncMock()

        arb = Arbiter(exchanges=[failing_mock])
        result = await arb.fetch_all_markets()
        assert result["polymarket"] == []
        await arb.close()
