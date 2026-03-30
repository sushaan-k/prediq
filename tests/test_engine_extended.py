"""Extended tests for arbiter.engine -- covering quality, export_dataset,
monitor, and error-handling edge cases.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
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
    OrderBookLevel,
    Outcome,
)


def _make_mock_exchange(
    name: ExchangeName,
    markets: list[Market],
) -> BaseExchange:
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
def now() -> datetime:
    return datetime(2026, 3, 15, 14, 0, 0, tzinfo=UTC)


@pytest.fixture()
def resolved_markets(now: datetime) -> list[Market]:
    """Enough resolved markets for quality scoring."""
    markets = []
    for i in range(15):
        markets.append(
            Market(
                id=f"res-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Resolved event {i}",
                category="Politics",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.7, volume=10_000.0),
                    Outcome(name="No", price=0.3, volume=8_000.0),
                ],
                resolution="yes",
                volume_total=50_000.0,
                created_at=now,
                closes_at=now,
                resolved_at=now,
                fetched_at=now,
            )
        )
    return markets


class TestArbiterQuality:
    @pytest.mark.asyncio
    async def test_quality_all_exchanges(self, resolved_markets: list[Market]) -> None:
        mock = _make_mock_exchange(ExchangeName.POLYMARKET, resolved_markets)
        arb = Arbiter(exchanges=[mock])
        quality = await arb.quality()
        assert quality.brier_score >= 0.0
        assert quality.sample_size > 0
        await arb.close()

    @pytest.mark.asyncio
    async def test_quality_specific_exchange(
        self, resolved_markets: list[Market]
    ) -> None:
        mock = _make_mock_exchange(ExchangeName.POLYMARKET, resolved_markets)
        arb = Arbiter(exchanges=[mock])
        quality = await arb.quality(exchange="polymarket")
        assert quality.exchange == ExchangeName.POLYMARKET
        await arb.close()

    @pytest.mark.asyncio
    async def test_quality_uses_cache(self, resolved_markets: list[Market]) -> None:
        mock = _make_mock_exchange(ExchangeName.POLYMARKET, resolved_markets)
        arb = Arbiter(exchanges=[mock])
        # Pre-fill cache
        arb._market_cache = {"polymarket": resolved_markets}
        quality = await arb.quality()
        # fetch_all_markets should NOT have been called because cache is set
        mock.fetch_markets.assert_not_called()
        assert quality.sample_size > 0
        await arb.close()

    @pytest.mark.asyncio
    async def test_quality_invalid_exchange_name(
        self, resolved_markets: list[Market]
    ) -> None:
        mock = _make_mock_exchange(ExchangeName.POLYMARKET, resolved_markets)
        arb = Arbiter(exchanges=[mock])
        with pytest.raises(ConfigError):
            await arb.quality(exchange="nonexistent")
        await arb.close()


class TestArbiterExportDataset:
    @pytest.mark.asyncio
    async def test_export_dataset(self, now: datetime, tmp_path: Path) -> None:
        markets = [
            Market(
                id="ex-1",
                exchange=ExchangeName.MANIFOLD,
                title="Export test",
                outcomes=[
                    Outcome(name="Yes", price=0.6, volume=0.0),
                    Outcome(name="No", price=0.4, volume=0.0),
                ],
                volume_total=10_000.0,
                fetched_at=now,
            )
        ]
        mock = _make_mock_exchange(ExchangeName.MANIFOLD, markets)
        arb = Arbiter(exchanges=[mock])

        out = str(tmp_path / "export.parquet")
        result = await arb.export_dataset(out)
        assert result.endswith(".parquet")
        assert Path(result).exists()
        await arb.close()

    @pytest.mark.asyncio
    async def test_export_dataset_filter_exchanges(
        self, now: datetime, tmp_path: Path
    ) -> None:
        pm_markets = [
            Market(
                id="pm-exp",
                exchange=ExchangeName.POLYMARKET,
                title="PM export",
                outcomes=[
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ],
                fetched_at=now,
            )
        ]
        kx_markets = [
            Market(
                id="kx-exp",
                exchange=ExchangeName.KALSHI,
                title="KX export",
                outcomes=[
                    Outcome(name="Yes", price=0.6, volume=0.0),
                    Outcome(name="No", price=0.4, volume=0.0),
                ],
                fetched_at=now,
            )
        ]
        pm_mock = _make_mock_exchange(ExchangeName.POLYMARKET, pm_markets)
        kx_mock = _make_mock_exchange(ExchangeName.KALSHI, kx_markets)
        arb = Arbiter(exchanges=[pm_mock, kx_mock])

        out = str(tmp_path / "filtered.parquet")
        result = await arb.export_dataset(out, exchanges=["polymarket"])
        assert Path(result).exists()
        await arb.close()

    @pytest.mark.asyncio
    async def test_export_dataset_uses_cache(
        self, now: datetime, tmp_path: Path
    ) -> None:
        markets = [
            Market(
                id="cached",
                exchange=ExchangeName.MANIFOLD,
                title="Cached",
                outcomes=[
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ],
                fetched_at=now,
            )
        ]
        mock = _make_mock_exchange(ExchangeName.MANIFOLD, markets)
        arb = Arbiter(exchanges=[mock])
        arb._market_cache = {"manifold": markets}

        out = str(tmp_path / "cached.parquet")
        await arb.export_dataset(out)
        mock.fetch_markets.assert_not_called()
        await arb.close()


class TestArbiterViolationsCache:
    @pytest.mark.asyncio
    async def test_violations_uses_cache(self, now: datetime) -> None:
        markets = [
            Market(
                id="v-cache",
                exchange=ExchangeName.POLYMARKET,
                title="Violations cache",
                outcomes=[
                    Outcome(name="Yes", price=0.55, volume=0.0),
                    Outcome(name="No", price=0.55, volume=0.0),
                ],
                fetched_at=now,
            )
        ]
        mock = _make_mock_exchange(ExchangeName.POLYMARKET, markets)
        arb = Arbiter(exchanges=[mock])
        arb._market_cache = {"polymarket": markets}

        _binary_v, _ = await arb.violations()
        mock.fetch_markets.assert_not_called()
        await arb.close()


class TestArbiterLiquidityWithOrderBook:
    @pytest.mark.asyncio
    async def test_liquidity_with_order_book(self, now: datetime) -> None:
        market = Market(
            id="liq-ob",
            exchange=ExchangeName.POLYMARKET,
            title="Liquidity OB",
            outcomes=[
                Outcome(name="Yes", price=0.70, volume=100_000.0),
                Outcome(name="No", price=0.30, volume=50_000.0),
            ],
            fetched_at=now,
        )
        order_book = OrderBook(
            bids=[OrderBookLevel(price=0.70, quantity=5_000.0)],
            asks=[OrderBookLevel(price=0.74, quantity=4_000.0)],
        )
        market = market.model_copy(
            update={"metadata": {"clobTokenIds": ["token-123", "token-456"]}}
        )
        mock = _make_mock_exchange(ExchangeName.POLYMARKET, [market])
        mock.fetch_order_book = AsyncMock(return_value=order_book)

        arb = Arbiter(exchanges=[mock])
        profile = await arb.liquidity("polymarket", "liq-ob")
        mock.fetch_order_book.assert_awaited_once_with("token-123")
        assert profile.best_bid == 0.70
        assert profile.best_ask == 0.74
        await arb.close()

    @pytest.mark.asyncio
    async def test_liquidity_order_book_fetch_failure(self, now: datetime) -> None:
        market = Market(
            id="liq-fail",
            exchange=ExchangeName.POLYMARKET,
            title="Liquidity fail",
            outcomes=[
                Outcome(name="Yes", price=0.70, volume=100_000.0),
                Outcome(name="No", price=0.30, volume=50_000.0),
            ],
            fetched_at=now,
        )
        mock = _make_mock_exchange(ExchangeName.POLYMARKET, [market])
        mock.fetch_order_book = AsyncMock(side_effect=RuntimeError("OB fail"))

        arb = Arbiter(exchanges=[mock])
        # Should not raise; gracefully falls back
        profile = await arb.liquidity("polymarket", "liq-fail")
        assert profile.market_id == "liq-fail"
        await arb.close()


class TestArbiterGetExchange:
    def test_get_exchange_case_insensitive(self, now: datetime) -> None:
        market = Market(
            id="m",
            exchange=ExchangeName.POLYMARKET,
            title="Test",
            outcomes=[
                Outcome(name="Yes", price=0.5, volume=0.0),
                Outcome(name="No", price=0.5, volume=0.0),
            ],
            fetched_at=now,
        )
        mock = _make_mock_exchange(ExchangeName.POLYMARKET, [market])
        arb = Arbiter(exchanges=[mock])
        # Should work with any casing
        ex = arb._get_exchange("Polymarket")
        assert ex is mock

    def test_get_exchange_not_found_message(self) -> None:
        arb = Arbiter(exchanges=[])
        with pytest.raises(ConfigError, match="not configured"):
            arb._get_exchange("nonexistent")


class TestArbiterMultiExchangeFetch:
    @pytest.mark.asyncio
    async def test_partial_failure(self, now: datetime) -> None:
        """When one exchange fails, the other should still return results."""
        good_markets = [
            Market(
                id="good",
                exchange=ExchangeName.KALSHI,
                title="Good",
                outcomes=[
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ],
                fetched_at=now,
            )
        ]
        good_mock = _make_mock_exchange(ExchangeName.KALSHI, good_markets)

        bad_mock = AsyncMock(spec=BaseExchange)
        bad_mock.name = ExchangeName.POLYMARKET
        bad_mock.fetch_markets = AsyncMock(side_effect=RuntimeError("API exploded"))
        bad_mock.close = AsyncMock()

        arb = Arbiter(exchanges=[bad_mock, good_mock])
        result = await arb.fetch_all_markets()

        assert result["polymarket"] == []
        assert len(result["kalshi"]) == 1
        await arb.close()
