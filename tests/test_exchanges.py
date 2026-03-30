"""Tests for exchange connectors using mocked HTTP responses."""

from __future__ import annotations

import httpx
import pytest
import respx

from arbiter.exchanges.kalshi import KalshiExchange
from arbiter.exchanges.manifold import ManifoldExchange
from arbiter.exchanges.metaculus import MetaculusExchange
from arbiter.exchanges.polymarket import PolymarketExchange
from arbiter.models import ContractType, ExchangeName, MarketStatus

# ── Polymarket ────────────────────────────────────────────────────────


class TestPolymarketExchange:
    """Tests for PolymarketExchange connector."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_markets(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "cond-1",
                        "question": "Will BTC hit $200K?",
                        "description": "Test",
                        "category": "Crypto",
                        "outcomes": "Yes,No",
                        "outcomePrices": "0.35,0.65",
                        "volume": 100000,
                        "active": True,
                        "closed": False,
                        "resolved": False,
                        "slug": "btc-200k",
                    }
                ],
            )
        )

        exchange = PolymarketExchange()
        markets = await exchange.fetch_markets(limit=10)
        await exchange.close()

        assert len(markets) == 1
        m = markets[0]
        assert m.exchange == ExchangeName.POLYMARKET
        assert m.title == "Will BTC hit $200K?"
        assert m.yes_price == pytest.approx(0.35)
        assert m.status == MarketStatus.ACTIVE

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_single_market(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets/cond-1").mock(
            return_value=httpx.Response(
                200,
                json={
                    "conditionId": "cond-1",
                    "question": "Test Market",
                    "outcomes": "Yes,No",
                    "outcomePrices": "0.50,0.50",
                    "volume": 50000,
                    "slug": "test",
                },
            )
        )

        exchange = PolymarketExchange()
        market = await exchange.fetch_market("cond-1")
        await exchange.close()

        assert market.id == "cond-1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_order_book(self) -> None:
        respx.get("https://clob.polymarket.com/book").mock(
            return_value=httpx.Response(
                200,
                json={
                    "bids": [
                        {"price": "0.72", "size": "5000"},
                        {"price": "0.70", "size": "8000"},
                    ],
                    "asks": [
                        {"price": "0.74", "size": "4000"},
                        {"price": "0.76", "size": "6000"},
                    ],
                },
            )
        )

        exchange = PolymarketExchange()
        book = await exchange.fetch_order_book("token-1")
        await exchange.close()

        assert book.best_bid == 0.72
        assert book.best_ask == 0.74
        assert len(book.bids) == 2
        assert len(book.asks) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_resolved_market(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "cond-r",
                        "question": "Resolved market",
                        "outcomes": "Yes,No",
                        "outcomePrices": "1.0,0.0",
                        "volume": 200000,
                        "closed": True,
                        "resolved": True,
                        "resolution": "Yes",
                        "slug": "resolved",
                    }
                ],
            )
        )

        exchange = PolymarketExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()

        assert markets[0].status == MarketStatus.RESOLVED


# ── Kalshi ────────────────────────────────────────────────────────────


class TestKalshiExchange:
    """Tests for KalshiExchange connector."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_markets(self) -> None:
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
            return_value=httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXBTC200K",
                            "title": "Bitcoin $200K by 2026?",
                            "status": "open",
                            "yes_ask": 35,
                            "volume": 75000,
                            "category": "Crypto",
                        }
                    ]
                },
            )
        )

        exchange = KalshiExchange()
        markets = await exchange.fetch_markets(limit=10)
        await exchange.close()

        assert len(markets) == 1
        m = markets[0]
        assert m.exchange == ExchangeName.KALSHI
        assert m.id == "KXBTC200K"
        assert m.yes_price == pytest.approx(0.35)

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_order_book(self) -> None:
        respx.get(
            "https://api.elections.kalshi.com/trade-api/v2/markets/KXTEST/orderbook"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "orderbook": {
                        "yes": [[72, 5000], [70, 8000]],
                        "no": [[28, 4000], [26, 6000]],
                    }
                },
            )
        )

        exchange = KalshiExchange()
        book = await exchange.fetch_order_book("KXTEST")
        await exchange.close()

        assert book.best_bid == pytest.approx(0.72)
        assert len(book.bids) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_header(self) -> None:
        route = respx.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
            return_value=httpx.Response(200, json={"markets": []})
        )

        exchange = KalshiExchange(api_key="test-key-123")
        await exchange.fetch_markets()
        await exchange.close()

        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer test-key-123"


# ── Manifold ──────────────────────────────────────────────────────────


class TestManifoldExchange:
    """Tests for ManifoldExchange connector."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_markets(self) -> None:
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "manifold-1",
                        "question": "Will it rain tomorrow?",
                        "probability": 0.65,
                        "outcomeType": "BINARY",
                        "volume": 25000,
                        "isResolved": False,
                        "createdTime": 1711000000000,
                    }
                ],
            )
        )

        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(limit=10)
        await exchange.close()

        assert len(markets) == 1
        m = markets[0]
        assert m.exchange == ExchangeName.MANIFOLD
        assert m.yes_price == pytest.approx(0.65)
        assert m.contract_type == ContractType.BINARY

    @respx.mock
    @pytest.mark.asyncio
    async def test_multi_choice_market(self) -> None:
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "mc-1",
                        "question": "Who wins?",
                        "outcomeType": "MULTIPLE_CHOICE",
                        "volume": 50000,
                        "isResolved": False,
                        "answers": [
                            {"text": "Alice", "probability": 0.40},
                            {"text": "Bob", "probability": 0.35},
                            {"text": "Carol", "probability": 0.25},
                        ],
                    }
                ],
            )
        )

        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()

        assert markets[0].contract_type == ContractType.MULTI_OUTCOME
        assert len(markets[0].outcomes) == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_order_book(self) -> None:
        exchange = ManifoldExchange()
        book = await exchange.fetch_order_book("any-id")
        await exchange.close()
        assert len(book.bids) == 0
        assert len(book.asks) == 0


# ── Metaculus ─────────────────────────────────────────────────────────


class TestMetaculusExchange:
    """Tests for MetaculusExchange connector."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_markets(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 12345,
                            "title": "Will AGI arrive by 2030?",
                            "community_prediction": {"full": {"q2": 0.25}},
                            "number_of_predictions": 500,
                            "description": "AI question",
                        }
                    ]
                },
            )
        )

        exchange = MetaculusExchange()
        markets = await exchange.fetch_markets(limit=10)
        await exchange.close()

        assert len(markets) == 1
        m = markets[0]
        assert m.exchange == ExchangeName.METACULUS
        assert m.yes_price == pytest.approx(0.25)
        assert m.id == "12345"

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_order_book(self) -> None:
        exchange = MetaculusExchange()
        book = await exchange.fetch_order_book("12345")
        await exchange.close()
        assert len(book.bids) == 0
        assert len(book.asks) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_resolved_question(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 99999,
                            "title": "Resolved question",
                            "community_prediction": 0.80,
                            "resolution": 1,
                            "number_of_predictions": 200,
                        }
                    ]
                },
            )
        )

        exchange = MetaculusExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()

        assert markets[0].status == MarketStatus.RESOLVED
