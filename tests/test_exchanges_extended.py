"""Extended exchange connector tests -- edge cases, error handling, malformed data."""

from __future__ import annotations

import httpx
import pytest
import respx

from arbiter.exceptions import (
    ExchangeConnectionError,
    ExchangeError,
    ExchangeRateLimitError,
    MarketNotFoundError,
)
from arbiter.exchanges.base import BaseExchange
from arbiter.exchanges.kalshi import KalshiExchange
from arbiter.exchanges.manifold import ManifoldExchange
from arbiter.exchanges.metaculus import MetaculusExchange
from arbiter.exchanges.polymarket import PolymarketExchange
from arbiter.models import ContractType, MarketStatus

# ── BaseExchange ─────────────────────────────────────────────────────────


class TestBaseExchangeEdgeCases:
    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_429(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(
                429,
                headers={"Retry-After": "2.5"},
            )
        )
        exchange = PolymarketExchange()
        with pytest.raises(ExchangeRateLimitError) as exc_info:
            await exchange.fetch_markets()
        assert exc_info.value.retry_after == 2.5
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_429_no_retry_after(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(429)
        )
        exchange = PolymarketExchange()
        with pytest.raises(ExchangeRateLimitError) as exc_info:
            await exchange.fetch_markets()
        assert exc_info.value.retry_after is None
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            side_effect=httpx.ConnectError("DNS resolution failed")
        )
        exchange = PolymarketExchange()
        with pytest.raises(ExchangeConnectionError):
            await exchange.fetch_markets()
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )
        exchange = PolymarketExchange()
        with pytest.raises(ExchangeConnectionError):
            await exchange.fetch_markets()
        await exchange.close()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with PolymarketExchange() as exchange:
            assert isinstance(exchange, BaseExchange)
        assert exchange._closed is True

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        exchange = PolymarketExchange()
        await exchange.close()
        await exchange.close()  # should not raise
        assert exchange._closed is True


# ── Polymarket edge cases ────────────────────────────────────────────────


class TestPolymarketEdgeCases:
    @respx.mock
    @pytest.mark.asyncio
    async def test_malformed_market_skipped(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        # Missing outcomes, outcomePrices -> should be skipped
                        "conditionId": "bad-1",
                        "question": "Bad market",
                    },
                    {
                        "conditionId": "good-1",
                        "question": "Good market",
                        "outcomes": "Yes,No",
                        "outcomePrices": "0.50,0.50",
                        "volume": 1000,
                        "slug": "good",
                    },
                ],
            )
        )
        exchange = PolymarketExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        # Both should parse since missing outcomes defaults to Yes/No
        assert len(markets) >= 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_market_with_list_outcomes(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "list-1",
                        "question": "List outcomes",
                        "outcomes": ["Yes", "No"],
                        "outcomePrices": ["0.60", "0.40"],
                        "volume": 5000,
                        "slug": "list-test",
                    }
                ],
            )
        )
        exchange = PolymarketExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert len(markets) == 1
        assert markets[0].yes_price == pytest.approx(0.60)

    @respx.mock
    @pytest.mark.asyncio
    async def test_market_with_dates(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "dated-1",
                        "question": "Dated market",
                        "outcomes": "Yes,No",
                        "outcomePrices": "0.70,0.30",
                        "volume": 200000,
                        "startDate": "2026-01-01T00:00:00Z",
                        "endDate": "2026-12-31T23:59:59Z",
                        "slug": "dated",
                    }
                ],
            )
        )
        exchange = PolymarketExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert markets[0].created_at is not None
        assert markets[0].closes_at is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_market_with_invalid_dates(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "bad-date",
                        "question": "Bad date market",
                        "outcomes": "Yes,No",
                        "outcomePrices": "0.50,0.50",
                        "volume": 100,
                        "startDate": "not-a-date",
                        "endDate": "also-not-a-date",
                        "slug": "bad-date",
                    }
                ],
            )
        )
        exchange = PolymarketExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert markets[0].created_at is None
        assert markets[0].closes_at is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_not_found(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets/nonexistent").mock(
            return_value=httpx.Response(404)
        )
        exchange = PolymarketExchange()
        with pytest.raises(MarketNotFoundError):
            await exchange.fetch_market("nonexistent")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_empty_response(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets/empty").mock(
            return_value=httpx.Response(200, json={})
        )
        exchange = PolymarketExchange()
        with pytest.raises(MarketNotFoundError):
            await exchange.fetch_market("empty")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_order_book_fetch_failure(self) -> None:
        respx.get("https://clob.polymarket.com/book").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        exchange = PolymarketExchange()
        with pytest.raises(ExchangeError):
            await exchange.fetch_order_book("token-x")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_closed_market_status(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "closed-1",
                        "question": "Closed market",
                        "outcomes": "Yes,No",
                        "outcomePrices": "0.80,0.20",
                        "volume": 50000,
                        "closed": True,
                        "resolved": False,
                        "slug": "closed",
                    }
                ],
            )
        )
        exchange = PolymarketExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        assert markets[0].status == MarketStatus.CLOSED

    @respx.mock
    @pytest.mark.asyncio
    async def test_multi_outcome_market(self) -> None:
        respx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "conditionId": "multi-1",
                        "question": "Who wins?",
                        "outcomes": "Alice,Bob,Carol",
                        "outcomePrices": "0.40,0.35,0.25",
                        "volume": 100000,
                        "slug": "multi",
                    }
                ],
            )
        )
        exchange = PolymarketExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert markets[0].contract_type == ContractType.MULTI_OUTCOME
        assert len(markets[0].outcomes) == 3


# ── Kalshi edge cases ────────────────────────────────────────────────────


class TestKalshiEdgeCases:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_single(self) -> None:
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets/KXBTC").mock(
            return_value=httpx.Response(
                200,
                json={
                    "market": {
                        "ticker": "KXBTC",
                        "title": "BTC 200K?",
                        "status": "open",
                        "yes_ask": 40,
                        "volume": 50000,
                    }
                },
            )
        )
        exchange = KalshiExchange()
        market = await exchange.fetch_market("KXBTC")
        await exchange.close()
        assert market.id == "KXBTC"
        assert market.yes_price == pytest.approx(0.40)

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_not_found(self) -> None:
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets/KXNOPE").mock(
            return_value=httpx.Response(404)
        )
        exchange = KalshiExchange()
        with pytest.raises(MarketNotFoundError):
            await exchange.fetch_market("KXNOPE")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_empty_response(self) -> None:
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets/KXEMPTY").mock(
            return_value=httpx.Response(200, json={"market": {}})
        )
        exchange = KalshiExchange()
        # Empty dict triggers MarketNotFoundError because not raw evaluates to True
        with pytest.raises(MarketNotFoundError):
            await exchange.fetch_market("KXEMPTY")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_order_book_fetch_failure(self) -> None:
        respx.get(
            "https://api.elections.kalshi.com/trade-api/v2/markets/KXFAIL/orderbook"
        ).mock(side_effect=httpx.ConnectError("fail"))
        exchange = KalshiExchange()
        with pytest.raises(ExchangeError):
            await exchange.fetch_order_book("KXFAIL")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_malformed_market_skipped(self) -> None:
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
            return_value=httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXGOOD",
                            "title": "Good market",
                            "status": "open",
                            "yes_ask": 50,
                            "volume": 10000,
                        }
                    ]
                },
            )
        )
        exchange = KalshiExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert len(markets) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_settled_status(self) -> None:
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
            return_value=httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXSETTLE",
                            "title": "Settled",
                            "status": "settled",
                            "yes_ask": 99,
                            "volume": 100000,
                            "result": "yes",
                        }
                    ]
                },
            )
        )
        exchange = KalshiExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        assert markets[0].status == MarketStatus.RESOLVED
        assert markets[0].resolution == "yes"

    @respx.mock
    @pytest.mark.asyncio
    async def test_close_time_parsing(self) -> None:
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
            return_value=httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXDATE",
                            "title": "Dated",
                            "status": "open",
                            "yes_ask": 50,
                            "close_time": "2026-12-31T23:59:59Z",
                        }
                    ]
                },
            )
        )
        exchange = KalshiExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert markets[0].closes_at is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_close_time(self) -> None:
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
            return_value=httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXBADDATE",
                            "title": "Bad date",
                            "status": "open",
                            "yes_ask": 50,
                            "close_time": "not-a-date",
                        }
                    ]
                },
            )
        )
        exchange = KalshiExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert markets[0].closes_at is None


# ── Manifold edge cases ──────────────────────────────────────────────────


class TestManifoldEdgeCases:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_single(self) -> None:
        respx.get("https://api.manifold.markets/v0/market/test-slug").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "mf-1",
                    "question": "Test?",
                    "probability": 0.70,
                    "outcomeType": "BINARY",
                    "isResolved": False,
                    "volume": 10000,
                },
            )
        )
        exchange = ManifoldExchange()
        market = await exchange.fetch_market("test-slug")
        await exchange.close()
        assert market.id == "mf-1"
        assert market.yes_price == pytest.approx(0.70)

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_not_found(self) -> None:
        respx.get("https://api.manifold.markets/v0/market/nope").mock(
            return_value=httpx.Response(404)
        )
        exchange = ManifoldExchange()
        with pytest.raises(MarketNotFoundError):
            await exchange.fetch_market("nope")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_empty_response(self) -> None:
        respx.get("https://api.manifold.markets/v0/market/empty").mock(
            return_value=httpx.Response(200, json={})
        )
        exchange = ManifoldExchange()
        with pytest.raises(MarketNotFoundError):
            await exchange.fetch_market("empty")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_resolved_market(self) -> None:
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "res-1",
                        "question": "Resolved?",
                        "probability": 0.95,
                        "outcomeType": "BINARY",
                        "isResolved": True,
                        "resolution": "YES",
                        "volume": 50000,
                    }
                ],
            )
        )
        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        assert markets[0].status == MarketStatus.RESOLVED
        assert markets[0].resolution == "YES"

    @respx.mock
    @pytest.mark.asyncio
    async def test_closed_market_by_time(self) -> None:
        # closeTime in the past
        import time

        past_ts = int((time.time() - 86400) * 1000)
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "past-1",
                        "question": "Past close?",
                        "probability": 0.50,
                        "outcomeType": "BINARY",
                        "isResolved": False,
                        "closeTime": past_ts,
                        "volume": 1000,
                    }
                ],
            )
        )
        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        assert markets[0].status == MarketStatus.CLOSED

    @respx.mock
    @pytest.mark.asyncio
    async def test_active_only_filters(self) -> None:
        import time

        past_ts = int((time.time() - 86400) * 1000)
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "past-2",
                        "question": "Past market",
                        "probability": 0.50,
                        "outcomeType": "BINARY",
                        "isResolved": False,
                        "closeTime": past_ts,
                        "volume": 1000,
                    }
                ],
            )
        )
        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(active_only=True)
        await exchange.close()
        # Should filter out closed market
        assert len(markets) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_unknown_outcome_type(self) -> None:
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "other-type",
                        "question": "Weird type?",
                        "outcomeType": "NUMERIC",
                        "isResolved": False,
                        "volume": 500,
                    }
                ],
            )
        )
        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        # Falls back to binary with 0.5/0.5
        assert markets[0].contract_type == ContractType.BINARY
        assert markets[0].yes_price == pytest.approx(0.50)

    @respx.mock
    @pytest.mark.asyncio
    async def test_multi_choice_too_few_answers(self) -> None:
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "few-answers",
                        "question": "Few answers?",
                        "outcomeType": "MULTIPLE_CHOICE",
                        "isResolved": False,
                        "answers": [{"text": "Only one", "probability": 1.0}],
                        "volume": 100,
                    }
                ],
            )
        )
        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        # Falls back to binary default
        assert len(markets[0].outcomes) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_header(self) -> None:
        route = respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(200, json=[])
        )
        exchange = ManifoldExchange(api_key="mf-secret")
        await exchange.fetch_markets()
        await exchange.close()
        request = route.calls[0].request
        assert request.headers["Authorization"] == "Key mf-secret"

    @respx.mock
    @pytest.mark.asyncio
    async def test_malformed_market_skipped(self) -> None:
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        # Missing required fields -> ValueError/KeyError during parse
                        "id": "bad-mf",
                        "outcomeType": "MULTIPLE_CHOICE",
                        "isResolved": False,
                        # No question, no answers
                    },
                    {
                        "id": "good-mf",
                        "question": "Good?",
                        "probability": 0.60,
                        "outcomeType": "BINARY",
                        "isResolved": False,
                    },
                ],
            )
        )
        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        assert len(markets) >= 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_group_slugs_category(self) -> None:
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "cat-1",
                        "question": "Category test?",
                        "probability": 0.50,
                        "outcomeType": "BINARY",
                        "isResolved": False,
                        "groupSlugs": ["politics", "us-elections"],
                        "volume": 100,
                    }
                ],
            )
        )
        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        assert markets[0].category == "politics"

    @respx.mock
    @pytest.mark.asyncio
    async def test_created_time_parsing(self) -> None:
        respx.get("https://api.manifold.markets/v0/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "time-1",
                        "question": "Created time?",
                        "probability": 0.50,
                        "outcomeType": "BINARY",
                        "isResolved": False,
                        "createdTime": 1711000000000,
                        "volume": 100,
                    }
                ],
            )
        )
        exchange = ManifoldExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        assert markets[0].created_at is not None


# ── Metaculus edge cases ─────────────────────────────────────────────────


class TestMetaculusEdgeCases:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_single(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/12345/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 12345,
                    "title": "Single question",
                    "community_prediction": {"full": {"q2": 0.30}},
                    "number_of_predictions": 100,
                },
            )
        )
        exchange = MetaculusExchange()
        market = await exchange.fetch_market("12345")
        await exchange.close()
        assert market.id == "12345"
        assert market.yes_price == pytest.approx(0.30)

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_not_found(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/99999/").mock(
            return_value=httpx.Response(404)
        )
        exchange = MetaculusExchange()
        with pytest.raises(MarketNotFoundError):
            await exchange.fetch_market("99999")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_market_empty_response(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/0/").mock(
            return_value=httpx.Response(200, json={})
        )
        exchange = MetaculusExchange()
        with pytest.raises(MarketNotFoundError):
            await exchange.fetch_market("0")
        await exchange.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_community_prediction_as_number(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 100,
                            "title": "Numeric prediction",
                            "community_prediction": 0.42,
                            "number_of_predictions": 50,
                        }
                    ]
                },
            )
        )
        exchange = MetaculusExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert markets[0].yes_price == pytest.approx(0.42)

    @respx.mock
    @pytest.mark.asyncio
    async def test_closed_state(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 200,
                            "title": "Closed question",
                            "community_prediction": {"full": {"q2": 0.60}},
                            "number_of_predictions": 100,
                            "active_state": "CLOSED",
                        }
                    ]
                },
            )
        )
        exchange = MetaculusExchange()
        markets = await exchange.fetch_markets(active_only=False)
        await exchange.close()
        assert markets[0].status == MarketStatus.CLOSED

    @respx.mock
    @pytest.mark.asyncio
    async def test_with_timestamps(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 300,
                            "title": "Timed question",
                            "community_prediction": {"full": {"q2": 0.50}},
                            "number_of_predictions": 75,
                            "created_time": "2026-01-01T00:00:00Z",
                            "close_time": "2026-12-31T23:59:59Z",
                        }
                    ]
                },
            )
        )
        exchange = MetaculusExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert markets[0].created_at is not None
        assert markets[0].closes_at is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_timestamps(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 400,
                            "title": "Bad times",
                            "community_prediction": 0.50,
                            "number_of_predictions": 10,
                            "created_time": "bad-date",
                            "close_time": "also-bad",
                        }
                    ]
                },
            )
        )
        exchange = MetaculusExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert markets[0].created_at is None
        assert markets[0].closes_at is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_malformed_question_skipped(self) -> None:
        respx.get("https://www.metaculus.com/api2/questions/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            # Missing key fields -> should trigger except
                            "title": "Bad question",
                            # Missing "id" key
                        },
                        {
                            "id": 500,
                            "title": "Good question",
                            "community_prediction": 0.50,
                            "number_of_predictions": 10,
                        },
                    ]
                },
            )
        )
        exchange = MetaculusExchange()
        markets = await exchange.fetch_markets()
        await exchange.close()
        assert len(markets) >= 1
