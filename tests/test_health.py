"""Tests for exchange health_check() and exchange_status CLI command."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from arbiter.exchanges.kalshi import KalshiExchange
from arbiter.exchanges.manifold import ManifoldExchange
from arbiter.exchanges.metaculus import MetaculusExchange
from arbiter.exchanges.polymarket import PolymarketExchange


@pytest.mark.asyncio
class TestExchangeHealthCheck:
    """Tests for BaseExchange.health_check() across all connectors."""

    async def test_polymarket_health_ok(self) -> None:
        exchange = PolymarketExchange()
        with patch.object(exchange, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = AsyncMock(status_code=200)
            result = await exchange.health_check()

        assert result["exchange"] == "polymarket"
        assert result["status"] == "ok"
        assert result["latency_ms"] >= 0
        assert "error" not in result

    async def test_kalshi_health_ok(self) -> None:
        exchange = KalshiExchange()
        with patch.object(exchange, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = AsyncMock(status_code=200)
            result = await exchange.health_check()

        assert result["exchange"] == "kalshi"
        assert result["status"] == "ok"

    async def test_manifold_health_ok(self) -> None:
        exchange = ManifoldExchange()
        with patch.object(exchange, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = AsyncMock(status_code=200)
            result = await exchange.health_check()

        assert result["exchange"] == "manifold"
        assert result["status"] == "ok"

    async def test_metaculus_health_ok(self) -> None:
        exchange = MetaculusExchange()
        with patch.object(exchange, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = AsyncMock(status_code=200)
            result = await exchange.health_check()

        assert result["exchange"] == "metaculus"
        assert result["status"] == "ok"

    async def test_health_error_on_connection_failure(self) -> None:
        exchange = PolymarketExchange()
        with patch.object(exchange, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = ConnectionError("Connection refused")
            result = await exchange.health_check()

        assert result["exchange"] == "polymarket"
        assert result["status"] == "error"
        assert "error" in result
        assert "Connection refused" in result["error"]
        assert result["latency_ms"] >= 0

    async def test_health_error_on_timeout(self) -> None:
        exchange = KalshiExchange()
        with patch.object(exchange, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = TimeoutError("Request timed out")
            result = await exchange.health_check()

        assert result["status"] == "error"
        assert "timed out" in result["error"]

    async def test_health_returns_exchange_name(self) -> None:
        """Each exchange should include its correct name in the result."""
        exchanges = [
            (PolymarketExchange(), "polymarket"),
            (KalshiExchange(), "kalshi"),
            (ManifoldExchange(), "manifold"),
            (MetaculusExchange(), "metaculus"),
        ]
        for exchange, expected_name in exchanges:
            with patch.object(exchange, "_get", new_callable=AsyncMock):
                result = await exchange.health_check()
            assert result["exchange"] == expected_name


class TestExchangeStatusCLI:
    """Tests for the exchange_status CLI command."""

    def test_exchange_status_command_exists(self) -> None:
        from arbiter.cli import app

        # Typer stores the callback function name, converting
        # underscores to hyphens for the CLI name automatically.
        callback_names = [
            cmd.callback.__name__
            for cmd in app.registered_commands
            if cmd.callback is not None
        ]
        assert "exchange_status" in callback_names
