"""Tests for arbiter top-level package imports and Exchange factory."""

from __future__ import annotations

import arbiter
from arbiter import (
    Arbiter,
    Exchange,
    ExchangeName,
    KalshiExchange,
    ManifoldExchange,
    Market,
    MetaculusExchange,
    PolymarketExchange,
)


class TestPackageImports:
    """Ensure the public API surface is correctly exported."""

    def test_version(self) -> None:
        assert arbiter.__version__ == "0.1.0"

    def test_arbiter_class(self) -> None:
        assert Arbiter is not None

    def test_exchange_factory(self) -> None:
        assert Exchange is not None

    def test_model_imports(self) -> None:
        assert Market is not None
        assert ExchangeName is not None


class TestExchangeFactory:
    """Tests for the Exchange convenience factory."""

    def test_polymarket(self) -> None:
        ex = Exchange.polymarket()
        assert isinstance(ex, PolymarketExchange)
        assert ex.name == ExchangeName.POLYMARKET

    def test_polymarket_with_key(self) -> None:
        ex = Exchange.polymarket(api_key="test-key")
        assert ex.config.api_key == "test-key"

    def test_kalshi(self) -> None:
        ex = Exchange.kalshi()
        assert isinstance(ex, KalshiExchange)
        assert ex.name == ExchangeName.KALSHI

    def test_kalshi_with_auth(self) -> None:
        ex = Exchange.kalshi(api_key="k", api_secret="s")
        assert ex.config.api_key == "k"
        assert ex.config.api_secret == "s"

    def test_metaculus(self) -> None:
        ex = Exchange.metaculus()
        assert isinstance(ex, MetaculusExchange)
        assert ex.name == ExchangeName.METACULUS

    def test_manifold(self) -> None:
        ex = Exchange.manifold()
        assert isinstance(ex, ManifoldExchange)
        assert ex.name == ExchangeName.MANIFOLD

    def test_manifold_with_key(self) -> None:
        ex = Exchange.manifold(api_key="mf-key")
        assert ex.config.api_key == "mf-key"
