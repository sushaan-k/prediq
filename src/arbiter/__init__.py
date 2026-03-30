"""arbiter -- Cross-exchange prediction market analytics engine.

Provides a unified interface for ingesting, normalizing, and analyzing
prediction market data across Polymarket, Kalshi, Metaculus, and Manifold.

Example::

    from arbiter import Arbiter, Exchange

    async with Arbiter(exchanges=[
        Exchange.polymarket(),
        Exchange.kalshi(api_key="..."),
        Exchange.manifold(),
    ]) as arb:
        for d in await arb.divergences(min_spread=0.03):
            print(f"{d.event}: {d.spread_pct:.1%} spread")
"""

from __future__ import annotations

from arbiter.engine import Arbiter
from arbiter.exchanges.base import BaseExchange
from arbiter.exchanges.kalshi import KalshiExchange
from arbiter.exchanges.manifold import ManifoldExchange
from arbiter.exchanges.metaculus import MetaculusExchange
from arbiter.exchanges.polymarket import PolymarketExchange
from arbiter.models import (
    Alert,
    ContractType,
    Divergence,
    EfficiencyMetrics,
    ExchangeConfig,
    ExchangeName,
    LiquidityProfile,
    Market,
    MarketPair,
    MarketQuality,
    MarketStatus,
    MultiOutcomeViolation,
    OrderBook,
    OrderBookLevel,
    Outcome,
    ProbabilityViolation,
    Side,
)

__version__ = "0.1.0"


class Exchange:
    """Factory for creating exchange connectors.

    Provides a clean constructor-based API matching the spec::

        Exchange.polymarket(api_key="...")
        Exchange.kalshi(api_key="...", api_secret="...")
        Exchange.metaculus()
        Exchange.manifold()
    """

    @staticmethod
    def polymarket(
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://clob.polymarket.com",
        gamma_url: str = "https://gamma-api.polymarket.com",
    ) -> PolymarketExchange:
        """Create a Polymarket exchange connector.

        Args:
            api_key: Optional API key for authenticated endpoints.
            api_secret: Optional API secret.
            base_url: CLOB API base URL.
            gamma_url: Gamma metadata API base URL.

        Returns:
            Configured PolymarketExchange instance.
        """
        return PolymarketExchange(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            gamma_url=gamma_url,
        )

    @staticmethod
    def kalshi(
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
    ) -> KalshiExchange:
        """Create a Kalshi exchange connector.

        Args:
            api_key: API key for Kalshi.
            api_secret: API secret for Kalshi.
            base_url: Kalshi API base URL.

        Returns:
            Configured KalshiExchange instance.
        """
        return KalshiExchange(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
        )

    @staticmethod
    def metaculus(
        api_key: str | None = None,
        base_url: str = "https://www.metaculus.com/api2",
    ) -> MetaculusExchange:
        """Create a Metaculus exchange connector.

        Args:
            api_key: Optional Metaculus API token. Falls back to the
                ``METACULUS_API_KEY`` environment variable.
            base_url: Metaculus API base URL.

        Returns:
            Configured MetaculusExchange instance.
        """
        return MetaculusExchange(api_key=api_key, base_url=base_url)

    @staticmethod
    def manifold(
        api_key: str | None = None,
        base_url: str = "https://api.manifold.markets/v0",
    ) -> ManifoldExchange:
        """Create a Manifold Markets exchange connector.

        Args:
            api_key: Optional API key for Manifold.
            base_url: Manifold API base URL.

        Returns:
            Configured ManifoldExchange instance.
        """
        return ManifoldExchange(api_key=api_key, base_url=base_url)


__all__ = [
    "Alert",
    "Arbiter",
    "BaseExchange",
    "ContractType",
    "Divergence",
    "EfficiencyMetrics",
    "Exchange",
    "ExchangeConfig",
    "ExchangeName",
    "KalshiExchange",
    "LiquidityProfile",
    "ManifoldExchange",
    "Market",
    "MarketPair",
    "MarketQuality",
    "MarketStatus",
    "MetaculusExchange",
    "MultiOutcomeViolation",
    "OrderBook",
    "OrderBookLevel",
    "Outcome",
    "PolymarketExchange",
    "ProbabilityViolation",
    "Side",
    "__version__",
]
