"""Kalshi exchange connector.

Kalshi is an SEC-regulated prediction market exchange (CFTC-regulated DCM).
It offers event contracts on politics, economics, weather, and more.

Docs: https://trading-api.readme.io/reference
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from arbiter.exceptions import ExchangeError, MarketNotFoundError
from arbiter.exchanges.base import BaseExchange
from arbiter.models import (
    ContractType,
    ExchangeConfig,
    ExchangeName,
    Market,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    Outcome,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiExchange(BaseExchange):
    """Connector for the Kalshi prediction market exchange.

    Kalshi prices are in cents (0-99), which we normalize to [0, 1].
    """

    name = ExchangeName.KALSHI

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        config = ExchangeConfig(
            name=ExchangeName.KALSHI,
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            rate_limit_per_second=3.0,
        )
        super().__init__(config)

    def _build_headers(self) -> dict[str, str]:
        """Build headers with Kalshi authentication."""
        headers = super()._build_headers()
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def fetch_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[Market]:
        """Fetch markets from the Kalshi API.

        Args:
            active_only: Only return actively trading markets.
            limit: Maximum number of markets to return.

        Returns:
            List of normalized Market objects.
        """
        params: dict[str, object] = {"limit": limit}
        if active_only:
            params["status"] = "open"

        response = await self._get("/markets", params=params)
        data = response.json()
        raw_markets = data.get("markets", [])

        markets: list[Market] = []
        for raw in raw_markets:
            try:
                market = self._parse_market(raw)
                markets.append(market)
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping unparseable Kalshi market: %s", exc)

        return markets

    async def fetch_market(self, market_id: str) -> Market:
        """Fetch a single Kalshi market by ticker.

        Args:
            market_id: The Kalshi market ticker.

        Returns:
            Normalized Market object.

        Raises:
            MarketNotFoundError: If the market does not exist.
        """
        try:
            response = await self._get(f"/markets/{market_id}")
        except Exception as exc:
            raise MarketNotFoundError(self.name.value, market_id) from exc

        data = response.json()
        raw = data.get("market", data)
        if not raw:
            raise MarketNotFoundError(self.name.value, market_id)
        return self._parse_market(raw)

    async def fetch_order_book(self, market_id: str) -> OrderBook:
        """Fetch the order book for a Kalshi market.

        Args:
            market_id: The Kalshi market ticker.

        Returns:
            Normalized OrderBook snapshot.
        """
        try:
            response = await self._get(
                f"/markets/{market_id}/orderbook",
                params={"depth": 20},
            )
        except Exception as exc:
            raise ExchangeError(
                self.name.value,
                f"Failed to fetch order book for {market_id}: {exc}",
            ) from exc

        data = response.json()
        ob = data.get("orderbook", data)

        bids = [
            OrderBookLevel(
                price=float(lvl[0]) / 100.0,
                quantity=float(lvl[1]),
            )
            for lvl in ob.get("yes", [])
        ]
        asks = [
            OrderBookLevel(
                price=1.0 - float(lvl[0]) / 100.0,
                quantity=float(lvl[1]),
            )
            for lvl in ob.get("no", [])
        ]

        return OrderBook(
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
            timestamp=datetime.now(UTC),
        )

    def _parse_market(self, raw: dict[str, Any]) -> Market:
        """Convert raw Kalshi API response to a Market model.

        Kalshi prices are in cents. We normalize to [0, 1].

        Args:
            raw: Dictionary from the Kalshi API.

        Returns:
            Normalized Market object.
        """
        status_map = {
            "open": MarketStatus.ACTIVE,
            "closed": MarketStatus.CLOSED,
            "settled": MarketStatus.RESOLVED,
        }
        raw_status = raw.get("status", "open").lower()
        status = status_map.get(raw_status, MarketStatus.ACTIVE)

        yes_price = float(raw.get("yes_ask", raw.get("last_price", 50))) / 100.0
        no_price = 1.0 - yes_price

        outcomes = [
            Outcome(name="Yes", price=min(max(yes_price, 0.0), 1.0), volume=0.0),
            Outcome(name="No", price=min(max(no_price, 0.0), 1.0), volume=0.0),
        ]

        volume = float(raw.get("volume", 0) or 0)

        closes_at = None
        if raw.get("close_time"):
            try:
                closes_at = datetime.fromisoformat(
                    raw["close_time"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return Market(
            id=raw.get("ticker", raw.get("id", "")),
            exchange=ExchangeName.KALSHI,
            title=raw.get("title", raw.get("question", "Unknown")),
            description=raw.get("subtitle", ""),
            category=raw.get("category", raw.get("event_category", "")),
            contract_type=ContractType.BINARY,
            status=status,
            outcomes=outcomes,
            url=f"https://kalshi.com/markets/{raw.get('ticker', '')}",
            volume_total=volume,
            closes_at=closes_at,
            resolution=raw.get("result") or None,
            fetched_at=datetime.now(UTC),
        )
