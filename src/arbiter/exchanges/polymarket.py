"""Polymarket exchange connector.

Polymarket is a decentralized prediction market built on Polygon. It uses a
CLOB (Central Limit Order Book) model with REST and WebSocket APIs.

Docs: https://docs.polymarket.com
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

_DEFAULT_BASE_URL = "https://clob.polymarket.com"
_DEFAULT_GAMMA_URL = "https://gamma-api.polymarket.com"


class PolymarketExchange(BaseExchange):
    """Connector for the Polymarket prediction market.

    Uses both the CLOB API (for order books and trading data) and the
    Gamma API (for market metadata and event information).
    """

    name = ExchangeName.POLYMARKET

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        gamma_url: str = _DEFAULT_GAMMA_URL,
    ) -> None:
        config = ExchangeConfig(
            name=ExchangeName.POLYMARKET,
            api_key=api_key,
            api_secret=api_secret,
            base_url=gamma_url,
            rate_limit_per_second=5.0,
        )
        super().__init__(config)
        self._clob_url = base_url
        self._gamma_url = gamma_url

    async def fetch_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[Market]:
        """Fetch markets from the Polymarket Gamma API.

        Args:
            active_only: Only return active, tradeable markets.
            limit: Maximum number of markets to retrieve.

        Returns:
            List of normalized Market objects.
        """
        params: dict[str, object] = {"limit": limit, "order": "volume24hr"}
        if active_only:
            params["active"] = True
            params["closed"] = False

        response = await self._get("/markets", params=params)
        raw_markets = response.json()

        markets: list[Market] = []
        for raw in raw_markets:
            try:
                market = self._parse_market(raw)
                markets.append(market)
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping unparseable market: %s", exc)

        return markets

    async def fetch_market(self, market_id: str) -> Market:
        """Fetch a single Polymarket market by condition ID.

        Args:
            market_id: The Polymarket condition ID or slug.

        Returns:
            Normalized Market object.

        Raises:
            MarketNotFoundError: If the market is not found.
        """
        try:
            response = await self._get(f"/markets/{market_id}")
        except Exception as exc:
            raise MarketNotFoundError(self.name.value, market_id) from exc

        raw = response.json()
        if not raw:
            raise MarketNotFoundError(self.name.value, market_id)
        return self._parse_market(raw)

    async def fetch_order_book(self, market_id: str) -> OrderBook:
        """Fetch the order book from the Polymarket CLOB API.

        Args:
            market_id: The token ID for the CLOB.

        Returns:
            OrderBook snapshot with bids and asks.
        """
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self._clob_url}/book",
                params={"token_id": market_id},
            )
            response.raise_for_status()
        except Exception as exc:
            raise ExchangeError(
                self.name.value,
                f"Failed to fetch order book for {market_id}: {exc}",
            ) from exc

        data = response.json()
        bids = [
            OrderBookLevel(price=float(lvl["price"]), quantity=float(lvl["size"]))
            for lvl in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(lvl["price"]), quantity=float(lvl["size"]))
            for lvl in data.get("asks", [])
        ]

        return OrderBook(
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
            timestamp=datetime.now(UTC),
        )

    def _parse_market(self, raw: dict[str, Any]) -> Market:
        """Convert raw Polymarket API response to a Market model.

        Args:
            raw: Dictionary from the Gamma API.

        Returns:
            Normalized Market object.
        """
        status = MarketStatus.ACTIVE
        if raw.get("closed", False):
            status = MarketStatus.CLOSED
        if raw.get("resolved", False):
            status = MarketStatus.RESOLVED

        outcomes_raw = raw.get("outcomes", ["Yes", "No"])
        prices_raw = raw.get("outcomePrices", ["0.5", "0.5"])

        if isinstance(outcomes_raw, str):
            outcomes_raw = [o.strip() for o in outcomes_raw.split(",")]
        if isinstance(prices_raw, str):
            prices_raw = [p.strip() for p in prices_raw.split(",")]

        outcomes = []
        for name, price_str in zip(outcomes_raw, prices_raw, strict=False):
            price = float(price_str) if price_str else 0.5
            outcomes.append(Outcome(name=str(name), price=price, volume=0.0))

        contract_type = (
            ContractType.BINARY if len(outcomes) == 2 else ContractType.MULTI_OUTCOME
        )

        volume = float(raw.get("volume", 0) or 0)

        created_at = None
        if raw.get("startDate"):
            try:
                created_at = datetime.fromisoformat(
                    raw["startDate"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        closes_at = None
        if raw.get("endDate"):
            try:
                closes_at = datetime.fromisoformat(
                    raw["endDate"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return Market(
            id=str(raw.get("conditionId", raw.get("id", ""))),
            exchange=ExchangeName.POLYMARKET,
            title=raw.get("question", raw.get("title", "Unknown")),
            description=raw.get("description", ""),
            category=raw.get("category", ""),
            contract_type=contract_type,
            status=status,
            outcomes=outcomes,
            url=f"https://polymarket.com/event/{raw.get('slug', '')}",
            volume_total=volume,
            created_at=created_at,
            closes_at=closes_at,
            resolution=raw.get("resolution") or None,
            metadata={
                key: raw[key]
                for key in ("clobTokenIds", "tokens")
                if key in raw and raw[key] is not None
            },
            fetched_at=datetime.now(UTC),
        )
