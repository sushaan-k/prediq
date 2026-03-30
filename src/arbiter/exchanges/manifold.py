"""Manifold Markets exchange connector.

Manifold is a play-money prediction market platform with a public API.
Despite using play money (Mana), its markets are widely referenced
and provide useful probability signals.

Docs: https://docs.manifold.markets/api
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from arbiter.exceptions import MarketNotFoundError
from arbiter.exchanges.base import BaseExchange
from arbiter.models import (
    ContractType,
    ExchangeConfig,
    ExchangeName,
    Market,
    MarketStatus,
    OrderBook,
    Outcome,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.manifold.markets/v0"


class ManifoldExchange(BaseExchange):
    """Connector for Manifold Markets.

    Manifold uses an automated market maker rather than a CLOB,
    so order book data is synthesized from the AMM curve.
    """

    name = ExchangeName.MANIFOLD

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        config = ExchangeConfig(
            name=ExchangeName.MANIFOLD,
            api_key=api_key,
            base_url=base_url,
            rate_limit_per_second=5.0,
        )
        super().__init__(config)

    def _build_headers(self) -> dict[str, str]:
        """Build headers with optional Manifold API key."""
        headers = super()._build_headers()
        if self.config.api_key:
            headers["Authorization"] = f"Key {self.config.api_key}"
        return headers

    async def fetch_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[Market]:
        """Fetch markets from Manifold Markets.

        Args:
            active_only: Only return open markets.
            limit: Maximum number of markets.

        Returns:
            List of normalized Market objects.
        """
        params: dict[str, object] = {
            "limit": limit,
            "sort": "updated-time",
            "order": "desc",
        }

        response = await self._get("/markets", params=params)
        raw_markets = response.json()

        markets: list[Market] = []
        for raw in raw_markets:
            try:
                market = self._parse_market(raw)
                if active_only and market.status != MarketStatus.ACTIVE:
                    continue
                markets.append(market)
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping unparseable Manifold market: %s", exc)

        return markets[:limit]

    async def fetch_market(self, market_id: str) -> Market:
        """Fetch a single Manifold market by slug or ID.

        Args:
            market_id: The Manifold market slug or ID.

        Returns:
            Normalized Market object.

        Raises:
            MarketNotFoundError: If the market does not exist.
        """
        try:
            response = await self._get(f"/market/{market_id}")
        except Exception as exc:
            raise MarketNotFoundError(self.name.value, market_id) from exc

        raw = response.json()
        if not raw:
            raise MarketNotFoundError(self.name.value, market_id)
        return self._parse_market(raw)

    async def fetch_order_book(self, market_id: str) -> OrderBook:
        """Return a synthetic order book from Manifold's AMM.

        Manifold uses an automated market maker, so there is no
        traditional order book. We return an empty book.

        Args:
            market_id: The market ID.

        Returns:
            An empty OrderBook.
        """
        return OrderBook(timestamp=datetime.now(UTC))

    def _parse_market(self, raw: dict[str, Any]) -> Market:
        """Convert raw Manifold API response to a Market model.

        Args:
            raw: Dictionary from the Manifold API.

        Returns:
            Normalized Market object.
        """
        is_resolved = raw.get("isResolved", False)
        close_time_ms = raw.get("closeTime")

        status = MarketStatus.ACTIVE
        if is_resolved:
            status = MarketStatus.RESOLVED
        elif close_time_ms:
            close_dt = datetime.fromtimestamp(close_time_ms / 1000, tz=UTC)
            if close_dt < datetime.now(UTC):
                status = MarketStatus.CLOSED

        mechanism = raw.get("outcomeType", "BINARY")

        if mechanism == "BINARY":
            prob = float(raw.get("probability", 0.5))
            prob = min(max(prob, 0.0), 1.0)
            outcomes = [
                Outcome(name="Yes", price=prob, volume=0.0),
                Outcome(name="No", price=1.0 - prob, volume=0.0),
            ]
            contract_type = ContractType.BINARY
        elif mechanism == "MULTIPLE_CHOICE":
            answers = raw.get("answers", [])
            outcomes = []
            for ans in answers:
                p = float(ans.get("probability", 0.0))
                outcomes.append(Outcome(name=ans.get("text", "?"), price=p, volume=0.0))
            if len(outcomes) < 2:
                outcomes = [
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ]
            contract_type = ContractType.MULTI_OUTCOME
        else:
            outcomes = [
                Outcome(name="Yes", price=0.5, volume=0.0),
                Outcome(name="No", price=0.5, volume=0.0),
            ]
            contract_type = ContractType.BINARY

        volume = float(raw.get("volume", 0) or 0)

        created_at = None
        if raw.get("createdTime"):
            try:
                created_at = datetime.fromtimestamp(raw["createdTime"] / 1000, tz=UTC)
            except (ValueError, TypeError, OSError):
                pass

        closes_at = None
        if close_time_ms:
            try:
                closes_at = datetime.fromtimestamp(close_time_ms / 1000, tz=UTC)
            except (ValueError, TypeError, OSError):
                pass

        resolution = None
        if is_resolved:
            resolution = raw.get("resolution")
            if resolution is not None:
                resolution = str(resolution)

        return Market(
            id=raw.get("id", ""),
            exchange=ExchangeName.MANIFOLD,
            title=raw.get("question", "Unknown"),
            description=raw.get("textDescription", ""),
            category=raw.get("groupSlugs", [""])[0] if raw.get("groupSlugs") else "",
            contract_type=contract_type,
            status=status,
            outcomes=outcomes,
            url=raw.get("url", f"https://manifold.markets/{raw.get('slug', '')}"),
            volume_total=volume,
            created_at=created_at,
            closes_at=closes_at,
            resolution=resolution,
            fetched_at=datetime.now(UTC),
        )
