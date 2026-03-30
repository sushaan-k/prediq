"""Metaculus exchange connector.

Metaculus is a forecasting platform focused on science, technology, and
geopolitics. Unlike CLOB-based exchanges, Metaculus uses aggregated
community predictions rather than order books.

Docs: https://www.metaculus.com/api/
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

_DEFAULT_BASE_URL = "https://www.metaculus.com/api2"


class MetaculusExchange(BaseExchange):
    """Connector for the Metaculus forecasting platform.

    Metaculus does not have a traditional order book. Prices represent
    the community's aggregated probability estimate.

    Note: The Metaculus API may require an API key for reliable access.
    Set the ``api_key`` parameter or the ``METACULUS_API_KEY`` environment
    variable. Without an API key, requests may be rejected with a 403
    status code.
    """

    name = ExchangeName.METACULUS

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        import os

        resolved_key = api_key or os.environ.get("METACULUS_API_KEY")
        config = ExchangeConfig(
            name=ExchangeName.METACULUS,
            api_key=resolved_key,
            base_url=base_url,
            rate_limit_per_second=2.0,
        )
        super().__init__(config)

    def _build_headers(self) -> dict[str, str]:
        """Build headers, including the Metaculus API token when available."""
        headers = super()._build_headers()
        if self.config.api_key:
            headers["Authorization"] = f"Token {self.config.api_key}"
        return headers

    async def fetch_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[Market]:
        """Fetch questions from the Metaculus API.

        Args:
            active_only: Only return open questions.
            limit: Maximum number of questions.

        Returns:
            List of normalized Market objects.
        """
        params: dict[str, object] = {
            "limit": limit,
            "order_by": "-activity",
            "type": "forecast",
        }
        if active_only:
            params["status"] = "open"

        response = await self._get("/questions/", params=params)
        data = response.json()
        results = data.get("results", [])

        markets: list[Market] = []
        for raw in results:
            try:
                market = self._parse_market(raw)
                markets.append(market)
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping unparseable Metaculus question: %s", exc)

        return markets

    async def fetch_market(self, market_id: str) -> Market:
        """Fetch a single Metaculus question by ID.

        Args:
            market_id: The Metaculus question ID (numeric string).

        Returns:
            Normalized Market object.

        Raises:
            MarketNotFoundError: If the question does not exist.
        """
        try:
            response = await self._get(f"/questions/{market_id}/")
        except Exception as exc:
            raise MarketNotFoundError(self.name.value, market_id) from exc

        raw = response.json()
        if not raw or "id" not in raw:
            raise MarketNotFoundError(self.name.value, market_id)
        return self._parse_market(raw)

    async def fetch_order_book(self, market_id: str) -> OrderBook:
        """Return an empty order book (Metaculus has no order book).

        Metaculus uses aggregated community predictions, not a CLOB.

        Args:
            market_id: The question ID.

        Returns:
            An empty OrderBook.
        """
        return OrderBook(timestamp=datetime.now(UTC))

    def _parse_market(self, raw: dict[str, Any]) -> Market:
        """Convert raw Metaculus API response to a Market model.

        Args:
            raw: Dictionary from the Metaculus API.

        Returns:
            Normalized Market object.
        """
        question_id = str(raw.get("id", ""))

        status = MarketStatus.ACTIVE
        if raw.get("resolution") is not None:
            status = MarketStatus.RESOLVED
        elif raw.get("active_state") == "CLOSED":
            status = MarketStatus.CLOSED

        community_prediction = raw.get("community_prediction", {})
        prob = 0.5
        if isinstance(community_prediction, dict):
            full = community_prediction.get("full", {})
            if isinstance(full, dict):
                prob = float(full.get("q2", 0.5))
        elif isinstance(community_prediction, (int, float)):
            prob = float(community_prediction)

        prob = min(max(prob, 0.0), 1.0)

        outcomes = [
            Outcome(name="Yes", price=prob, volume=0.0),
            Outcome(name="No", price=1.0 - prob, volume=0.0),
        ]

        created_at = None
        if raw.get("created_time"):
            try:
                created_at = datetime.fromisoformat(
                    str(raw["created_time"]).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        closes_at = None
        if raw.get("close_time"):
            try:
                closes_at = datetime.fromisoformat(
                    str(raw["close_time"]).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        votes = int(raw.get("number_of_predictions", 0) or 0)

        return Market(
            id=question_id,
            exchange=ExchangeName.METACULUS,
            title=raw.get("title", "Unknown"),
            description=raw.get("description", ""),
            category=raw.get("group", ""),
            contract_type=ContractType.BINARY,
            status=status,
            outcomes=outcomes,
            url=f"https://www.metaculus.com/questions/{question_id}/",
            volume_total=float(votes),
            created_at=created_at,
            closes_at=closes_at,
            resolution=str(raw["resolution"])
            if raw.get("resolution") is not None
            else None,
            fetched_at=datetime.now(UTC),
        )
