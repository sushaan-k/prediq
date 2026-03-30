"""Price and contract normalization across exchanges.

Different exchanges represent prices differently:
- Polymarket: 0.0 to 1.0 (probability)
- Kalshi: 0 to 99 (cents)
- Manifold: 0.0 to 1.0 (probability, play money)
- Metaculus: 0.0 to 1.0 (community forecast)

This module provides normalization to a common [0, 1] probability scale.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC

from arbiter.models import ExchangeName, Market, Outcome

logger = logging.getLogger(__name__)


class PriceNormalizer:
    """Normalizes prices and contract structures across exchanges.

    All prices are converted to a [0, 1] probability scale.
    Timestamps are aligned to UTC.
    """

    # Fee schedules by exchange (approximate, for cost estimation)
    FEE_RATES: dict[ExchangeName, float] = {
        ExchangeName.POLYMARKET: 0.02,  # ~2% effective fee
        ExchangeName.KALSHI: 0.01,  # ~1% fee on profit
        ExchangeName.METACULUS: 0.0,  # No fees (not a real market)
        ExchangeName.MANIFOLD: 0.0,  # Play money, no real fees
    }

    def normalize_price(self, price: float, exchange: ExchangeName) -> float:
        """Normalize a price to the [0, 1] probability scale.

        Args:
            price: Raw price from the exchange.
            exchange: Which exchange the price came from.

        Returns:
            Normalized price in [0, 1].
        """
        if exchange == ExchangeName.KALSHI and price > 1.0:
            # Kalshi prices may arrive in cents (0-99) -- normalize
            price = price / 100.0
        return min(max(price, 0.0), 1.0)

    def normalize_market(self, market: Market) -> Market:
        """Normalize all prices and timestamps in a Market.

        Args:
            market: Market with potentially exchange-specific pricing.

        Returns:
            New Market with normalized prices and UTC timestamps.
        """
        normalized_outcomes = []
        for outcome in market.outcomes:
            norm_price = self.normalize_price(outcome.price, market.exchange)
            normalized_outcomes.append(
                Outcome(
                    name=outcome.name,
                    price=norm_price,
                    order_book=outcome.order_book,
                    volume=outcome.volume,
                )
            )

        return market.model_copy(
            update={
                "outcomes": normalized_outcomes,
                "fetched_at": market.fetched_at.replace(tzinfo=UTC)
                if market.fetched_at.tzinfo is None
                else market.fetched_at,
            }
        )

    def estimate_fee(self, exchange: ExchangeName, trade_size: float) -> float:
        """Estimate the fee for a trade on a given exchange.

        Args:
            exchange: The exchange.
            trade_size: Trade size in dollars.

        Returns:
            Estimated fee in dollars.
        """
        rate = self.FEE_RATES.get(exchange, 0.02)
        return trade_size * rate

    @staticmethod
    def clean_title(title: str) -> str:
        """Clean and normalize a market title for matching.

        Strips punctuation, normalizes whitespace, and lowercases.

        Args:
            title: Raw market title.

        Returns:
            Cleaned title string.
        """
        title = title.lower().strip()
        title = re.sub(r"[^\w\s]", " ", title)
        title = re.sub(r"\s+", " ", title)
        return title.strip()
