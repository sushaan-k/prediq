"""Liquidity analysis for prediction markets.

Analyzes order book depth, spread, and estimates the price impact
of trades at various sizes. This is critical for understanding
the true cost of executing trades beyond the quoted price.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from arbiter.models import LiquidityProfile, Market, OrderBook

logger = logging.getLogger(__name__)


class LiquidityAnalyzer:
    """Analyzes market liquidity from order book data.

    Computes spread, depth at various levels, and estimates
    price impact for hypothetical trade sizes.
    """

    # Standard trade sizes for impact estimation
    DEFAULT_TRADE_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]

    def __init__(
        self,
        trade_sizes: list[int] | None = None,
    ) -> None:
        """Initialize the liquidity analyzer.

        Args:
            trade_sizes: Dollar amounts for price impact estimation.
                Defaults to [1K, 5K, 10K, 50K, 100K].
        """
        self.trade_sizes = trade_sizes or self.DEFAULT_TRADE_SIZES

    def analyze(self, market: Market) -> LiquidityProfile:
        """Analyze liquidity for a market.

        Uses order book data from the first outcome (typically YES).

        Args:
            market: Market with order book data attached to outcomes.

        Returns:
            LiquidityProfile with spread, depth, and impact estimates.
        """
        order_book = self._get_primary_order_book(market)
        if order_book is None:
            return LiquidityProfile(
                market=market.title,
                market_id=market.id,
                exchange=market.exchange,
                analyzed_at=datetime.now(UTC),
            )

        best_bid = order_book.best_bid
        best_ask = order_book.best_ask
        spread = order_book.spread

        depth_1pct = self._depth_within_pct(order_book, 0.01)
        depth_5pct = self._depth_within_pct(order_book, 0.05)

        impact = self._estimate_impact(order_book)

        return LiquidityProfile(
            market=market.title,
            market_id=market.id,
            exchange=market.exchange,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            depth_at_1pct=depth_1pct,
            depth_at_5pct=depth_5pct,
            estimated_impact=impact,
            analyzed_at=datetime.now(UTC),
        )

    def analyze_order_book(
        self,
        order_book: OrderBook,
        market_title: str = "",
        market_id: str = "",
        exchange: str = "",
    ) -> LiquidityProfile:
        """Analyze liquidity directly from an OrderBook object.

        Args:
            order_book: The order book to analyze.
            market_title: Human-readable market name.
            market_id: Exchange market identifier.
            exchange: Exchange name.

        Returns:
            LiquidityProfile with computed metrics.
        """
        from arbiter.models import ExchangeName

        exchange_enum = ExchangeName.POLYMARKET
        for ex in ExchangeName:
            if ex.value == exchange.lower():
                exchange_enum = ex
                break

        return LiquidityProfile(
            market=market_title,
            market_id=market_id,
            exchange=exchange_enum,
            best_bid=order_book.best_bid,
            best_ask=order_book.best_ask,
            spread=order_book.spread,
            depth_at_1pct=self._depth_within_pct(order_book, 0.01),
            depth_at_5pct=self._depth_within_pct(order_book, 0.05),
            estimated_impact=self._estimate_impact(order_book),
            analyzed_at=datetime.now(UTC),
        )

    def _depth_within_pct(self, order_book: OrderBook, pct: float) -> float:
        """Calculate total dollar depth within a percentage of midpoint.

        Args:
            order_book: The order book snapshot.
            pct: Percentage band around midpoint (e.g. 0.01 for 1%).

        Returns:
            Total dollar depth within the band.
        """
        midpoint = order_book.midpoint
        if midpoint is None:
            return 0.0

        lower = midpoint - pct
        upper = midpoint + pct
        depth = 0.0

        for level in order_book.bids:
            if level.price >= lower:
                depth += level.quantity

        for level in order_book.asks:
            if level.price <= upper:
                depth += level.quantity

        return depth

    def _estimate_impact(self, order_book: OrderBook) -> dict[int, float]:
        """Estimate price impact for various trade sizes.

        Walks through the order book to simulate execution and
        determine how much each trade size would move the price.

        Args:
            order_book: The order book snapshot.

        Returns:
            Dictionary mapping trade size ($) -> price impact.
        """
        impact: dict[int, float] = {}

        for size in self.trade_sizes:
            impact[size] = self._simulate_buy_impact(order_book, float(size))

        return impact

    @staticmethod
    def _simulate_buy_impact(order_book: OrderBook, trade_size: float) -> float:
        """Simulate buying a given dollar amount and compute price impact.

        Walks through ask levels filling the order. The impact is the
        difference between the final execution price and the initial
        best ask.

        Args:
            order_book: The order book snapshot.
            trade_size: Dollar amount to buy.

        Returns:
            Price impact (positive = price moved up).
        """
        if not order_book.asks:
            return 0.0

        initial_price = order_book.asks[0].price
        remaining = trade_size
        total_cost = 0.0
        total_qty = 0.0

        for level in order_book.asks:
            if remaining <= 0:
                break

            available = level.quantity
            fill = min(available, remaining)
            total_cost += fill * level.price
            total_qty += fill
            remaining -= fill

        if total_qty == 0:
            return 0.0

        vwap = total_cost / total_qty
        return vwap - initial_price

    @staticmethod
    def _get_primary_order_book(market: Market) -> OrderBook | None:
        """Get the primary order book from a market's outcomes.

        For binary markets, returns the YES outcome's order book.

        Args:
            market: The market.

        Returns:
            OrderBook if available, None otherwise.
        """
        for outcome in market.outcomes:
            if outcome.order_book is not None and (
                outcome.order_book.bids or outcome.order_book.asks
            ):
                return outcome.order_book
        return None
