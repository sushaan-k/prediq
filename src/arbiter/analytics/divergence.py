"""Cross-exchange price divergence detection.

Identifies when the same event is priced differently on two exchanges,
signaling potential arbitrage opportunities or market inefficiencies.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from arbiter.matching.normalizer import PriceNormalizer
from arbiter.models import (
    ContractType,
    Divergence,
    Market,
    MarketPair,
)

logger = logging.getLogger(__name__)


class DivergenceDetector:
    """Detects price divergences between matched market pairs.

    A divergence occurs when the same outcome is priced differently
    on two exchanges by more than a configurable threshold.
    """

    def __init__(
        self,
        min_spread: float = 0.02,
        min_liquidity: float = 0.0,
    ) -> None:
        """Initialize the divergence detector.

        Args:
            min_spread: Minimum absolute spread to flag (default 2 cents).
            min_liquidity: Minimum dollar liquidity to consider.
        """
        self.min_spread = min_spread
        self.min_liquidity = min_liquidity
        self._normalizer = PriceNormalizer()

    def detect(self, pairs: list[MarketPair]) -> list[Divergence]:
        """Scan matched market pairs for price divergences.

        For binary markets, compares the YES price across exchanges.
        For multi-outcome markets, compares each named outcome.

        Args:
            pairs: List of matched market pairs from the SemanticMatcher.

        Returns:
            List of Divergence objects, sorted by spread descending.
        """
        divergences: list[Divergence] = []

        for pair in pairs:
            pair_divs = self._check_pair(pair)
            divergences.extend(pair_divs)

        divergences.sort(key=lambda d: d.spread, reverse=True)
        return divergences

    def _check_pair(self, pair: MarketPair) -> list[Divergence]:
        """Check a single market pair for divergences.

        Args:
            pair: A matched pair of markets.

        Returns:
            List of divergences found in this pair.
        """
        ma = self._normalizer.normalize_market(pair.market_a)
        mb = self._normalizer.normalize_market(pair.market_b)

        if ma.contract_type == ContractType.BINARY:
            return self._check_binary_pair(ma, mb)
        return self._check_multi_outcome_pair(ma, mb)

    def _check_binary_pair(self, ma: Market, mb: Market) -> list[Divergence]:
        """Check binary market pair for YES price divergence.

        Args:
            ma: First market (normalized).
            mb: Second market (normalized).

        Returns:
            List with 0 or 1 divergence.
        """
        price_a = ma.yes_price
        price_b = mb.yes_price

        if price_a is None or price_b is None:
            return []

        spread = abs(price_a - price_b)
        if spread < self.min_spread:
            return []

        midpoint = (price_a + price_b) / 2.0
        spread_pct = spread / midpoint if midpoint > 0 else 0.0

        liq_a = self._estimate_liquidity(ma)
        liq_b = self._estimate_liquidity(mb)

        if (
            liq_a < self.min_liquidity
            and liq_b < self.min_liquidity
            and self.min_liquidity > 0
        ):
            return []

        fee_a = self._normalizer.estimate_fee(ma.exchange, min(liq_a, liq_b))
        fee_b = self._normalizer.estimate_fee(mb.exchange, min(liq_a, liq_b))
        tradeable = min(liq_a, liq_b)
        net_profit = max(0.0, spread * tradeable - fee_a - fee_b)

        return [
            Divergence(
                event=ma.title,
                outcome="Yes",
                exchange_a=ma.exchange,
                exchange_b=mb.exchange,
                price_a=price_a,
                price_b=price_b,
                spread=spread,
                spread_pct=spread_pct,
                liquidity_a=liq_a,
                liquidity_b=liq_b,
                net_arb_profit_estimate=net_profit,
                window_opened=datetime.now(UTC),
                market_a_id=ma.id,
                market_b_id=mb.id,
            )
        ]

    def _check_multi_outcome_pair(self, ma: Market, mb: Market) -> list[Divergence]:
        """Check multi-outcome market pair for per-outcome divergences.

        Matches outcomes by name and checks each pair.

        Args:
            ma: First market (normalized).
            mb: Second market (normalized).

        Returns:
            List of divergences found across outcomes.
        """
        outcomes_b = {o.name.lower(): o for o in mb.outcomes}
        divergences: list[Divergence] = []

        for oa in ma.outcomes:
            ob = outcomes_b.get(oa.name.lower())
            if ob is None:
                continue

            liq_a = oa.volume
            liq_b = ob.volume
            if (
                liq_a < self.min_liquidity
                and liq_b < self.min_liquidity
                and self.min_liquidity > 0
            ):
                continue

            spread = abs(oa.price - ob.price)
            if spread < self.min_spread:
                continue

            midpoint = (oa.price + ob.price) / 2.0
            spread_pct = spread / midpoint if midpoint > 0 else 0.0

            divergences.append(
                Divergence(
                    event=ma.title,
                    outcome=oa.name,
                    exchange_a=ma.exchange,
                    exchange_b=mb.exchange,
                    price_a=oa.price,
                    price_b=ob.price,
                    spread=spread,
                    spread_pct=spread_pct,
                    liquidity_a=liq_a,
                    liquidity_b=liq_b,
                    window_opened=datetime.now(UTC),
                    market_a_id=ma.id,
                    market_b_id=mb.id,
                )
            )

        return divergences

    @staticmethod
    def _estimate_liquidity(market: Market) -> float:
        """Estimate available liquidity for a market.

        Uses order book depth if available, otherwise falls back
        to total market volume as a rough proxy.

        Args:
            market: The market to estimate liquidity for.

        Returns:
            Estimated liquidity in dollars.
        """
        total_depth = 0.0
        for outcome in market.outcomes:
            if outcome.order_book:
                for level in outcome.order_book.bids:
                    total_depth += level.quantity
                for level in outcome.order_book.asks:
                    total_depth += level.quantity

        if total_depth > 0:
            return total_depth

        return market.volume_total * 0.01
