"""Probability violation detection.

Detects markets where prices violate basic probability axioms:
- Binary markets: YES + NO should sum to ~1.0 (minus fees)
- Multi-outcome markets: all outcome prices should sum to ~1.0

Violations represent risk-free arbitrage opportunities.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from arbiter.models import (
    ContractType,
    Market,
    MultiOutcomeViolation,
    ProbabilityViolation,
)

logger = logging.getLogger(__name__)


class ViolationDetector:
    """Detects probability violations in prediction markets.

    A probability violation occurs when market prices are inconsistent
    with the axioms of probability theory, creating risk-free profit
    opportunities.
    """

    def __init__(
        self,
        binary_tolerance: float = 0.03,
        multi_tolerance: float = 0.05,
    ) -> None:
        """Initialize the violation detector.

        Args:
            binary_tolerance: Maximum acceptable deviation from 1.0
                for YES + NO price sums. Exchange fees typically account
                for ~2-3%, so 0.03 is a reasonable default.
            multi_tolerance: Maximum acceptable deviation from 1.0
                for multi-outcome price sums.
        """
        self.binary_tolerance = binary_tolerance
        self.multi_tolerance = multi_tolerance

    def detect_binary_violations(
        self, markets: list[Market]
    ) -> list[ProbabilityViolation]:
        """Scan binary markets for YES + NO pricing violations.

        A violation exists when YES + NO > 1.0 + tolerance (overpriced,
        sell both sides) or YES + NO < 1.0 - tolerance (underpriced,
        buy both sides).

        Args:
            markets: List of markets to scan.

        Returns:
            List of detected probability violations.
        """
        violations: list[ProbabilityViolation] = []

        for market in markets:
            if market.contract_type != ContractType.BINARY:
                continue

            yes_price = market.yes_price
            no_price = market.no_price

            if yes_price is None or no_price is None:
                continue

            price_sum = yes_price + no_price
            deviation = abs(price_sum - 1.0)

            if deviation <= self.binary_tolerance:
                continue

            implied_arb = price_sum - 1.0 if price_sum > 1.0 else 1.0 - price_sum

            volume = min(
                self._outcome_volume(market, "yes"),
                self._outcome_volume(market, "no"),
            )

            violation = ProbabilityViolation(
                market=market.title,
                market_id=market.id,
                exchange=market.exchange,
                yes_price=yes_price,
                no_price=no_price,
                price_sum=price_sum,
                implied_arb=implied_arb,
                volume_available=volume,
                detected_at=datetime.now(UTC),
            )
            violations.append(violation)
            logger.info(
                "Binary violation: %s on %s (sum=%.4f, arb=%.4f)",
                market.title,
                market.exchange.value,
                price_sum,
                implied_arb,
            )

        violations.sort(key=lambda v: v.implied_arb, reverse=True)
        return violations

    def detect_multi_outcome_violations(
        self, markets: list[Market]
    ) -> list[MultiOutcomeViolation]:
        """Scan multi-outcome markets for probability sum violations.

        All outcome probabilities should sum to approximately 1.0.
        Deviations beyond tolerance indicate mispricing.

        Args:
            markets: List of markets to scan.

        Returns:
            List of detected multi-outcome violations.
        """
        violations: list[MultiOutcomeViolation] = []

        for market in markets:
            if market.contract_type != ContractType.MULTI_OUTCOME:
                continue

            if len(market.outcomes) < 2:
                continue

            outcome_prices = {o.name: o.price for o in market.outcomes}
            price_sum = sum(outcome_prices.values())
            deviation = abs(price_sum - 1.0)

            if deviation <= self.multi_tolerance:
                continue

            violation = MultiOutcomeViolation(
                market=market.title,
                market_id=market.id,
                exchange=market.exchange,
                outcomes=outcome_prices,
                price_sum=price_sum,
                deviation=deviation,
                detected_at=datetime.now(UTC),
            )
            violations.append(violation)
            logger.info(
                "Multi-outcome violation: %s on %s (sum=%.4f, dev=%.4f)",
                market.title,
                market.exchange.value,
                price_sum,
                deviation,
            )

        violations.sort(key=lambda v: v.deviation, reverse=True)
        return violations

    def detect_all(
        self, markets: list[Market]
    ) -> tuple[list[ProbabilityViolation], list[MultiOutcomeViolation]]:
        """Run all violation checks on a list of markets.

        Args:
            markets: List of markets to scan.

        Returns:
            Tuple of (binary_violations, multi_outcome_violations).
        """
        binary = self.detect_binary_violations(markets)
        multi = self.detect_multi_outcome_violations(markets)
        return binary, multi

    @staticmethod
    def _outcome_volume(market: Market, side: str) -> float:
        """Get volume for a specific outcome side.

        Args:
            market: The market.
            side: 'yes' or 'no'.

        Returns:
            Volume in dollars, or total market volume as fallback.
        """
        for outcome in market.outcomes:
            if outcome.name.lower() == side:
                if outcome.volume > 0:
                    return outcome.volume
                if outcome.order_book:
                    depth = sum(lvl.quantity for lvl in outcome.order_book.bids) + sum(
                        lvl.quantity for lvl in outcome.order_book.asks
                    )
                    if depth > 0:
                        return depth
        return market.volume_total
