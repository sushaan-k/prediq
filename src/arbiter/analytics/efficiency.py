"""Efficiency metrics for prediction markets.

Measures how quickly and accurately markets incorporate new information,
including price discovery speed, arbitrage window duration, and
information incorporation rate.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from arbiter.models import Divergence, EfficiencyMetrics, Market

logger = logging.getLogger(__name__)


class EfficiencyAnalyzer:
    """Analyzes market efficiency over time.

    Uses historical divergence data and price series to compute
    metrics that quantify how well markets process information.
    """

    def __init__(self) -> None:
        """Initialize the efficiency analyzer."""

    def compute_arb_window_stats(
        self,
        divergences: list[Divergence],
        window_durations_minutes: list[float] | None = None,
    ) -> dict[str, float]:
        """Compute statistics on arbitrage window durations.

        An arbitrage window is the time period during which a price
        divergence persists between two exchanges.

        Args:
            divergences: Historical divergence records.
            window_durations_minutes: Pre-computed durations in minutes.
                If None, uses divergence timestamps for estimation.

        Returns:
            Dictionary with 'mean', 'median', 'p95' window durations.
        """
        if window_durations_minutes:
            durations = sorted(window_durations_minutes)
        elif divergences:
            # Estimate from divergence data — spread magnitude correlates
            # with window duration (larger spreads take longer to close)
            durations = sorted(max(1.0, d.spread * 500.0) for d in divergences)
        else:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0}

        n = len(durations)
        mean_val = sum(durations) / n
        median_val = durations[n // 2]
        p95_idx = int(n * 0.95)
        p95_val = durations[min(p95_idx, n - 1)]

        return {
            "mean": mean_val,
            "median": median_val,
            "p95": p95_val,
        }

    def compute_price_discovery_speed(
        self,
        price_series: list[tuple[datetime, float]],
        final_price: float,
        threshold: float = 0.90,
    ) -> float:
        """Compute how quickly price reaches threshold of final value.

        Measures the minutes it takes for the market price to first
        reach a given percentage of its final (resolution) value.

        Args:
            price_series: List of (timestamp, price) tuples,
                sorted chronologically.
            final_price: The final resolved price (0 or 1 for binary).
            threshold: Fraction of distance to final price (default 90%).

        Returns:
            Minutes until price first reached threshold of final value.
            Returns 0.0 if series is empty or starts at final price.
        """
        if not price_series or len(price_series) < 2:
            return 0.0

        start_time = price_series[0][0]
        start_price = price_series[0][1]

        target_distance = abs(final_price - start_price) * threshold
        if target_distance == 0:
            return 0.0

        target_price = start_price + target_distance * (
            1.0 if final_price > start_price else -1.0
        )

        for ts, price in price_series[1:]:
            if (final_price > start_price and price >= target_price) or (
                final_price < start_price and price <= target_price
            ):
                delta = (ts - start_time).total_seconds() / 60.0
                return max(0.0, delta)

        return 0.0

    def compute_information_ratio(
        self,
        price_changes: list[float],
        time_intervals_minutes: list[float],
    ) -> float:
        """Compute the information incorporation rate.

        Measures the ratio of directional price movement to total
        price volatility, normalized by time. Higher values indicate
        more efficient information processing.

        Args:
            price_changes: Sequence of price changes (deltas).
            time_intervals_minutes: Time between each observation.

        Returns:
            Information ratio (higher = more efficient).
        """
        if not price_changes or not time_intervals_minutes:
            return 0.0

        n = min(len(price_changes), len(time_intervals_minutes))
        if n == 0:
            return 0.0

        total_directional = abs(sum(price_changes[:n]))
        total_absolute = sum(abs(pc) for pc in price_changes[:n])
        total_time = sum(time_intervals_minutes[:n])

        if total_absolute == 0 or total_time == 0:
            return 0.0

        # Directional efficiency: what fraction of total movement was
        # in the "right" direction
        directional_efficiency = total_directional / total_absolute

        # Time efficiency: movement per unit time
        speed = total_absolute / total_time

        return directional_efficiency * speed

    def analyze_market(
        self,
        market: Market,
        divergences: list[Divergence] | None = None,
        price_history: list[tuple[datetime, float]] | None = None,
    ) -> EfficiencyMetrics:
        """Compute efficiency metrics for a single market.

        Args:
            market: The market to analyze.
            divergences: Historical divergences involving this market.
            price_history: Historical price series for this market.

        Returns:
            EfficiencyMetrics for the market.
        """
        arb_window = 0.0
        if divergences:
            stats = self.compute_arb_window_stats(divergences)
            arb_window = stats["mean"]

        discovery_speed = 0.0
        info_ratio = 0.0

        if price_history and len(price_history) >= 2:
            final = price_history[-1][1]
            discovery_speed = self.compute_price_discovery_speed(price_history, final)

            changes = [
                price_history[i][1] - price_history[i - 1][1]
                for i in range(1, len(price_history))
            ]
            intervals = [
                (price_history[i][0] - price_history[i - 1][0]).total_seconds() / 60.0
                for i in range(1, len(price_history))
            ]
            info_ratio = self.compute_information_ratio(changes, intervals)

        return EfficiencyMetrics(
            market=market.title,
            market_id=market.id,
            exchange=market.exchange,
            price_discovery_speed_minutes=discovery_speed,
            avg_arb_window_minutes=arb_window,
            information_ratio=info_ratio,
            computed_at=datetime.now(UTC),
        )
