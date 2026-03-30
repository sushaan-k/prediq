"""Market quality scoring for prediction markets.

Evaluates how well a market (or set of markets) forecasts outcomes.
Metrics include Brier score, calibration error, and manipulation detection.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from arbiter.exceptions import InsufficientDataError
from arbiter.models import ExchangeName, Market, MarketQuality, MarketStatus

logger = logging.getLogger(__name__)


class QualityScorer:
    """Scores prediction market quality using resolved market data.

    Computes accuracy, calibration, and manipulation metrics that
    help answer: "How good is this exchange at predicting outcomes?"
    """

    def __init__(self, min_sample_size: int = 10) -> None:
        """Initialize the quality scorer.

        Args:
            min_sample_size: Minimum number of resolved markets
                required to compute meaningful quality scores.
        """
        self.min_sample_size = min_sample_size

    def score(
        self,
        markets: list[Market],
        exchange: ExchangeName | None = None,
        category: str = "all",
    ) -> MarketQuality:
        """Compute quality scores for a set of resolved markets.

        Args:
            markets: List of resolved markets with known outcomes.
            exchange: If provided, only score markets from this exchange.
            category: Category label for the score (e.g. 'Politics').

        Returns:
            MarketQuality with computed metrics.

        Raises:
            InsufficientDataError: If fewer than min_sample_size
                resolved markets are available.
        """
        resolved = self._filter_resolved(markets, exchange, category)

        if len(resolved) < self.min_sample_size:
            raise InsufficientDataError(
                analysis="market quality scoring",
                required=self.min_sample_size,
                available=len(resolved),
            )

        brier = self._brier_score(resolved)
        calibration = self._calibration_error(resolved)
        resolution_hours = self._avg_resolution_time(resolved)
        manipulation = self._manipulation_score(resolved)
        vol_corr = self._volume_accuracy_correlation(resolved)

        target_exchange = exchange or (
            resolved[0].exchange if resolved else ExchangeName.POLYMARKET
        )

        return MarketQuality(
            exchange=target_exchange,
            category=category,
            brier_score=brier,
            calibration_error=calibration,
            avg_resolution_hours=resolution_hours,
            manipulation_score=manipulation,
            volume_accuracy_correlation=vol_corr,
            sample_size=len(resolved),
            computed_at=datetime.now(UTC),
        )

    def _filter_resolved(
        self,
        markets: list[Market],
        exchange: ExchangeName | None,
        category: str,
    ) -> list[Market]:
        """Filter to resolved markets matching criteria.

        Args:
            markets: All markets.
            exchange: Optional exchange filter.
            category: Category filter ('all' matches everything).

        Returns:
            Filtered list of resolved markets.
        """
        filtered = []
        for m in markets:
            if m.status != MarketStatus.RESOLVED:
                continue
            if m.resolution is None:
                continue
            if exchange and m.exchange != exchange:
                continue
            if category != "all" and m.category.lower() != category.lower():
                continue
            filtered.append(m)
        return filtered

    @staticmethod
    def _brier_score(markets: list[Market]) -> float:
        """Compute the Brier score across resolved markets.

        Brier score = mean of (forecast - outcome)^2 across all markets.
        Lower is better. Perfect = 0.0, worst = 1.0.

        For each market, the forecast is the YES price at the time of
        fetch, and the outcome is 1 if YES resolved, 0 otherwise.

        Args:
            markets: List of resolved markets.

        Returns:
            Brier score in [0, 1].
        """
        if not markets:
            return 1.0

        total = 0.0
        for market in markets:
            yes_price = market.yes_price or 0.5
            outcome = (
                1.0
                if market.resolution
                and market.resolution.lower() in ("yes", "y", "true", "1")
                else 0.0
            )
            total += (yes_price - outcome) ** 2

        return total / len(markets)

    @staticmethod
    def _calibration_error(markets: list[Market], n_bins: int = 10) -> float:
        """Compute mean absolute calibration error.

        Groups markets by predicted probability into bins, then measures
        how closely the observed frequency matches the predicted probability
        within each bin.

        Args:
            markets: List of resolved markets.
            n_bins: Number of bins for calibration analysis.

        Returns:
            Mean absolute calibration error in [0, 1].
        """
        if not markets:
            return 1.0

        bins: dict[int, list[tuple[float, float]]] = {i: [] for i in range(n_bins)}

        for market in markets:
            prob = market.yes_price or 0.5
            outcome = (
                1.0
                if market.resolution
                and market.resolution.lower() in ("yes", "y", "true", "1")
                else 0.0
            )
            bin_idx = min(int(prob * n_bins), n_bins - 1)
            bins[bin_idx].append((prob, outcome))

        total_error = 0.0
        populated_bins = 0

        for bin_entries in bins.values():
            if not bin_entries:
                continue
            avg_pred = sum(p for p, _ in bin_entries) / len(bin_entries)
            avg_outcome = sum(o for _, o in bin_entries) / len(bin_entries)
            total_error += abs(avg_pred - avg_outcome)
            populated_bins += 1

        if populated_bins == 0:
            return 1.0

        return total_error / populated_bins

    @staticmethod
    def _avg_resolution_time(markets: list[Market]) -> float:
        """Compute average resolution time in hours.

        Measures the delay between market close and resolution.

        Args:
            markets: List of resolved markets.

        Returns:
            Average resolution time in hours.
        """
        durations: list[float] = []

        for market in markets:
            if market.resolved_at and market.closes_at:
                delta = market.resolved_at - market.closes_at
                hours = delta.total_seconds() / 3600.0
                if hours >= 0:
                    durations.append(hours)

        if not durations:
            return 0.0

        return sum(durations) / len(durations)

    @staticmethod
    def _manipulation_score(markets: list[Market]) -> float:
        """Estimate manipulation severity based on price patterns.

        Looks for signs of manipulation:
        - Extreme prices (very close to 0 or 1) that resolved opposite
        - Sudden volume spikes near resolution

        This is a heuristic and should not be treated as definitive.

        Args:
            markets: List of resolved markets.

        Returns:
            Score in [0, 1] where 0 = no manipulation detected.
        """
        if not markets:
            return 0.0

        suspicious = 0
        for market in markets:
            yes_price = market.yes_price or 0.5
            resolved_yes = market.resolution and market.resolution.lower() in (
                "yes",
                "y",
                "true",
                "1",
            )

            # Flag extreme confidence that turned out wrong
            if (yes_price > 0.90 and not resolved_yes) or (
                yes_price < 0.10 and resolved_yes
            ):
                suspicious += 1

        return min(1.0, suspicious / max(len(markets), 1))

    @staticmethod
    def _volume_accuracy_correlation(markets: list[Market]) -> float:
        """Compute correlation between volume and prediction accuracy.

        Higher volume should correlate with better predictions (lower
        Brier scores) if markets aggregate information efficiently.

        Args:
            markets: List of resolved markets.

        Returns:
            Pearson correlation in [-1, 1].
        """
        if len(markets) < 3:
            return 0.0

        volumes: list[float] = []
        accuracies: list[float] = []

        for market in markets:
            yes_price = market.yes_price or 0.5
            outcome = (
                1.0
                if market.resolution
                and market.resolution.lower() in ("yes", "y", "true", "1")
                else 0.0
            )
            error = (yes_price - outcome) ** 2
            accuracy = 1.0 - error

            volumes.append(market.volume_total)
            accuracies.append(accuracy)

        n = len(volumes)
        mean_v = sum(volumes) / n
        mean_a = sum(accuracies) / n

        cov = (
            sum(
                (v - mean_v) * (a - mean_a)
                for v, a in zip(volumes, accuracies, strict=True)
            )
            / n
        )

        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in volumes) / n)
        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in accuracies) / n)

        if std_v == 0 or std_a == 0:
            return 0.0

        return max(-1.0, min(1.0, cov / (std_v * std_a)))
