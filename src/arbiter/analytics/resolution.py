"""Market resolution tracking and accuracy metrics.

Tracks which markets have resolved, computes per-exchange Brier scores,
and provides calibration analysis to evaluate exchange forecasting accuracy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from arbiter.models import ExchangeName, Market, MarketStatus


@dataclass(frozen=True)
class ResolutionRecord:
    """Record of a single market resolution."""

    market_id: str
    exchange: ExchangeName
    title: str
    forecast: float
    outcome: float
    resolved_at: datetime


@dataclass(frozen=True)
class ExchangeAccuracy:
    """Accuracy metrics for a single exchange."""

    exchange: ExchangeName
    brier_score: float
    sample_size: int
    calibration_buckets: dict[str, float]


@dataclass
class ResolutionTracker:
    """Tracks market resolutions and computes accuracy metrics.

    Collects resolved markets, records the final forecast price and
    actual outcome, then computes per-exchange Brier scores and
    calibration analysis.
    """

    _records: list[ResolutionRecord] = field(default_factory=list)
    _seen_ids: set[str] = field(default_factory=set)

    def record(self, market: Market) -> bool:
        """Record a resolved market.

        Args:
            market: A resolved market with a known outcome.

        Returns:
            True if the market was newly recorded, False if already tracked
            or if the market is not resolved.
        """
        if market.status != MarketStatus.RESOLVED:
            return False

        if market.resolution is None:
            return False

        composite_id = f"{market.exchange.value}:{market.id}"
        if composite_id in self._seen_ids:
            return False

        yes_price = market.yes_price
        forecast = yes_price if yes_price is not None else 0.5
        outcome = 1.0 if market.resolution.lower() in ("yes", "y", "true", "1") else 0.0

        resolved_at = market.resolved_at or datetime.now(UTC)

        self._records.append(
            ResolutionRecord(
                market_id=market.id,
                exchange=market.exchange,
                title=market.title,
                forecast=forecast,
                outcome=outcome,
                resolved_at=resolved_at,
            )
        )
        self._seen_ids.add(composite_id)
        return True

    def record_batch(self, markets: list[Market]) -> int:
        """Record a batch of markets, returning the count of newly added.

        Args:
            markets: List of markets (resolved ones will be recorded).

        Returns:
            Number of newly recorded resolutions.
        """
        return sum(1 for m in markets if self.record(m))

    def brier_score(self, exchange: ExchangeName | None = None) -> float:
        """Compute the Brier score for an exchange or all exchanges.

        Brier score = mean of (forecast - outcome)^2.
        Lower is better: 0.0 = perfect, 1.0 = worst.

        Args:
            exchange: Filter to a specific exchange, or None for all.

        Returns:
            Brier score, or 1.0 if no records exist.
        """
        records = self._filter(exchange)
        if not records:
            return 1.0

        total = sum((r.forecast - r.outcome) ** 2 for r in records)
        return total / len(records)

    def calibration(
        self,
        exchange: ExchangeName | None = None,
        n_buckets: int = 5,
    ) -> dict[str, float]:
        """Compute calibration by probability bucket.

        Groups forecasts into buckets (e.g., 0-20%, 20-40%, ...) and
        computes the observed frequency in each bucket. Perfect
        calibration means the observed frequency matches the bucket
        midpoint.

        Args:
            exchange: Filter to a specific exchange, or None for all.
            n_buckets: Number of calibration buckets.

        Returns:
            Dictionary mapping bucket labels to observed frequencies.
        """
        records = self._filter(exchange)
        if not records:
            return {}

        buckets: dict[int, list[float]] = defaultdict(list)
        bucket_size = 1.0 / n_buckets

        for r in records:
            idx = min(int(r.forecast / bucket_size), n_buckets - 1)
            buckets[idx].append(r.outcome)

        result: dict[str, float] = {}
        for i in range(n_buckets):
            low = i * bucket_size
            high = (i + 1) * bucket_size
            label = f"{low:.0%}-{high:.0%}"
            outcomes = buckets.get(i, [])
            if outcomes:
                result[label] = sum(outcomes) / len(outcomes)
            else:
                result[label] = float("nan")

        return result

    def accuracy_by_exchange(self) -> list[ExchangeAccuracy]:
        """Compute accuracy metrics for each exchange with recorded data.

        Returns:
            List of ExchangeAccuracy objects, one per exchange.
        """
        exchanges: set[ExchangeName] = {r.exchange for r in self._records}
        results: list[ExchangeAccuracy] = []

        for ex in sorted(exchanges, key=lambda e: e.value):
            records = self._filter(ex)
            brier = self.brier_score(ex)
            cal = self.calibration(ex)
            results.append(
                ExchangeAccuracy(
                    exchange=ex,
                    brier_score=brier,
                    sample_size=len(records),
                    calibration_buckets=cal,
                )
            )

        return results

    @property
    def total_records(self) -> int:
        """Total number of resolution records."""
        return len(self._records)

    def _filter(self, exchange: ExchangeName | None) -> list[ResolutionRecord]:
        """Filter records by exchange."""
        if exchange is None:
            return self._records
        return [r for r in self._records if r.exchange == exchange]
