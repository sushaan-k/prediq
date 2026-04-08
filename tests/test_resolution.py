"""Tests for arbiter.analytics.resolution -- market resolution tracking."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from arbiter.analytics.resolution import ResolutionTracker
from arbiter.models import (
    ContractType,
    ExchangeName,
    Market,
    MarketStatus,
    Outcome,
)


def _resolved_market(
    market_id: str = "m-1",
    exchange: ExchangeName = ExchangeName.POLYMARKET,
    yes_price: float = 0.80,
    resolution: str = "yes",
    title: str = "Will event happen?",
) -> Market:
    """Create a resolved market for testing."""
    return Market(
        id=market_id,
        exchange=exchange,
        title=title,
        contract_type=ContractType.BINARY,
        status=MarketStatus.RESOLVED,
        outcomes=[
            Outcome(name="Yes", price=yes_price, volume=0.0),
            Outcome(name="No", price=1.0 - yes_price, volume=0.0),
        ],
        resolution=resolution,
        resolved_at=datetime(2026, 3, 15, tzinfo=UTC),
        volume_total=50_000.0,
        fetched_at=datetime(2026, 3, 15, tzinfo=UTC),
    )


class TestResolutionTracker:
    """Tests for ResolutionTracker."""

    def test_record_resolved_market(self) -> None:
        tracker = ResolutionTracker()
        market = _resolved_market()
        assert tracker.record(market) is True
        assert tracker.total_records == 1

    def test_skip_non_resolved_market(self) -> None:
        tracker = ResolutionTracker()
        market = Market(
            id="active-1",
            exchange=ExchangeName.POLYMARKET,
            title="Active market",
            status=MarketStatus.ACTIVE,
            outcomes=[
                Outcome(name="Yes", price=0.50, volume=0.0),
                Outcome(name="No", price=0.50, volume=0.0),
            ],
            fetched_at=datetime(2026, 3, 15, tzinfo=UTC),
        )
        assert tracker.record(market) is False
        assert tracker.total_records == 0

    def test_skip_duplicate_market(self) -> None:
        tracker = ResolutionTracker()
        market = _resolved_market()
        tracker.record(market)
        assert tracker.record(market) is False
        assert tracker.total_records == 1

    def test_same_id_different_exchange_not_duplicate(self) -> None:
        tracker = ResolutionTracker()
        m1 = _resolved_market(market_id="x", exchange=ExchangeName.POLYMARKET)
        m2 = _resolved_market(market_id="x", exchange=ExchangeName.KALSHI)
        tracker.record(m1)
        assert tracker.record(m2) is True
        assert tracker.total_records == 2

    def test_record_batch(self) -> None:
        tracker = ResolutionTracker()
        markets = [
            _resolved_market(market_id="m-1"),
            _resolved_market(market_id="m-2", yes_price=0.20, resolution="no"),
            # Non-resolved should be skipped
            Market(
                id="active",
                exchange=ExchangeName.POLYMARKET,
                title="Active",
                status=MarketStatus.ACTIVE,
                outcomes=[
                    Outcome(name="Yes", price=0.50, volume=0.0),
                    Outcome(name="No", price=0.50, volume=0.0),
                ],
                fetched_at=datetime(2026, 3, 15, tzinfo=UTC),
            ),
        ]
        count = tracker.record_batch(markets)
        assert count == 2
        assert tracker.total_records == 2

    def test_brier_score_perfect(self) -> None:
        tracker = ResolutionTracker()
        # Perfect forecast: 1.0 for yes, resolved yes
        tracker.record(_resolved_market(market_id="p1", yes_price=1.0, resolution="yes"))
        tracker.record(_resolved_market(market_id="p2", yes_price=0.0, resolution="no"))
        assert tracker.brier_score() == pytest.approx(0.0)

    def test_brier_score_worst(self) -> None:
        tracker = ResolutionTracker()
        # Completely wrong forecasts
        tracker.record(_resolved_market(market_id="w1", yes_price=1.0, resolution="no"))
        tracker.record(_resolved_market(market_id="w2", yes_price=0.0, resolution="yes"))
        assert tracker.brier_score() == pytest.approx(1.0)

    def test_brier_score_empty(self) -> None:
        tracker = ResolutionTracker()
        assert tracker.brier_score() == 1.0

    def test_brier_score_per_exchange(self) -> None:
        tracker = ResolutionTracker()
        # Polymarket: good forecast
        tracker.record(
            _resolved_market(
                market_id="pm-1",
                exchange=ExchangeName.POLYMARKET,
                yes_price=0.90,
                resolution="yes",
            )
        )
        # Kalshi: bad forecast
        tracker.record(
            _resolved_market(
                market_id="kx-1",
                exchange=ExchangeName.KALSHI,
                yes_price=0.90,
                resolution="no",
            )
        )
        pm_brier = tracker.brier_score(ExchangeName.POLYMARKET)
        kx_brier = tracker.brier_score(ExchangeName.KALSHI)
        assert pm_brier < kx_brier

    def test_calibration_basic(self) -> None:
        tracker = ResolutionTracker()
        # All forecasts in the 60-80% bucket, all resolve yes
        for i in range(5):
            tracker.record(
                _resolved_market(
                    market_id=f"cal-{i}",
                    yes_price=0.70,
                    resolution="yes",
                )
            )
        cal = tracker.calibration(n_buckets=5)
        assert len(cal) == 5
        # The 60%-80% bucket should show 100% resolution rate
        assert cal["60%-80%"] == pytest.approx(1.0)

    def test_calibration_empty(self) -> None:
        tracker = ResolutionTracker()
        assert tracker.calibration() == {}

    def test_calibration_nan_for_empty_bucket(self) -> None:
        tracker = ResolutionTracker()
        tracker.record(_resolved_market(market_id="c1", yes_price=0.10, resolution="no"))
        cal = tracker.calibration(n_buckets=5)
        # Only 0%-20% bucket has data; others should be nan
        assert not math.isnan(cal["0%-20%"])
        assert math.isnan(cal["40%-60%"])

    def test_accuracy_by_exchange(self) -> None:
        tracker = ResolutionTracker()
        tracker.record(
            _resolved_market(market_id="pm-1", exchange=ExchangeName.POLYMARKET)
        )
        tracker.record(
            _resolved_market(market_id="kx-1", exchange=ExchangeName.KALSHI)
        )
        results = tracker.accuracy_by_exchange()
        assert len(results) == 2
        exchanges = {r.exchange for r in results}
        assert ExchangeName.POLYMARKET in exchanges
        assert ExchangeName.KALSHI in exchanges
        for r in results:
            assert r.sample_size == 1
            assert 0.0 <= r.brier_score <= 1.0

    def test_skip_market_without_resolution_value(self) -> None:
        tracker = ResolutionTracker()
        market = Market(
            id="no-res",
            exchange=ExchangeName.POLYMARKET,
            title="Resolved but no resolution value",
            status=MarketStatus.RESOLVED,
            outcomes=[
                Outcome(name="Yes", price=0.50, volume=0.0),
                Outcome(name="No", price=0.50, volume=0.0),
            ],
            resolution=None,
            fetched_at=datetime(2026, 3, 15, tzinfo=UTC),
        )
        assert tracker.record(market) is False
