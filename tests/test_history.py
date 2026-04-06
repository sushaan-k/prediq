"""Tests for arbiter.analytics.history -- historical divergence tracking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from arbiter.analytics.history import (
    DivergenceHistory,
    TrendDirection,
)
from arbiter.models import Divergence, ExchangeName


def _make_divergence(
    event: str = "BTC $200K",
    outcome: str = "Yes",
    spread: float = 0.05,
    spread_pct: float = 0.15,
) -> Divergence:
    """Helper to create a Divergence with minimal boilerplate."""
    return Divergence(
        event=event,
        outcome=outcome,
        exchange_a=ExchangeName.POLYMARKET,
        exchange_b=ExchangeName.KALSHI,
        price_a=0.35,
        price_b=0.35 - spread,
        spread=spread,
        spread_pct=spread_pct,
    )


class TestDivergenceHistory:
    """Tests for DivergenceHistory tracking and trend analysis."""

    def test_track_records_snapshots(self) -> None:
        history = DivergenceHistory()
        divs = [_make_divergence(), _make_divergence(outcome="No", spread=0.03)]
        count = history.track(divs)
        assert count == 2
        assert history.total_snapshots == 2

    def test_track_with_explicit_timestamp(self) -> None:
        history = DivergenceHistory()
        ts = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        history.track([_make_divergence()], timestamp=ts)
        snaps = history.snapshots_for("BTC $200K", "Yes")
        assert len(snaps) == 1
        assert snaps[0].timestamp == ts

    def test_events_returns_tracked_keys(self) -> None:
        history = DivergenceHistory()
        history.track([
            _make_divergence(event="A", outcome="Yes"),
            _make_divergence(event="B", outcome="No"),
        ])
        keys = history.events()
        assert ("A", "Yes") in keys
        assert ("B", "No") in keys

    def test_trend_unknown_with_single_snapshot(self) -> None:
        history = DivergenceHistory()
        history.track([_make_divergence()])
        trend = history.trend("BTC $200K", "Yes")
        assert trend.direction == TrendDirection.UNKNOWN
        assert trend.snapshots == 1

    def test_trend_unknown_for_missing_event(self) -> None:
        history = DivergenceHistory()
        trend = history.trend("nonexistent", "Yes")
        assert trend.direction == TrendDirection.UNKNOWN
        assert trend.snapshots == 0

    def test_trend_growing(self) -> None:
        history = DivergenceHistory()
        base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        for i in range(5):
            ts = base + timedelta(hours=i)
            spread = 0.02 + 0.01 * i  # 0.02, 0.03, 0.04, 0.05, 0.06
            history.track([_make_divergence(spread=spread)], timestamp=ts)

        trend = history.trend("BTC $200K", "Yes")
        assert trend.direction == TrendDirection.GROWING
        assert trend.slope > 0
        assert trend.snapshots == 5

    def test_trend_shrinking(self) -> None:
        history = DivergenceHistory()
        base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        for i in range(5):
            ts = base + timedelta(hours=i)
            spread = 0.10 - 0.02 * i  # 0.10, 0.08, 0.06, 0.04, 0.02
            history.track([_make_divergence(spread=spread)], timestamp=ts)

        trend = history.trend("BTC $200K", "Yes")
        assert trend.direction == TrendDirection.SHRINKING
        assert trend.slope < 0

    def test_trend_stable(self) -> None:
        history = DivergenceHistory()
        base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        for i in range(5):
            ts = base + timedelta(hours=i)
            history.track([_make_divergence(spread=0.05)], timestamp=ts)

        trend = history.trend("BTC $200K", "Yes")
        assert trend.direction == TrendDirection.STABLE

    def test_snapshots_for_returns_copies(self) -> None:
        history = DivergenceHistory()
        history.track([_make_divergence()])
        snaps1 = history.snapshots_for("BTC $200K", "Yes")
        snaps2 = history.snapshots_for("BTC $200K", "Yes")
        # Should be equal but independent lists
        assert snaps1 == snaps2
        assert snaps1 is not snaps2

    def test_multiple_events_tracked_independently(self) -> None:
        history = DivergenceHistory()
        base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)

        for i in range(3):
            ts = base + timedelta(hours=i)
            history.track(
                [
                    _make_divergence(event="Alpha", spread=0.02 + 0.01 * i),
                    _make_divergence(event="Beta", spread=0.10 - 0.02 * i),
                ],
                timestamp=ts,
            )

        trend_a = history.trend("Alpha", "Yes")
        trend_b = history.trend("Beta", "Yes")
        assert trend_a.direction == TrendDirection.GROWING
        assert trend_b.direction == TrendDirection.SHRINKING

    def test_trend_first_last_seen(self) -> None:
        history = DivergenceHistory()
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 6, 1, tzinfo=UTC)
        history.track([_make_divergence(spread=0.02)], timestamp=t1)
        history.track([_make_divergence(spread=0.04)], timestamp=t2)

        trend = history.trend("BTC $200K", "Yes")
        assert trend.first_seen == t1
        assert trend.last_seen == t2
