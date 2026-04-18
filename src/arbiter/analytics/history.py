"""Historical divergence tracking.

Records divergence snapshots over time and computes trends to show
whether a particular cross-exchange spread is growing or shrinking.
This enables time-series analysis of market efficiency.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from arbiter.models import Divergence


@dataclass(frozen=True)
class DivergenceSnapshot:
    """A point-in-time record of a divergence measurement."""

    event: str
    outcome: str
    spread: float
    spread_pct: float
    timestamp: datetime


class TrendDirection:
    """Constants for divergence trend direction."""

    GROWING = "growing"
    SHRINKING = "shrinking"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class Trend:
    """Trend analysis for a single divergence over time."""

    event: str
    outcome: str
    direction: str
    slope: float
    snapshots: int
    first_seen: datetime
    last_seen: datetime


@dataclass
class DivergenceHistory:
    """Stores and analyses divergence snapshots over time.

    Each divergence is keyed by (event, outcome). Calling ``track()``
    records the current set of divergences with a timestamp.  The
    ``trend()`` method performs a simple linear regression on recorded
    spreads to indicate whether a divergence is growing or shrinking.
    """

    _snapshots: dict[tuple[str, str], list[DivergenceSnapshot]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _timestamps: list[datetime] = field(default_factory=list)

    def track(
        self,
        divergences: list[Divergence],
        timestamp: datetime | None = None,
    ) -> int:
        """Record a batch of divergences at a point in time.

        Args:
            divergences: Current divergences from the detector.
            timestamp: Override the recording timestamp (defaults to now).

        Returns:
            Number of snapshots recorded.
        """
        ts = timestamp or datetime.now(UTC)
        bisect.insort(self._timestamps, ts)

        count = 0
        for d in divergences:
            key = (d.event, d.outcome)
            snapshot = DivergenceSnapshot(
                event=d.event,
                outcome=d.outcome,
                spread=d.spread,
                spread_pct=d.spread_pct,
                timestamp=ts,
            )
            self._snapshots[key].append(snapshot)
            count += 1

        return count

    def trend(self, event: str, outcome: str) -> Trend:
        """Compute the trend for a specific divergence.

        Uses simple linear regression on the spread values over time.
        A positive slope means the divergence is growing; negative means
        it is shrinking.

        Args:
            event: The event name.
            outcome: The outcome label.

        Returns:
            A Trend object describing the direction and magnitude.
        """
        key = (event, outcome)
        snaps = self._snapshots.get(key, [])

        if len(snaps) < 2:
            direction = TrendDirection.UNKNOWN
            slope = 0.0
            first = snaps[0].timestamp if snaps else datetime.now(UTC)
            last = first
            return Trend(
                event=event,
                outcome=outcome,
                direction=direction,
                slope=slope,
                snapshots=len(snaps),
                first_seen=first,
                last_seen=last,
            )

        # Simple linear regression: y = spread, x = seconds from first snapshot
        t0 = snaps[0].timestamp.timestamp()
        xs = [s.timestamp.timestamp() - t0 for s in snaps]
        ys = [s.spread for s in snaps]

        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        ss_xx = sum((x - mean_x) ** 2 for x in xs)
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))

        slope = 0.0 if ss_xx == 0 else ss_xy / ss_xx

        # Classify direction based on slope magnitude
        threshold = 1e-9  # per-second threshold
        if slope > threshold:
            direction = TrendDirection.GROWING
        elif slope < -threshold:
            direction = TrendDirection.SHRINKING
        else:
            direction = TrendDirection.STABLE

        return Trend(
            event=event,
            outcome=outcome,
            direction=direction,
            slope=slope,
            snapshots=n,
            first_seen=snaps[0].timestamp,
            last_seen=snaps[-1].timestamp,
        )

    def events(self) -> list[tuple[str, str]]:
        """Return all tracked (event, outcome) keys."""
        return list(self._snapshots.keys())

    def snapshots_for(self, event: str, outcome: str) -> list[DivergenceSnapshot]:
        """Return all snapshots for a given divergence.

        Args:
            event: The event name.
            outcome: The outcome label.

        Returns:
            List of snapshots, ordered by recording time.
        """
        return list(self._snapshots.get((event, outcome), []))

    @property
    def total_snapshots(self) -> int:
        """Total number of individual snapshots stored."""
        return sum(len(v) for v in self._snapshots.values())
