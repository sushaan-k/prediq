"""Market quality analysis example.

Fetches resolved markets from an exchange and computes quality metrics
including Brier score (accuracy), calibration error, and manipulation
detection heuristics.

Usage:
    python examples/market_quality.py
"""

from __future__ import annotations

import asyncio

from arbiter import Arbiter, Exchange
from arbiter.analytics.quality import QualityScorer
from arbiter.analytics.violations import ViolationDetector
from arbiter.exceptions import InsufficientDataError


async def main() -> None:
    """Analyze market quality and detect probability violations."""
    async with Arbiter(
        exchanges=[Exchange.manifold()],
    ) as arb:
        print("Fetching markets (including resolved)...")
        all_markets = await arb.fetch_all_markets(active_only=False, limit=100)

        total = sum(len(ms) for ms in all_markets.values())
        print(f"  Total markets fetched: {total}")

        # Flatten for analysis
        flat_markets = [m for ms in all_markets.values() for m in ms]

        # --- Probability violations ---
        print("\nScanning for probability violations...")
        detector = ViolationDetector(binary_tolerance=0.02, multi_tolerance=0.03)
        binary_violations, multi_violations = detector.detect_all(flat_markets)

        if binary_violations:
            print(f"\n  Binary violations found: {len(binary_violations)}")
            for v in binary_violations[:5]:
                print(
                    f"    {v.market[:50]}: "
                    f"YES={v.yes_price:.3f} + NO={v.no_price:.3f} "
                    f"= {v.price_sum:.3f} (arb: {v.implied_arb:.3f})"
                )
        else:
            print("  No binary violations found.")

        if multi_violations:
            print(f"\n  Multi-outcome violations: {len(multi_violations)}")
            for v in multi_violations[:5]:
                print(
                    f"    {v.market[:50]}: "
                    f"sum={v.price_sum:.3f} (deviation: {v.deviation:.3f})"
                )
        else:
            print("  No multi-outcome violations found.")

        # --- Quality scoring ---
        print("\nComputing quality scores...")
        scorer = QualityScorer(min_sample_size=5)

        try:
            quality = scorer.score(flat_markets, category="all")
            print(f"\n  Brier score:            {quality.brier_score:.4f}")
            print(f"  Calibration error:      {quality.calibration_error:.4f}")
            print(f"  Manipulation score:     {quality.manipulation_score:.4f}")
            print(
                f"  Vol-accuracy corr:      {quality.volume_accuracy_correlation:.4f}"
            )
            print(f"  Sample size:            {quality.sample_size}")
        except InsufficientDataError as e:
            print(f"  {e}")
            print("  (Need more resolved markets for quality scoring)")


if __name__ == "__main__":
    asyncio.run(main())
