"""Cross-exchange divergence scanner.

Connects to multiple prediction market exchanges and scans for
price divergences -- cases where the same event is priced differently
on two platforms, signaling potential arbitrage opportunities.

Usage:
    python examples/divergence_scanner.py
"""

from __future__ import annotations

import asyncio

from arbiter import Arbiter, Exchange


async def main() -> None:
    """Scan for cross-exchange price divergences."""
    # Connect to exchanges that don't require authentication
    async with Arbiter(
        exchanges=[
            Exchange.manifold(),
            Exchange.metaculus(),
        ],
        similarity_threshold=0.5,
    ) as arb:
        print("Fetching markets from all exchanges...")
        markets = await arb.fetch_all_markets(active_only=True, limit=50)

        for exchange_name, market_list in markets.items():
            print(f"  {exchange_name}: {len(market_list)} markets")

        print("\nMatching equivalent markets across exchanges...")
        pairs = arb.match_markets()
        print(f"  Found {len(pairs)} matched pairs")

        for pair in pairs[:5]:
            print(
                f"  - '{pair.market_a.title}' <-> '{pair.market_b.title}' "
                f"(similarity: {pair.similarity_score:.2f})"
            )

        print("\nScanning for divergences (min spread: 2%)...")
        divergences = await arb.divergences(min_spread=0.02)

        if not divergences:
            print("  No significant divergences found.")
            return

        print(f"\n{'=' * 70}")
        print(f"Found {len(divergences)} divergences:")
        print(f"{'=' * 70}")

        for d in divergences[:10]:
            print(f"\n  Event:     {d.event}")
            print(f"  Outcome:   {d.outcome}")
            print(
                f"  {d.exchange_a.value:>12}: {d.price_a:.3f}  vs  "
                f"{d.exchange_b.value}: {d.price_b:.3f}"
            )
            print(f"  Spread:    {d.spread:.3f} ({d.spread_pct:.1%})")

            if d.net_arb_profit_estimate > 0:
                print(f"  Est. profit: ${d.net_arb_profit_estimate:,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
