"""Research dataset export example.

Fetches prediction market data and exports it to Parquet format
for academic research and quantitative analysis.

Usage:
    python examples/research_export.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from arbiter import Arbiter, Exchange
from arbiter.output.export import DataExporter


async def main() -> None:
    """Export prediction market data to research-ready formats."""
    output_dir = Path("research_data")
    output_dir.mkdir(exist_ok=True)

    async with Arbiter(
        exchanges=[Exchange.manifold()],
        storage_path=str(output_dir / "arbiter.duckdb"),
    ) as arb:
        print("Fetching markets from all exchanges...")
        all_markets = await arb.fetch_all_markets(active_only=False, limit=100)

        flat_markets = [m for ms in all_markets.values() for m in ms]
        print(f"  Total markets: {len(flat_markets)}")

        # Store in DuckDB
        print("\nStoring in DuckDB...")
        arb._storage.connect()
        inserted = arb._storage.insert_markets(flat_markets)
        print(f"  Inserted {inserted} market snapshots")

        # Export to Parquet
        exporter = DataExporter()
        parquet_path = output_dir / "markets.parquet"
        exporter.export_markets_to_parquet(flat_markets, parquet_path)
        print(f"\nExported to {parquet_path}")
        print(f"  File size: {parquet_path.stat().st_size / 1024:.1f} KB")

        # Export to CSV
        csv_path = output_dir / "markets.csv"
        exporter.export_markets_to_csv(flat_markets, csv_path)
        print(f"Exported to {csv_path}")
        print(f"  File size: {csv_path.stat().st_size / 1024:.1f} KB")

        # Export from DuckDB directly
        db_parquet = output_dir / "markets_from_db.parquet"
        arb._storage.export_to_parquet("markets", db_parquet)
        print(f"Exported from DuckDB to {db_parquet}")

        # Query example
        print("\nSample DuckDB query -- top 5 markets by volume:")
        results = arb._storage.query_markets(limit=5)
        for r in results:
            print(
                f"  [{r['exchange']}] {r['title'][:50]} "
                f"(vol: ${r['volume_total']:,.0f})"
            )

        print(f"\nAll outputs saved to {output_dir.resolve()}/")


if __name__ == "__main__":
    asyncio.run(main())
