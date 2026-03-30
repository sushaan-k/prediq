# Research Workflows

`arbiter` is useful as a research tool when you want a reproducible pipeline
from multi-exchange market data to analysis artifacts.

## What It Is Good At

The package is strongest for:

- cross-exchange snapshot collection
- normalized market comparison
- divergence studies
- exchange-quality analysis
- liquidity and execution-cost estimation
- export of structured datasets for offline work

It is not an execution engine. The library is designed to measure markets, not
to place trades.

## A Minimal Research Loop

```python
import asyncio
from arbiter import Arbiter, Exchange


async def main() -> None:
    async with Arbiter(
        exchanges=[
            Exchange.polymarket(),
            Exchange.kalshi(),
            Exchange.metaculus(),
            Exchange.manifold(),
        ]
    ) as arb:
        divergences = await arb.divergences(min_spread=0.03)
        binary_v, multi_v = await arb.violations()
        await arb.export_dataset("markets_snapshot.parquet")
        print(len(divergences), len(binary_v), len(multi_v))


asyncio.run(main())
```

## Useful Research Questions

- How often do exchanges disagree on the same event?
- Which venues tend to lead price discovery?
- How long do large divergences persist?
- Which categories show the poorest probability coherence?
- How much apparent spread survives realistic liquidity constraints?

## Data Export

The export path is the bridge from live analytics to notebooks and offline
analysis.

```bash
arbiter export markets.parquet
```

The exported dataset is a better fit than ad hoc API scraping when you want to:

- version snapshots
- run DuckDB or pandas analysis
- compare runs over time
- share a reproducible artifact with collaborators

## Monitoring and API Snapshots

For repeated collection you can use:

- `arbiter monitor(...)` in Python
- `arbiter serve` for an always-refreshing API snapshot

The serve path keeps updating divergences, violations, liquidity, and quality
state in the background so downstream dashboards can poll a stable interface.

## Reproducibility Advice

- fix connector configuration per run
- persist raw exports
- record timestamps and exchange sets
- keep the same matching thresholds across comparisons
- distinguish exploratory scans from publishable measurements

## Limits for Research Claims

- Exchange APIs can change or rate-limit.
- Matching equivalent markets is heuristic.
- A single snapshot is rarely enough for strong efficiency claims.
- Venue-specific microstructure differences can matter more than top-line prices.

For publishable or high-confidence work, build a time series from repeated
exports rather than drawing conclusions from one scan.
