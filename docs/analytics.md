# Analytics

`arbiter` is primarily an analytics engine. Once markets are normalized and
matched, the package computes several higher-level signals over them.

## Core Analyses

The current analytics surface centers on:

- cross-exchange price divergences
- probability violations
- liquidity profiles
- market quality scores
- efficiency metrics

Relevant modules live under `src/arbiter/analytics/`.

## Divergences

Divergence analysis asks whether the same event is priced differently across
two exchanges.

Programmatic example:

```python
divergences = await arb.divergences(min_spread=0.03)
```

Divergence records include fields such as:

- event
- outcome
- `exchange_a` / `exchange_b`
- prices on each venue
- spread and spread percentage
- liquidity estimates

Use this when searching for obvious cross-market inefficiencies or tracking how
often venues disagree.

## Probability Violations

`arbiter` can identify markets whose quoted prices violate basic probability
constraints.

```python
binary_violations, multi_violations = await arb.violations()
```

Examples:

- binary YES/NO sums materially above 1.0
- multi-outcome books whose total probability is inconsistent

These are useful for exchange-quality audits and possible arbitrage research.

## Liquidity

Liquidity analysis uses order-book data to estimate:

- best bid / ask
- spread
- depth near the mid price
- trade-size price impact

```python
profile = await arb.liquidity("polymarket", market_id="...")
```

This is the most execution-oriented analysis in the package.

## Quality Scoring

Quality scoring summarizes market quality for a venue or category using metrics
such as:

- calibration-style error
- Brier-style scoring inputs
- manipulation-oriented signals
- sample-size aware summaries

```python
quality = await arb.quality("polymarket", category="US Politics")
```

The exact score inputs depend on the available data for the chosen exchange and
market set.

## Efficiency Metrics

The efficiency layer estimates higher-level properties such as:

- divergence persistence windows
- price discovery speed
- information-ratio style summaries

This is the most research-oriented part of the current analytics stack and is
best used on historical or repeated snapshots rather than a single scan.

## API and CLI Surfaces

Most analytics are available through:

- Python `Arbiter` methods
- CLI commands such as `scan`, `violations`, and `export`
- API endpoints exposed by `arbiter serve`

## Practical Workflow

1. create exchange connectors
2. fetch markets
3. match equivalent markets across venues
4. run divergences and violations
5. inspect liquidity on the interesting markets
6. export snapshots for later research or dashboards

## Limits

- Market matching is heuristic, so false matches and misses are possible.
- Liquidity quality depends on exchange-side book depth quality.
- Research-grade analysis benefits from repeated snapshots or stored history, not
  just one live pull.
