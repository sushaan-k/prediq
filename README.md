# prediq

[![CI](https://github.com/sushaan-k/prediq/actions/workflows/ci.yml/badge.svg)](https://github.com/sushaan-k/prediq/actions)
[![PyPI](https://img.shields.io/pypi/v/prediq.svg)](https://pypi.org/project/prediq/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/prediq.svg)](https://pypi.org/project/prediq/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Cross-exchange prediction market analytics and arbitrage detection.**

`prediq` ingests live order books from Kalshi, Polymarket, and Manifold, normalizes them into a unified probability space, detects arbitrage across exchanges and correlated markets, and surfaces trading signals with Bayesian-updated forecasts.

---

## The Problem

Prediction markets are fragmented. The same question trades at different probabilities on different platforms simultaneously. Correlated markets (e.g., "Fed raises rates" and "NVDA Q3 miss") are priced independently when they should be coupled. No open tool unifies this, normalizes it, and finds the edges in real time.

## Solution

```python
from arbiter import MarketAggregator, ArbitrageScanner, ForecastEngine

agg = MarketAggregator(exchanges=["kalshi", "polymarket", "manifold"])
snapshot = await agg.snapshot()

# Find markets where cross-exchange spread exceeds transaction costs
scanner = ArbitrageScanner(min_edge=0.02, min_liquidity=500)
arbs = scanner.scan(snapshot)

for arb in arbs:
    print(f"{arb.question[:60]}")
    print(f"  Buy {arb.buy_exchange} @ {arb.buy_price:.2%}")
    print(f"  Sell {arb.sell_exchange} @ {arb.sell_price:.2%}")
    print(f"  Edge: {arb.edge:.2%} | Capacity: ${arb.capacity:,.0f}")

# Bayesian forecast engine with resolution data for calibration
engine = ForecastEngine()
forecast = engine.calibrated_probability("Will CPI exceed 3.5% in May 2026?")
```

## At a Glance

- **Unified probability space** — normalizes YES/NO, multiple-choice, and range markets
- **Cross-exchange arbitrage** — detects mispricings after bid/ask and fee adjustment
- **Correlation graph** — links related markets (macro, sector, policy) for correlated bet detection
- **Bayesian calibration** — uses historical resolution data to correct for market overconfidence/underconfidence
- **Live WebSocket feeds** — low-latency streaming from all supported exchanges

## Install

```bash
pip install prediq
```

## Supported Exchanges

| Exchange | Order Book | Historical | WebSocket |
|---|---|---|---|
| Kalshi | ✅ | ✅ | ✅ |
| Polymarket | ✅ | ✅ | ✅ |
| Manifold | ✅ | ✅ | — |

## Architecture

```
MarketAggregator
 ├── KalshiClient / PolymarketClient / ManifoldClient
 ├── Normalizer        # unified probability + liquidity representation
 ├── ArbitrageScanner  # cross-exchange spread detection
 ├── CorrelationGraph  # related market clustering
 └── ForecastEngine    # Bayesian-calibrated probability estimates
```

## Contributing

PRs welcome. Run `pip install -e ".[dev]"` then `pytest`. Star the repo if you find it useful ⭐
