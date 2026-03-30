# arbiter

## Cross-Exchange Prediction Market Analytics Engine

### The Problem

Prediction markets are exploding. Polymarket did billions in volume in 2025. Kalshi is SEC-regulated and growing fast. New exchanges are launching monthly. Academic research from 2025 documented **$40 million in arbitrage profits extracted from Polymarket alone** across 86 million bets.

But the analytics infrastructure is nonexistent:
- There's no unified API to query multiple prediction markets simultaneously
- No tools for cross-exchange price comparison or inefficiency detection
- No historical analysis of pricing efficiency over time
- No liquidity modeling or execution cost estimation
- Research papers are being written about prediction market microstructure, but the data tools to reproduce their findings don't exist publicly

The trading bots exist (you've built one). What's missing is the **analytics layer** — the telescope, not the gun.

### The Solution

`arbiter` is a real-time analytics platform that ingests data from Polymarket, Kalshi, and other prediction markets, normalizes it, detects cross-exchange pricing inefficiencies, models liquidity, and provides backtestable historical data.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        arbiter                            │
│                                                           │
│  ┌──────────────────────────────────────────────────┐     │
│  │               Exchange Connectors                  │     │
│  │                                                   │     │
│  │  ┌───────────┐ ┌──────────┐ ┌──────────────────┐  │     │
│  │  │Polymarket │ │ Kalshi   │ │ Metaculus /       │  │     │
│  │  │           │ │          │ │ Manifold / others │  │     │
│  │  │ - REST    │ │ - REST   │ │                   │  │     │
│  │  │ - WS feed │ │ - WS feed│ │ - REST APIs       │  │     │
│  │  │ - CLOB    │ │ - Order  │ │ - Polling         │  │     │
│  │  │   data    │ │   book   │ │                   │  │     │
│  │  └─────┬─────┘ └────┬────┘ └────────┬──────────┘  │     │
│  │        └─────────────┴──────────────┘              │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │            Normalization Engine                     │     │
│  │                                                   │     │
│  │  - Match equivalent markets across exchanges       │     │
│  │    (fuzzy + semantic matching for event titles)    │     │
│  │  - Normalize pricing (some use 0-1, some cents)   │     │
│  │  - Align timestamps across exchange clocks         │     │
│  │  - Handle different contract structures             │     │
│  │    (binary, multi-outcome, ranged)                 │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Analytics Engines                      │     │
│  │                                                   │     │
│  │  ┌──────────────────┐  ┌───────────────────────┐  │     │
│  │  │ Price Divergence  │  │ Liquidity Analyzer    │  │     │
│  │  │ Detector          │  │                       │  │     │
│  │  │                   │  │ - Order book depth    │  │     │
│  │  │ - Cross-exchange  │  │ - Spread analysis     │  │     │
│  │  │   spread tracking │  │ - Impact estimation   │  │     │
│  │  │ - YES+NO pricing  │  │ - Execution cost      │  │     │
│  │  │   anomalies       │  │   modeling             │  │     │
│  │  │ - Multi-outcome   │  │ - Slippage curves     │  │     │
│  │  │   probability     │  │                       │  │     │
│  │  │   violations      │  │                       │  │     │
│  │  └──────────────────┘  └───────────────────────┘  │     │
│  │                                                   │     │
│  │  ┌──────────────────┐  ┌───────────────────────┐  │     │
│  │  │ Efficiency        │  │ Market Quality        │  │     │
│  │  │ Metrics           │  │ Scorer                │  │     │
│  │  │                   │  │                       │  │     │
│  │  │ - Price discovery │  │ - Brier score         │  │     │
│  │  │   speed           │  │   (calibration)       │  │     │
│  │  │ - Arbitrage       │  │ - Resolution speed    │  │     │
│  │  │   window duration │  │ - Volume correlation  │  │     │
│  │  │ - Information     │  │   with accuracy       │  │     │
│  │  │   incorporation   │  │ - Manipulation        │  │     │
│  │  │   rate             │  │   detection           │  │     │
│  │  └──────────────────┘  └───────────────────────┘  │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Output Layer                          │     │
│  │                                                   │     │
│  │  - Real-time dashboard (web UI)                    │     │
│  │  - REST API for programmatic access                │     │
│  │  - Webhook alerts (price divergence > threshold)   │     │
│  │  - Historical data export (Parquet / CSV)          │     │
│  │  - Research-ready datasets                         │     │
│  └──────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Core Analytics

#### 1. Cross-Exchange Price Divergence

The simplest and most valuable metric. When the same event is priced differently on two exchanges:

```python
# Example output
Divergence(
    event="US Presidential Election 2028 — Republican Nominee",
    outcome="Ron DeSantis",
    polymarket_price=0.23,
    kalshi_price=0.19,
    spread=0.04,                    # 4 cents
    spread_pct=17.4%,               # relative to mid
    polymarket_liquidity=125000,    # $ available at this price
    kalshi_liquidity=43000,
    net_arb_profit_estimate=1200,   # after fees and slippage
    window_opened="2026-03-15T14:23:00Z",
    window_duration_so_far="2h 15m",
    historical_avg_window="45m",    # usually closes faster
)
```

#### 2. Probability Violation Detection

Markets sometimes misprice in ways that violate basic probability:

```python
# YES + NO should sum to ~$1.00 (minus fees)
# When they don't, there's a risk-free arbitrage
ProbabilityViolation(
    market="Will Bitcoin hit $200K by Dec 2026?",
    exchange="Polymarket",
    yes_price=0.35,
    no_price=0.68,
    sum=1.03,                       # Should be ≤1.00
    implied_arb=0.03,               # 3 cents risk-free per contract
    volume_available=50000,
)

# Multi-outcome violation: probabilities should sum to 1
MultiOutcomeViolation(
    market="2028 Presidential Election Winner",
    exchange="Kalshi",
    outcomes={"DeSantis": 0.22, "Trump Jr": 0.08, "Newsom": 0.31,
              "Harris": 0.15, "Other": 0.28},
    sum=1.04,                       # 4% overpriced in aggregate
)
```

#### 3. Liquidity Analysis

```python
# Order book analysis
LiquidityProfile(
    market="Fed Rate Decision March 2026",
    exchange="Kalshi",
    best_bid=0.72,
    best_ask=0.74,
    spread=0.02,
    depth_at_1pct=15000,            # $ available within 1% of mid
    depth_at_5pct=85000,
    estimated_impact={               # How much does buying X move price?
        1000: 0.001,                 # $1K moves price 0.1 cents
        10000: 0.008,                # $10K moves price 0.8 cents
        100000: 0.035,               # $100K moves price 3.5 cents
    },
)
```

#### 4. Market Quality Scoring

```python
# How good is this market at predicting outcomes?
MarketQuality(
    exchange="Polymarket",
    category="US Politics",
    brier_score=0.18,               # Lower is better (perfect = 0)
    calibration_error=0.03,         # Are 70% events resolving 70% of the time?
    avg_resolution_time="2.3 days", # After event, how fast does market resolve?
    manipulation_score=0.12,        # 0 = no manipulation detected, 1 = severe
    volume_accuracy_correlation=0.67, # Higher volume → better accuracy?
)
```

### Technical Stack

- **Language**: Python 3.11+ (async throughout)
- **Exchange APIs**: `httpx` + `websockets` for real-time feeds
- **Matching**: `sentence-transformers` for semantic market matching across exchanges
- **Storage**: `duckdb` (analytical queries), `parquet` (historical data)
- **Dashboard**: `fastapi` + React frontend (or `streamlit` for v1)
- **Scheduling**: `apscheduler` for periodic data collection

### API Surface (Draft)

```python
from arbiter import Arbiter, Exchange

# Initialize with exchanges
arb = Arbiter(
    exchanges=[
        Exchange.polymarket(api_key="..."),
        Exchange.kalshi(api_key="..."),
        Exchange.metaculus(),           # No auth needed for public data
        Exchange.manifold(),
    ]
)

# Find current divergences
divergences = await arb.divergences(min_spread=0.02, min_liquidity=5000)
for d in divergences:
    print(f"{d.event}: {d.spread_pct:.1%} spread ({d.exchange_a} vs {d.exchange_b})")

# Probability violations
violations = await arb.violations()

# Liquidity analysis
liquidity = await arb.liquidity("polymarket", market_id="...")

# Historical efficiency
history = await arb.historical(
    category="US Politics",
    start="2025-01-01",
    end="2026-03-01",
    metrics=["brier_score", "calibration", "avg_spread"],
)
history.plot()

# Real-time monitoring
async for alert in arb.monitor(divergence_threshold=0.03):
    print(f"ALERT: {alert}")

# Export research dataset
await arb.export_dataset(
    "prediction_market_data_2025.parquet",
    exchanges=["polymarket", "kalshi"],
    include_orderbook_snapshots=True,
)
```

### Dashboard

Web UI showing:
- **Live divergence scanner** — real-time cross-exchange spread table
- **Market map** — visualization of all tracked markets, colored by efficiency
- **Historical charts** — price convergence patterns over time
- **Liquidity heatmap** — where is the money?
- **Leaderboard** — which exchange is most accurate by category?

### What Makes This Novel

1. **First open-source cross-exchange prediction market analytics platform** — nothing like this exists
2. **Semantic market matching** — automatically matches equivalent events across exchanges with different naming
3. **Liquidity modeling with impact estimation** — not just "what's the price" but "what does it cost to trade"
4. **Research-grade data export** — enables academic research on market efficiency
5. **Keeps your actual trading strategy private** — this is the analytics layer, not the execution layer

### Repo Structure

```
arbiter/
├── README.md
├── pyproject.toml
├── src/
│   └── arbiter/
│       ├── __init__.py
│       ├── exchanges/
│       │   ├── base.py             # Exchange interface
│       │   ├── polymarket.py       # Polymarket connector
│       │   ├── kalshi.py           # Kalshi connector
│       │   ├── metaculus.py        # Metaculus connector
│       │   └── manifold.py         # Manifold connector
│       ├── matching/
│       │   ├── semantic.py         # Semantic market matching
│       │   └── normalizer.py       # Price/contract normalization
│       ├── analytics/
│       │   ├── divergence.py       # Cross-exchange divergence detection
│       │   ├── violations.py       # Probability violation detection
│       │   ├── liquidity.py        # Liquidity analysis
│       │   ├── quality.py          # Market quality scoring
│       │   └── efficiency.py       # Efficiency metrics
│       ├── output/
│       │   ├── api.py              # REST API
│       │   ├── dashboard.py        # Web dashboard
│       │   ├── alerts.py           # Webhook alerts
│       │   └── export.py           # Data export
│       └── storage.py              # DuckDB storage layer
├── tests/
├── examples/
│   ├── divergence_scanner.py
│   ├── market_quality.py
│   └── research_export.py
├── dashboard/
│   └── ... (React frontend)
└── docs/
    ├── exchanges.md
    ├── analytics.md
    └── research.md
```

### Research References

- "Arbitrage Profits in Prediction Markets" (SSRN, 2025) — $40M in Polymarket arbitrage documented
- "AI Agents Are Quietly Rewriting Prediction Market Trading" (CoinDesk, March 2026)
- "Polymarket leads Kalshi in price discovery when liquidity is high" (SSRN, 2025)
- Kalshi API documentation (kalshi.com/docs)
- Polymarket API documentation (docs.polymarket.com)
- "The Wisdom of Crowds: Theory and Evidence" (foundational prediction market research)
