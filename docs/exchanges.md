# Exchanges

`arbiter` normalizes data from multiple prediction-market venues into a common
schema so the analytics layer can compare like with like.

## Supported Connectors

The current public factory surface is:

```python
from arbiter import Exchange

Exchange.polymarket(...)
Exchange.kalshi(...)
Exchange.metaculus(...)
Exchange.manifold(...)
```

These factories return connector instances used by `Arbiter`.

## Exchange Matrix

| Exchange | Factory | Auth | Notes |
|---|---|---|---|
| Polymarket | `Exchange.polymarket(...)` | optional key/secret | uses CLOB + Gamma metadata endpoints |
| Kalshi | `Exchange.kalshi(...)` | optional key/secret | order-book and market data support |
| Metaculus | `Exchange.metaculus(...)` | optional API token | forecast/question style market data |
| Manifold | `Exchange.manifold(...)` | optional API key | public REST-oriented market access |

## Connector Responsibilities

Each connector is responsible for:

- fetching market listings
- fetching individual markets
- fetching order-book or liquidity-like data where available
- normalizing raw exchange payloads into `arbiter.models`

The common model layer includes types such as:

- `Market`
- `OrderBook`
- `OrderBookLevel`
- `Outcome`
- `ExchangeName`

## Typical Usage

```python
import asyncio
from arbiter import Arbiter, Exchange


async def main() -> None:
    async with Arbiter(
        exchanges=[
            Exchange.polymarket(),
            Exchange.kalshi(api_key="...", api_secret="..."),
            Exchange.metaculus(),
        ]
    ) as arb:
        markets = await arb.fetch_all_markets(limit=25)
        print({name: len(items) for name, items in markets.items()})


asyncio.run(main())
```

## Normalization

Exchanges differ in:

- price units
- contract shapes
- order-book depth formats
- market naming conventions
- timestamps and lifecycle states

`arbiter` normalizes those differences before running analytics. That enables
divergence and violation analysis over a shared representation instead of raw
venue-specific JSON.

## Authentication

Authentication is exchange-specific:

- Polymarket: optional API credentials for authenticated endpoints
- Kalshi: optional API key and secret
- Metaculus: optional API token
- Manifold: optional API key

Public data paths remain usable without authentication where the exchange
permits it.

## Connector Selection Guidance

- Use `Polymarket` and `Kalshi` when order-book and liquidity analysis matters.
- Use `Metaculus` when you want forecast-oriented question coverage.
- Use `Manifold` for broad public-market sampling and comparison.

## Limits

- WebSocket support varies by exchange and by the paths exposed in the current
  connector implementations.
- Some exchanges expose richer liquidity data than others.
- Matching equivalent markets across venues is heuristic, not guaranteed.
