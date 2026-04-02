#!/usr/bin/env python3
"""Offline demo for arbiter."""

from __future__ import annotations

from arbiter.analytics.divergence import DivergenceDetector
from arbiter.models import (
    ContractType,
    ExchangeName,
    Market,
    MarketPair,
    Outcome,
)


def build_market(
    market_id: str,
    exchange: ExchangeName,
    yes_price: float,
    no_price: float,
) -> Market:
    return Market(
        id=market_id,
        exchange=exchange,
        title="Will the Federal Reserve cut rates by September 2026?",
        contract_type=ContractType.BINARY,
        outcomes=[
            Outcome(name="Yes", price=yes_price, volume=25_000),
            Outcome(name="No", price=no_price, volume=23_000),
        ],
        volume_total=500_000,
    )


def main() -> None:
    polymarket = build_market("poly-1", ExchangeName.POLYMARKET, 0.41, 0.59)
    manifold = build_market("mani-1", ExchangeName.MANIFOLD, 0.56, 0.44)

    detector = DivergenceDetector(min_spread=0.02)
    divergences = detector.detect(
        [MarketPair(market_a=polymarket, market_b=manifold, similarity_score=0.97)]
    )

    print("arbiter demo")
    print("markets compared: 2")
    print(f"divergences found: {len(divergences)}")
    if divergences:
        top = divergences[0]
        print(
            f"largest spread: {top.exchange_a.value} {top.price_a:.2f} vs "
            f"{top.exchange_b.value} {top.price_b:.2f} ({top.spread_pct:.1%})"
        )


if __name__ == "__main__":
    main()
