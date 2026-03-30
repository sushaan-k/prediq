"""Shared fixtures for arbiter tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arbiter.models import (
    ContractType,
    ExchangeName,
    Market,
    MarketPair,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    Outcome,
)


@pytest.fixture()
def now() -> datetime:
    """Current UTC timestamp for deterministic tests."""
    return datetime(2026, 3, 15, 14, 0, 0, tzinfo=UTC)


@pytest.fixture()
def binary_market_polymarket(now: datetime) -> Market:
    """A binary market from Polymarket."""
    return Market(
        id="pm-btc-200k",
        exchange=ExchangeName.POLYMARKET,
        title="Will Bitcoin hit $200K by Dec 2026?",
        description="Resolves Yes if BTC price reaches $200,000.",
        category="Crypto",
        contract_type=ContractType.BINARY,
        status=MarketStatus.ACTIVE,
        outcomes=[
            Outcome(name="Yes", price=0.35, volume=100_000.0),
            Outcome(name="No", price=0.68, volume=80_000.0),
        ],
        url="https://polymarket.com/event/btc-200k",
        volume_total=500_000.0,
        created_at=now,
        fetched_at=now,
    )


@pytest.fixture()
def binary_market_kalshi(now: datetime) -> Market:
    """A binary market from Kalshi for the same event."""
    return Market(
        id="KXBTC200K",
        exchange=ExchangeName.KALSHI,
        title="Bitcoin to reach $200,000 by end of 2026?",
        description="Settles if BTC trades at or above $200,000.",
        category="Crypto",
        contract_type=ContractType.BINARY,
        status=MarketStatus.ACTIVE,
        outcomes=[
            Outcome(name="Yes", price=0.30, volume=50_000.0),
            Outcome(name="No", price=0.72, volume=45_000.0),
        ],
        url="https://kalshi.com/markets/KXBTC200K",
        volume_total=200_000.0,
        created_at=now,
        fetched_at=now,
    )


@pytest.fixture()
def multi_outcome_market(now: datetime) -> Market:
    """A multi-outcome market."""
    return Market(
        id="pm-election-2028",
        exchange=ExchangeName.POLYMARKET,
        title="2028 Presidential Election Winner",
        category="Politics",
        contract_type=ContractType.MULTI_OUTCOME,
        status=MarketStatus.ACTIVE,
        outcomes=[
            Outcome(name="DeSantis", price=0.22, volume=200_000.0),
            Outcome(name="Newsom", price=0.31, volume=180_000.0),
            Outcome(name="Harris", price=0.15, volume=150_000.0),
            Outcome(name="Trump Jr", price=0.08, volume=100_000.0),
            Outcome(name="Other", price=0.28, volume=90_000.0),
        ],
        url="https://polymarket.com/event/election-2028",
        volume_total=1_000_000.0,
        created_at=now,
        fetched_at=now,
    )


@pytest.fixture()
def resolved_markets(now: datetime) -> list[Market]:
    """A collection of resolved markets for quality scoring."""
    markets = []
    # Well-calibrated set: 10 markets with varying prices and resolutions
    test_data = [
        ("Will event A happen?", 0.80, "yes"),
        ("Will event B happen?", 0.20, "no"),
        ("Will event C happen?", 0.90, "yes"),
        ("Will event D happen?", 0.10, "no"),
        ("Will event E happen?", 0.60, "yes"),
        ("Will event F happen?", 0.40, "no"),
        ("Will event G happen?", 0.70, "yes"),
        ("Will event H happen?", 0.30, "no"),
        ("Will event I happen?", 0.55, "yes"),
        ("Will event J happen?", 0.50, "no"),
        ("Will event K happen?", 0.85, "yes"),
        ("Will event L happen?", 0.15, "no"),
    ]
    for i, (title, price, resolution) in enumerate(test_data):
        markets.append(
            Market(
                id=f"resolved-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=title,
                category="Politics",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=price, volume=float(10_000 * (i + 1))),
                    Outcome(
                        name="No", price=1.0 - price, volume=float(8_000 * (i + 1))
                    ),
                ],
                volume_total=float(50_000 * (i + 1)),
                resolution=resolution,
                created_at=now,
                closes_at=now,
                resolved_at=now,
                fetched_at=now,
            )
        )
    return markets


@pytest.fixture()
def order_book() -> OrderBook:
    """A realistic order book with multiple levels."""
    return OrderBook(
        bids=[
            OrderBookLevel(price=0.72, quantity=5_000.0),
            OrderBookLevel(price=0.71, quantity=8_000.0),
            OrderBookLevel(price=0.70, quantity=12_000.0),
            OrderBookLevel(price=0.68, quantity=20_000.0),
            OrderBookLevel(price=0.65, quantity=30_000.0),
        ],
        asks=[
            OrderBookLevel(price=0.74, quantity=4_000.0),
            OrderBookLevel(price=0.75, quantity=7_000.0),
            OrderBookLevel(price=0.76, quantity=10_000.0),
            OrderBookLevel(price=0.78, quantity=15_000.0),
            OrderBookLevel(price=0.80, quantity=25_000.0),
        ],
        timestamp=datetime(2026, 3, 15, 14, 0, 0, tzinfo=UTC),
    )


@pytest.fixture()
def market_pair(
    binary_market_polymarket: Market,
    binary_market_kalshi: Market,
) -> MarketPair:
    """A matched pair of markets."""
    return MarketPair(
        market_a=binary_market_polymarket,
        market_b=binary_market_kalshi,
        similarity_score=0.85,
    )
