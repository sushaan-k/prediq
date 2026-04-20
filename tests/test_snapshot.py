"""Tests for portable market snapshots and offline replay analysis."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from arbiter.models import ExchangeName, Market, Outcome
from arbiter.output.snapshot import MarketSnapshot, analyze_snapshot


def _market(
    market_id: str,
    exchange: ExchangeName,
    yes_price: float,
    no_price: float,
    volume_total: float,
) -> Market:
    return Market(
        id=market_id,
        exchange=exchange,
        title="Will Bitcoin hit 200K in 2026?",
        outcomes=[
            Outcome(name="Yes", price=yes_price, volume=volume_total / 2),
            Outcome(name="No", price=no_price, volume=volume_total / 2),
        ],
        volume_total=volume_total,
        fetched_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC),
    )


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot.from_markets(
        {
            "polymarket": [
                _market("pm-btc", ExchangeName.POLYMARKET, 0.65, 0.44, 80_000)
            ],
            "kalshi": [_market("kx-btc", ExchangeName.KALSHI, 0.48, 0.52, 20_000)],
        },
        generated_at="2026-05-22T12:00:00+00:00",
    )


def test_snapshot_round_trips_json(tmp_path) -> None:
    path = tmp_path / "markets.snapshot.json"
    written = _snapshot().write(path)

    loaded = MarketSnapshot.from_file(written)

    assert loaded.market_count == 2
    assert loaded.exchange_counts == {"polymarket": 1, "kalshi": 1}
    assert loaded.markets["polymarket"][0].yes_price == pytest.approx(0.65)


def test_analyze_snapshot_replays_without_exchange_clients() -> None:
    analysis = analyze_snapshot(
        _snapshot(),
        min_spread=0.05,
        min_disagreement=0.05,
    )

    assert analysis.market_count == 2
    assert analysis.pair_count == 1
    assert len(analysis.divergences) == 1
    assert analysis.divergences[0].spread == pytest.approx(0.17)
    assert len(analysis.binary_violations) == 1
    assert analysis.consensus[0]["event"] == "Will Bitcoin hit 200K in 2026?"


def test_snapshot_analysis_serializes_and_renders_markdown() -> None:
    analysis = analyze_snapshot(_snapshot(), min_spread=0.05)

    payload = json.loads(analysis.model_dump_json())
    markdown = analysis.to_markdown()

    assert payload["pair_count"] == 1
    assert "Top Divergences" in markdown
    assert "Will Bitcoin hit 200K" in markdown
