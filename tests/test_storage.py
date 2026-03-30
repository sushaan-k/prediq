"""Tests for arbiter.storage -- DuckDB storage layer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from arbiter.models import (
    ContractType,
    Divergence,
    ExchangeName,
    Market,
    MarketStatus,
    Outcome,
)
from arbiter.storage import Storage


class TestStorage:
    """Tests for the Storage class."""

    def test_in_memory_connect(self) -> None:
        storage = Storage()
        storage.connect()
        assert storage.conn is not None
        storage.close()

    def test_context_manager(self) -> None:
        with Storage() as storage:
            assert storage.conn is not None

    def test_insert_and_query_markets(self, now: datetime) -> None:
        markets = [
            Market(
                id=f"test-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Test market {i}",
                category="Test",
                contract_type=ContractType.BINARY,
                status=MarketStatus.ACTIVE,
                outcomes=[
                    Outcome(name="Yes", price=0.5 + i * 0.1, volume=0.0),
                    Outcome(name="No", price=0.5 - i * 0.1, volume=0.0),
                ],
                volume_total=float(10_000 * (i + 1)),
                fetched_at=now,
            )
            for i in range(3)
        ]

        with Storage() as storage:
            count = storage.insert_markets(markets)
            assert count == 3

            results = storage.query_markets(limit=10)
            assert len(results) == 3

    def test_query_markets_with_filter(self, now: datetime) -> None:
        markets = [
            Market(
                id="pm-1",
                exchange=ExchangeName.POLYMARKET,
                title="PM market",
                outcomes=[
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ],
                fetched_at=now,
            ),
            Market(
                id="kx-1",
                exchange=ExchangeName.KALSHI,
                title="Kalshi market",
                outcomes=[
                    Outcome(name="Yes", price=0.6, volume=0.0),
                    Outcome(name="No", price=0.4, volume=0.0),
                ],
                fetched_at=now,
            ),
        ]

        with Storage() as storage:
            storage.insert_markets(markets)

            pm_results = storage.query_markets(exchange="polymarket")
            assert len(pm_results) == 1
            assert pm_results[0]["exchange"] == "polymarket"

    def test_insert_and_query_divergences(self) -> None:
        divergences = [
            Divergence(
                event="Test Event",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.55,
                price_b=0.48,
                spread=0.07,
                spread_pct=0.136,
                market_a_id="pm-1",
                market_b_id="kx-1",
            )
        ]

        with Storage() as storage:
            count = storage.insert_divergences(divergences)
            assert count == 1

            results = storage.query_divergences(min_spread=0.05)
            assert len(results) == 1
            assert results[0]["spread"] == pytest.approx(0.07)

    def test_query_divergences_min_spread(self) -> None:
        divergences = [
            Divergence(
                event=f"Event {i}",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.50,
                price_b=0.50 - 0.02 * (i + 1),
                spread=0.02 * (i + 1),
                spread_pct=0.04 * (i + 1),
            )
            for i in range(5)
        ]

        with Storage() as storage:
            storage.insert_divergences(divergences)
            big = storage.query_divergences(min_spread=0.08)
            assert all(r["spread"] >= 0.08 for r in big)

    def test_insert_empty_lists(self) -> None:
        with Storage() as storage:
            assert storage.insert_markets([]) == 0
            assert storage.insert_divergences([]) == 0

    def test_export_to_parquet(self, tmp_path: Path, now: datetime) -> None:
        markets = [
            Market(
                id="export-test",
                exchange=ExchangeName.POLYMARKET,
                title="Export test",
                outcomes=[
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ],
                volume_total=10_000.0,
                fetched_at=now,
            )
        ]
        out_path = tmp_path / "test_export.parquet"

        with Storage() as storage:
            storage.insert_markets(markets)
            result = storage.export_to_parquet("markets", out_path)
            assert result.exists()
            assert result.suffix == ".parquet"

    def test_file_backed_storage(self, tmp_path: Path, now: datetime) -> None:
        db_path = tmp_path / "test.duckdb"
        market = Market(
            id="persist-test",
            exchange=ExchangeName.MANIFOLD,
            title="Persistence test",
            outcomes=[
                Outcome(name="Yes", price=0.6, volume=0.0),
                Outcome(name="No", price=0.4, volume=0.0),
            ],
            fetched_at=now,
        )

        # Write
        with Storage(db_path) as storage:
            storage.insert_markets([market])

        # Read back in new connection
        with Storage(db_path) as storage:
            results = storage.query_markets()
            assert len(results) == 1
            assert results[0]["id"] == "persist-test"
