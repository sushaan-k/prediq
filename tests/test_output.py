"""Tests for arbiter.output -- alerts, export, and API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from arbiter.models import (
    ContractType,
    Divergence,
    ExchangeName,
    Market,
    MarketStatus,
    Outcome,
    ProbabilityViolation,
)
from arbiter.output.alerts import AlertManager
from arbiter.output.export import DataExporter

# ── AlertManager ─────────────────────────────────────────────────────


class TestAlertManager:
    """Tests for the AlertManager class."""

    def test_alert_from_divergence_above_threshold(self) -> None:
        manager = AlertManager()
        div = Divergence(
            event="Test Event",
            outcome="Yes",
            exchange_a=ExchangeName.POLYMARKET,
            exchange_b=ExchangeName.KALSHI,
            price_a=0.55,
            price_b=0.45,
            spread=0.10,
            spread_pct=0.20,
        )
        alert = manager.alert_from_divergence(div, threshold=0.05)
        assert alert is not None
        assert alert.alert_type == "divergence"
        assert alert.severity == "critical"  # spread >= 0.10

    def test_alert_from_divergence_below_threshold(self) -> None:
        manager = AlertManager()
        div = Divergence(
            event="Small",
            outcome="Yes",
            exchange_a=ExchangeName.POLYMARKET,
            exchange_b=ExchangeName.KALSHI,
            price_a=0.51,
            price_b=0.50,
            spread=0.01,
            spread_pct=0.02,
        )
        alert = manager.alert_from_divergence(div, threshold=0.05)
        assert alert is None

    def test_alert_severity_levels(self) -> None:
        manager = AlertManager()
        test_cases = [
            (0.03, "low"),
            (0.05, "medium"),
            (0.08, "high"),
            (0.12, "critical"),
        ]
        for spread, expected_severity in test_cases:
            div = Divergence(
                event="Test",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.50 + spread / 2,
                price_b=0.50 - spread / 2,
                spread=spread,
                spread_pct=spread * 2,
            )
            alert = manager.alert_from_divergence(div, threshold=0.01)
            assert alert is not None
            assert alert.severity == expected_severity, (
                f"spread={spread} expected {expected_severity}, got {alert.severity}"
            )

    def test_alert_from_violation(self) -> None:
        manager = AlertManager()
        violation = ProbabilityViolation(
            market="Test Violation",
            market_id="test-v1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.55,
            no_price=0.52,
            price_sum=1.07,
            implied_arb=0.07,
            volume_available=50_000.0,
        )
        alert = manager.alert_from_violation(violation)
        assert alert.alert_type == "violation"
        assert "1.070" in alert.message

    @pytest.mark.asyncio
    async def test_emit_stores_history(self) -> None:
        manager = AlertManager()
        violation = ProbabilityViolation(
            market="Test",
            market_id="t-1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.55,
            no_price=0.52,
            price_sum=1.07,
            implied_arb=0.07,
        )
        alert = manager.alert_from_violation(violation)
        await manager.emit(alert)
        assert len(manager.history) == 1

    def test_add_webhook(self) -> None:
        manager = AlertManager()
        manager.add_webhook("https://example.com/hook")
        assert len(manager._webhook_urls) == 1

    def test_add_callback(self) -> None:
        received = []
        manager = AlertManager()
        manager.add_callback(lambda alert: received.append(alert))
        assert len(manager._callbacks) == 1

    @pytest.mark.asyncio
    async def test_callback_invoked_on_emit(self) -> None:
        received = []
        manager = AlertManager()
        manager.add_callback(lambda alert: received.append(alert))

        violation = ProbabilityViolation(
            market="Callback test",
            market_id="cb-1",
            exchange=ExchangeName.KALSHI,
            yes_price=0.60,
            no_price=0.50,
            price_sum=1.10,
            implied_arb=0.10,
        )
        alert = manager.alert_from_violation(violation)
        await manager.emit(alert)
        assert len(received) == 1


# ── DataExporter ─────────────────────────────────────────────────────


class TestDataExporter:
    """Tests for the DataExporter class."""

    @pytest.fixture()
    def sample_markets(self, now: datetime) -> list[Market]:
        return [
            Market(
                id=f"exp-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Export market {i}",
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

    def test_export_to_parquet(
        self, sample_markets: list[Market], tmp_path: Path
    ) -> None:
        exporter = DataExporter()
        out = tmp_path / "markets.parquet"
        result = exporter.export_markets_to_parquet(sample_markets, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_to_csv(self, sample_markets: list[Market], tmp_path: Path) -> None:
        exporter = DataExporter()
        out = tmp_path / "markets.csv"
        result = exporter.export_markets_to_csv(sample_markets, out)
        assert result.exists()
        content = result.read_text()
        assert "Export market 0" in content
        assert "polymarket" in content

    def test_export_empty_markets(self, tmp_path: Path) -> None:
        exporter = DataExporter()
        out = tmp_path / "empty.parquet"
        result = exporter.export_markets_to_parquet([], out)
        assert result.exists()
        assert pq.read_table(result).column_names == DataExporter.MARKET_EXPORT_FIELDS

    def test_export_empty_markets_csv_header(self, tmp_path: Path) -> None:
        exporter = DataExporter()
        out = tmp_path / "empty.csv"
        result = exporter.export_markets_to_csv([], out)
        assert result.exists()
        assert result.read_text().splitlines() == [
            ",".join(DataExporter.MARKET_EXPORT_FIELDS)
        ]

    def test_markets_to_csv_string(self, sample_markets: list[Market]) -> None:
        exporter = DataExporter()
        csv_str = exporter.markets_to_csv_string(sample_markets)
        assert "Export market 0" in csv_str
        lines = csv_str.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_markets_to_csv_string_empty(self) -> None:
        exporter = DataExporter()
        assert exporter.markets_to_csv_string([]).strip() == ",".join(
            DataExporter.MARKET_EXPORT_FIELDS
        )

    def test_export_divergences_to_parquet(self, tmp_path: Path) -> None:
        exporter = DataExporter()
        divergences = [
            Divergence(
                event="Test Div",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.55,
                price_b=0.48,
                spread=0.07,
                spread_pct=0.136,
            )
        ]
        out = tmp_path / "divergences.parquet"
        result = exporter.export_divergences_to_parquet(divergences, out)
        assert result.exists()

    def test_market_record_fields(self, now: datetime) -> None:
        market = Market(
            id="rec-test",
            exchange=ExchangeName.KALSHI,
            title="Record test",
            category="Test",
            outcomes=[
                Outcome(name="Yes", price=0.7, volume=0.0),
                Outcome(name="No", price=0.3, volume=0.0),
            ],
            volume_total=50_000.0,
            fetched_at=now,
        )
        record = DataExporter._market_to_record(market)
        assert record["id"] == "rec-test"
        assert record["exchange"] == "kalshi"
        assert record["yes_price"] == 0.7
        assert record["volume_total"] == 50_000.0
