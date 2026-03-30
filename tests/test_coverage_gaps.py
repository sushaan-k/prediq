"""Tests for remaining coverage gaps in models, export, quality, and violations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from arbiter.analytics.divergence import DivergenceDetector
from arbiter.analytics.quality import QualityScorer
from arbiter.analytics.violations import ViolationDetector
from arbiter.exceptions import InsufficientDataError
from arbiter.models import (
    ContractType,
    ExchangeName,
    Market,
    MarketPair,
    MarketStatus,
    Outcome,
)
from arbiter.output.export import DataExporter

# ── Models: yes_price / no_price fallback paths ─────────────────────────


class TestMarketPriceFallbacks:
    """Cover lines 144 and 154 in models.py -- when outcome names are not Yes/No."""

    def test_yes_price_fallback_to_first_outcome(self) -> None:
        """When no outcome is named 'Yes', fallback to outcomes[0].price."""
        market = Market(
            id="fb-1",
            exchange=ExchangeName.POLYMARKET,
            title="Fallback test",
            contract_type=ContractType.BINARY,
            outcomes=[
                Outcome(name="Agree", price=0.60, volume=0.0),
                Outcome(name="Disagree", price=0.40, volume=0.0),
            ],
            fetched_at=datetime.now(UTC),
        )
        assert market.yes_price == 0.60

    def test_no_price_fallback_to_second_outcome(self) -> None:
        """When no outcome is named 'No', fallback to outcomes[1].price."""
        market = Market(
            id="fb-2",
            exchange=ExchangeName.POLYMARKET,
            title="Fallback test",
            contract_type=ContractType.BINARY,
            outcomes=[
                Outcome(name="Agree", price=0.60, volume=0.0),
                Outcome(name="Disagree", price=0.40, volume=0.0),
            ],
            fetched_at=datetime.now(UTC),
        )
        assert market.no_price == 0.40


# ── Export: empty CSV export ─────────────────────────────────────────────


class TestExportEmptyCSV:
    def test_export_empty_csv(self, tmp_path: Path) -> None:
        """Cover lines 85-87 in export.py -- empty CSV export."""
        exporter = DataExporter()
        out = tmp_path / "empty.csv"
        result = exporter.export_markets_to_csv([], out)
        assert result.exists()
        assert result.read_text().splitlines() == [
            ",".join(DataExporter.MARKET_EXPORT_FIELDS)
        ]

    def test_export_empty_divergences_parquet(self, tmp_path: Path) -> None:
        """Cover line 119 in export.py -- empty divergences parquet."""
        exporter = DataExporter()
        out = tmp_path / "empty_divs.parquet"
        result = exporter.export_divergences_to_parquet([], out)
        assert result.exists()
        assert (
            pq.read_table(result).column_names == DataExporter.DIVERGENCE_EXPORT_FIELDS
        )


# ── Quality: branch coverage in _filter_resolved ─────────────────────────


class TestQualityScorerBranches:
    def test_filter_skips_active_markets(self) -> None:
        """Line 105: non-resolved market should be skipped."""
        now = datetime.now(UTC)
        active = Market(
            id="act",
            exchange=ExchangeName.POLYMARKET,
            title="Active",
            contract_type=ContractType.BINARY,
            status=MarketStatus.ACTIVE,
            outcomes=[
                Outcome(name="Yes", price=0.50, volume=0.0),
                Outcome(name="No", price=0.50, volume=0.0),
            ],
            fetched_at=now,
        )
        scorer = QualityScorer(min_sample_size=1)
        with pytest.raises(InsufficientDataError):
            scorer.score([active])

    def test_filter_skips_wrong_exchange(self) -> None:
        """Line 132 (equivalent): exchange filter skips non-matching."""
        now = datetime.now(UTC)
        markets = [
            Market(
                id=f"m-{i}",
                exchange=ExchangeName.KALSHI,
                title=f"Kalshi {i}",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.70, volume=0.0),
                    Outcome(name="No", price=0.30, volume=0.0),
                ],
                resolution="yes",
                volume_total=10_000.0,
                fetched_at=now,
            )
            for i in range(10)
        ]
        scorer = QualityScorer(min_sample_size=1)
        # Filter for Polymarket should find 0 Kalshi markets
        with pytest.raises(InsufficientDataError):
            scorer.score(markets, exchange=ExchangeName.POLYMARKET)

    def test_brier_score_empty_markets(self) -> None:
        """_brier_score with empty list should return 1.0."""
        assert QualityScorer._brier_score([]) == 1.0

    def test_calibration_error_empty(self) -> None:
        """_calibration_error with empty list should return 1.0."""
        assert QualityScorer._calibration_error([]) == 1.0

    def test_avg_resolution_time_no_timestamps(self) -> None:
        """Markets without closes_at/resolved_at yield 0.0."""
        now = datetime.now(UTC)
        markets = [
            Market(
                id="no-ts",
                exchange=ExchangeName.POLYMARKET,
                title="No timestamps",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.50, volume=0.0),
                    Outcome(name="No", price=0.50, volume=0.0),
                ],
                resolution="yes",
                fetched_at=now,
            )
        ]
        result = QualityScorer._avg_resolution_time(markets)
        assert result == 0.0

    def test_manipulation_score_empty(self) -> None:
        assert QualityScorer._manipulation_score([]) == 0.0

    def test_manipulation_score_low_confidence_wrong(self) -> None:
        """Low confidence YES (0.05) that resolved YES is suspicious."""
        now = datetime.now(UTC)
        markets = [
            Market(
                id="low-sus",
                exchange=ExchangeName.POLYMARKET,
                title="Low and wrong",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.05, volume=0.0),
                    Outcome(name="No", price=0.95, volume=0.0),
                ],
                resolution="yes",  # 5% said Yes, but it resolved Yes
                volume_total=10_000.0,
                fetched_at=now,
            )
        ]
        score = QualityScorer._manipulation_score(markets)
        assert score > 0.0


# ── Violations: outcome volume fallback ──────────────────────────────────


class TestViolationsOutcomeVolume:
    def test_outcome_volume_fallback_to_market_volume(self) -> None:
        """When outcome volume is 0 and no order book, use market volume."""
        now = datetime.now(UTC)
        market = Market(
            id="vol-fb",
            exchange=ExchangeName.POLYMARKET,
            title="Volume fallback",
            outcomes=[
                Outcome(name="Yes", price=0.60, volume=0.0),
                Outcome(name="No", price=0.55, volume=0.0),
            ],
            volume_total=100_000.0,
            fetched_at=now,
        )
        vol = ViolationDetector._outcome_volume(market, "yes")
        assert vol == 100_000.0

    def test_outcome_volume_with_side_not_found(self) -> None:
        """When the named side doesn't exist."""
        now = datetime.now(UTC)
        market = Market(
            id="no-side",
            exchange=ExchangeName.POLYMARKET,
            title="No side",
            outcomes=[
                Outcome(name="Agree", price=0.60, volume=5_000.0),
                Outcome(name="Disagree", price=0.40, volume=3_000.0),
            ],
            volume_total=50_000.0,
            fetched_at=now,
        )
        vol = ViolationDetector._outcome_volume(market, "yes")
        # "yes" side not found -> fallback to market volume
        assert vol == 50_000.0


# ── Divergence: binary pair with None yes_price ──────────────────────────


class TestDivergenceBinaryNone:
    def test_binary_pair_none_yes_price(self) -> None:
        """When yes_price is None on one side, return empty."""
        now = datetime.now(UTC)
        # Market A is multi-outcome but we force binary check
        ma = Market(
            id="pm-nones",
            exchange=ExchangeName.POLYMARKET,
            title="None prices",
            contract_type=ContractType.BINARY,
            outcomes=[
                Outcome(name="Agree", price=0.50, volume=0.0),
                Outcome(name="Disagree", price=0.50, volume=0.0),
            ],
            fetched_at=now,
        )
        mb = Market(
            id="kx-nones",
            exchange=ExchangeName.KALSHI,
            title="None prices",
            contract_type=ContractType.BINARY,
            outcomes=[
                Outcome(name="Yes", price=0.40, volume=0.0),
                Outcome(name="No", price=0.60, volume=0.0),
            ],
            fetched_at=now,
        )
        pair = MarketPair(market_a=ma, market_b=mb, similarity_score=0.9)
        detector = DivergenceDetector(min_spread=0.01)
        divs = detector.detect([pair])
        # ma.yes_price fallback to first outcome (0.50), mb.yes_price = 0.40
        # spread = 0.10 > 0.01
        assert len(divs) == 1
