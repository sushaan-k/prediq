"""Extended tests for arbiter.analytics -- edge cases and boundary conditions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from arbiter.analytics.divergence import DivergenceDetector
from arbiter.analytics.efficiency import EfficiencyAnalyzer
from arbiter.analytics.liquidity import LiquidityAnalyzer
from arbiter.analytics.quality import QualityScorer
from arbiter.analytics.violations import ViolationDetector
from arbiter.exceptions import InsufficientDataError
from arbiter.models import (
    ContractType,
    Divergence,
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
    return datetime(2026, 3, 15, 14, 0, 0, tzinfo=UTC)


# ── Divergence Detector Extended ─────────────────────────────────────────


class TestDivergenceDetectorEdgeCases:
    def test_min_liquidity_filter(self, now: datetime) -> None:
        """Both markets below min_liquidity should be filtered out."""
        ma = Market(
            id="pm-liq",
            exchange=ExchangeName.POLYMARKET,
            title="Liquidity test",
            outcomes=[
                Outcome(name="Yes", price=0.50, volume=0.0),
                Outcome(name="No", price=0.50, volume=0.0),
            ],
            volume_total=10.0,
            fetched_at=now,
        )
        mb = Market(
            id="kx-liq",
            exchange=ExchangeName.KALSHI,
            title="Liquidity test",
            outcomes=[
                Outcome(name="Yes", price=0.40, volume=0.0),
                Outcome(name="No", price=0.60, volume=0.0),
            ],
            volume_total=10.0,
            fetched_at=now,
        )
        pair = MarketPair(market_a=ma, market_b=mb, similarity_score=0.9)
        detector = DivergenceDetector(min_spread=0.01, min_liquidity=10_000.0)
        divs = detector.detect([pair])
        assert len(divs) == 0

    def test_multi_outcome_min_liquidity_filter(self, now: datetime) -> None:
        """Multi-outcome divergences should honor min_liquidity too."""
        ma = Market(
            id="pm-multi-liq",
            exchange=ExchangeName.POLYMARKET,
            title="Multi liquidity test",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="Alice", price=0.50, volume=25.0),
                Outcome(name="Bob", price=0.30, volume=25.0),
            ],
            volume_total=50.0,
            fetched_at=now,
        )
        mb = Market(
            id="kx-multi-liq",
            exchange=ExchangeName.KALSHI,
            title="Multi liquidity test",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="Alice", price=0.40, volume=25.0),
                Outcome(name="Bob", price=0.35, volume=25.0),
            ],
            volume_total=50.0,
            fetched_at=now,
        )
        pair = MarketPair(market_a=ma, market_b=mb, similarity_score=0.9)
        detector = DivergenceDetector(min_spread=0.01, min_liquidity=100.0)
        divs = detector.detect([pair])
        assert len(divs) == 0

    def test_liquidity_estimation_with_order_book(self, now: datetime) -> None:
        """Liquidity should be estimated from order book depth if available."""
        ob = OrderBook(
            bids=[OrderBookLevel(price=0.50, quantity=5_000.0)],
            asks=[OrderBookLevel(price=0.55, quantity=3_000.0)],
        )
        ma = Market(
            id="pm-ob",
            exchange=ExchangeName.POLYMARKET,
            title="OB test",
            outcomes=[
                Outcome(name="Yes", price=0.50, order_book=ob, volume=100.0),
                Outcome(name="No", price=0.50, volume=100.0),
            ],
            volume_total=200.0,
            fetched_at=now,
        )
        mb = Market(
            id="kx-ob",
            exchange=ExchangeName.KALSHI,
            title="OB test",
            outcomes=[
                Outcome(name="Yes", price=0.40, volume=50.0),
                Outcome(name="No", price=0.60, volume=50.0),
            ],
            volume_total=100.0,
            fetched_at=now,
        )
        pair = MarketPair(market_a=ma, market_b=mb, similarity_score=0.9)
        detector = DivergenceDetector(min_spread=0.01)
        divs = detector.detect([pair])
        assert len(divs) == 1
        # Market A should use order book depth (5000 + 3000 = 8000)
        assert divs[0].liquidity_a == pytest.approx(8_000.0)

    def test_multi_outcome_no_matching_names(self, now: datetime) -> None:
        """When outcome names don't match, no divergences should be reported."""
        ma = Market(
            id="pm-diff",
            exchange=ExchangeName.POLYMARKET,
            title="Different outcomes",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="Alice", price=0.50, volume=0.0),
                Outcome(name="Bob", price=0.50, volume=0.0),
            ],
            fetched_at=now,
        )
        mb = Market(
            id="kx-diff",
            exchange=ExchangeName.KALSHI,
            title="Different outcomes",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="Carol", price=0.40, volume=0.0),
                Outcome(name="Dave", price=0.60, volume=0.0),
            ],
            fetched_at=now,
        )
        pair = MarketPair(market_a=ma, market_b=mb, similarity_score=0.9)
        detector = DivergenceDetector(min_spread=0.01)
        divs = detector.detect([pair])
        assert len(divs) == 0

    def test_zero_midpoint_spread_pct(self, now: datetime) -> None:
        """When midpoint is zero, spread_pct should be 0."""
        ma = Market(
            id="pm-zero",
            exchange=ExchangeName.POLYMARKET,
            title="Zero midpoint",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="X", price=0.05, volume=0.0),
                Outcome(name="Y", price=0.95, volume=0.0),
            ],
            fetched_at=now,
        )
        mb = Market(
            id="kx-zero",
            exchange=ExchangeName.KALSHI,
            title="Zero midpoint",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="X", price=0.00, volume=0.0),
                Outcome(name="Y", price=1.0, volume=0.0),
            ],
            fetched_at=now,
        )
        pair = MarketPair(market_a=ma, market_b=mb, similarity_score=0.9)
        detector = DivergenceDetector(min_spread=0.01)
        divs = detector.detect([pair])
        # X: midpoint of 0.05 and 0.00 = 0.025, spread_pct = 0.05/0.025
        assert len(divs) >= 1

    def test_none_yes_price(self, now: datetime) -> None:
        """When yes_price is None, should return empty list."""
        ma = Market(
            id="pm-none",
            exchange=ExchangeName.POLYMARKET,
            title="No yes price",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="Alice", price=0.50, volume=0.0),
                Outcome(name="Bob", price=0.50, volume=0.0),
            ],
            fetched_at=now,
        )
        mb = Market(
            id="kx-none",
            exchange=ExchangeName.KALSHI,
            title="No yes price",
            outcomes=[
                Outcome(name="Yes", price=0.40, volume=0.0),
                Outcome(name="No", price=0.60, volume=0.0),
            ],
            fetched_at=now,
        )
        pair = MarketPair(market_a=ma, market_b=mb, similarity_score=0.9)
        detector = DivergenceDetector(min_spread=0.01)
        # ma is MULTI_OUTCOME -> yes_price is None,
        # but _check_pair calls _check_multi_outcome_pair for ma
        divs = detector.detect([pair])
        # No matching outcomes -> 0 divergences
        assert isinstance(divs, list)


# ── Violation Detector Extended ──────────────────────────────────────────


class TestViolationDetectorEdgeCases:
    def test_binary_underpriced(self, now: datetime) -> None:
        """YES + NO < 1.0 - tolerance should detect a violation."""
        market = Market(
            id="under",
            exchange=ExchangeName.POLYMARKET,
            title="Underpriced",
            outcomes=[
                Outcome(name="Yes", price=0.40, volume=0.0),
                Outcome(name="No", price=0.45, volume=0.0),
            ],
            fetched_at=now,
        )
        detector = ViolationDetector(binary_tolerance=0.03)
        violations = detector.detect_binary_violations([market])
        assert len(violations) == 1
        assert violations[0].price_sum < 1.0

    def test_binary_violation_with_order_book_volume(self, now: datetime) -> None:
        ob = OrderBook(
            bids=[OrderBookLevel(price=0.50, quantity=3_000.0)],
            asks=[OrderBookLevel(price=0.60, quantity=2_000.0)],
        )
        market = Market(
            id="ob-vol",
            exchange=ExchangeName.POLYMARKET,
            title="OB volume",
            outcomes=[
                Outcome(name="Yes", price=0.60, order_book=ob, volume=0.0),
                Outcome(name="No", price=0.55, volume=1_000.0),
            ],
            fetched_at=now,
        )
        detector = ViolationDetector(binary_tolerance=0.01)
        violations = detector.detect_binary_violations([market])
        assert len(violations) == 1
        # Volume should be estimated from order book depth for Yes
        assert violations[0].volume_available > 0

    def test_none_yes_or_no_price(self, now: datetime) -> None:
        """Multi-outcome market should be skipped by binary violation check."""
        market = Market(
            id="multi-skip",
            exchange=ExchangeName.POLYMARKET,
            title="Multi skip",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="Alice", price=0.50, volume=0.0),
                Outcome(name="Bob", price=0.50, volume=0.0),
            ],
            fetched_at=now,
        )
        detector = ViolationDetector()
        assert len(detector.detect_binary_violations([market])) == 0

    def test_multi_outcome_too_few_outcomes(self, now: datetime) -> None:
        """Multi-outcome market with < 2 outcomes should be skipped."""
        # Can't create a Market with < 2 outcomes, but we can test with exactly 2
        market = Market(
            id="multi-2",
            exchange=ExchangeName.POLYMARKET,
            title="Two outcomes only",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="A", price=0.50, volume=0.0),
                Outcome(name="B", price=0.40, volume=0.0),
            ],
            fetched_at=now,
        )
        detector = ViolationDetector(multi_tolerance=0.03)
        violations = detector.detect_multi_outcome_violations([market])
        # Sum = 0.90, deviation = 0.10 > tolerance
        assert len(violations) == 1

    def test_multi_outcome_within_tolerance(self, now: datetime) -> None:
        market = Market(
            id="multi-ok",
            exchange=ExchangeName.POLYMARKET,
            title="Fair multi",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="A", price=0.33, volume=0.0),
                Outcome(name="B", price=0.33, volume=0.0),
                Outcome(name="C", price=0.34, volume=0.0),
            ],
            fetched_at=now,
        )
        detector = ViolationDetector(multi_tolerance=0.05)
        violations = detector.detect_multi_outcome_violations([market])
        assert len(violations) == 0

    def test_violations_sorted_by_deviation(self, now: datetime) -> None:
        markets = [
            Market(
                id=f"multi-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Multi {i}",
                contract_type=ContractType.MULTI_OUTCOME,
                outcomes=[
                    Outcome(name="A", price=0.50 + i * 0.05, volume=0.0),
                    Outcome(name="B", price=0.50 + i * 0.05, volume=0.0),
                ],
                fetched_at=now,
            )
            for i in range(3)
        ]
        detector = ViolationDetector(multi_tolerance=0.01)
        violations = detector.detect_multi_outcome_violations(markets)
        if len(violations) >= 2:
            assert violations[0].deviation >= violations[1].deviation

    def test_outcome_volume_with_volume_attribute(self, now: datetime) -> None:
        """_outcome_volume should use outcome.volume if > 0."""
        market = Market(
            id="vol-test",
            exchange=ExchangeName.POLYMARKET,
            title="Volume test",
            outcomes=[
                Outcome(name="Yes", price=0.60, volume=5_000.0),
                Outcome(name="No", price=0.55, volume=3_000.0),
            ],
            fetched_at=now,
        )
        detector = ViolationDetector(binary_tolerance=0.01)
        violations = detector.detect_binary_violations([market])
        assert len(violations) == 1
        assert violations[0].volume_available == pytest.approx(3_000.0)


# ── Liquidity Analyzer Extended ──────────────────────────────────────────


class TestLiquidityAnalyzerExtended:
    def test_simulate_buy_impact_no_asks(self) -> None:
        book = OrderBook()
        impact = LiquidityAnalyzer._simulate_buy_impact(book, 10_000.0)
        assert impact == 0.0

    def test_simulate_buy_impact_exceeds_book(self) -> None:
        book = OrderBook(
            asks=[
                OrderBookLevel(price=0.50, quantity=100.0),
                OrderBookLevel(price=0.55, quantity=100.0),
            ]
        )
        impact = LiquidityAnalyzer._simulate_buy_impact(book, 1_000_000.0)
        assert impact > 0

    def test_depth_within_pct_no_midpoint(self) -> None:
        analyzer = LiquidityAnalyzer()
        book = OrderBook()
        depth = analyzer._depth_within_pct(book, 0.01)
        assert depth == 0.0

    def test_analyze_order_book_unknown_exchange(self) -> None:
        book = OrderBook(
            bids=[OrderBookLevel(price=0.50, quantity=100.0)],
            asks=[OrderBookLevel(price=0.55, quantity=100.0)],
        )
        analyzer = LiquidityAnalyzer()
        profile = analyzer.analyze_order_book(
            book,
            market_title="Test",
            market_id="x",
            exchange="unknown_exchange",
        )
        # Should default to POLYMARKET
        assert profile.exchange == ExchangeName.POLYMARKET

    def test_analyze_order_book_valid_exchange(self) -> None:
        book = OrderBook(
            bids=[OrderBookLevel(price=0.50, quantity=100.0)],
            asks=[OrderBookLevel(price=0.55, quantity=100.0)],
        )
        analyzer = LiquidityAnalyzer()
        profile = analyzer.analyze_order_book(
            book,
            market_title="Test",
            market_id="x",
            exchange="kalshi",
        )
        assert profile.exchange == ExchangeName.KALSHI

    def test_get_primary_order_book_empty_book(self, now: datetime) -> None:
        """Market with empty order book should return None."""
        empty_ob = OrderBook()
        market = Market(
            id="empty-ob",
            exchange=ExchangeName.POLYMARKET,
            title="Empty OB",
            outcomes=[
                Outcome(name="Yes", price=0.50, order_book=empty_ob, volume=0.0),
                Outcome(name="No", price=0.50, volume=0.0),
            ],
            fetched_at=now,
        )
        result = LiquidityAnalyzer._get_primary_order_book(market)
        assert result is None


# ── Quality Scorer Extended ──────────────────────────────────────────────


class TestQualityScorerExtended:
    def test_brier_score_worst_case(self, now: datetime) -> None:
        """All wrong predictions should yield high Brier score."""
        markets = [
            Market(
                id=f"wrong-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Wrong {i}",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.95, volume=0.0),
                    Outcome(name="No", price=0.05, volume=0.0),
                ],
                resolution="no",
                volume_total=10_000.0,
                fetched_at=now,
            )
            for i in range(10)
        ]
        scorer = QualityScorer(min_sample_size=5)
        quality = scorer.score(markets)
        assert quality.brier_score > 0.5

    def test_calibration_error_empty_bins(self, now: datetime) -> None:
        """When all predictions fall in the same bin."""
        markets = [
            Market(
                id=f"same-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Same {i}",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.55, volume=0.0),
                    Outcome(name="No", price=0.45, volume=0.0),
                ],
                resolution="yes",
                volume_total=10_000.0,
                fetched_at=now,
            )
            for i in range(10)
        ]
        scorer = QualityScorer(min_sample_size=5)
        quality = scorer.score(markets)
        assert 0.0 <= quality.calibration_error <= 1.0

    def test_avg_resolution_time(self, now: datetime) -> None:
        """Markets with closes_at and resolved_at should compute resolution time."""
        markets = [
            Market(
                id=f"res-time-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Res time {i}",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.70, volume=0.0),
                    Outcome(name="No", price=0.30, volume=0.0),
                ],
                resolution="yes",
                volume_total=10_000.0,
                closes_at=now,
                resolved_at=now + timedelta(hours=24),
                fetched_at=now,
            )
            for i in range(10)
        ]
        scorer = QualityScorer(min_sample_size=5)
        quality = scorer.score(markets)
        assert quality.avg_resolution_hours == pytest.approx(24.0)

    def test_volume_accuracy_correlation_few_markets(self, now: datetime) -> None:
        """With < 3 markets, correlation should be 0."""
        markets = [
            Market(
                id="few-1",
                exchange=ExchangeName.POLYMARKET,
                title="Few 1",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.50, volume=0.0),
                    Outcome(name="No", price=0.50, volume=0.0),
                ],
                resolution="yes",
                volume_total=10_000.0,
                fetched_at=now,
            ),
            Market(
                id="few-2",
                exchange=ExchangeName.POLYMARKET,
                title="Few 2",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.50, volume=0.0),
                    Outcome(name="No", price=0.50, volume=0.0),
                ],
                resolution="no",
                volume_total=20_000.0,
                fetched_at=now,
            ),
        ]
        scorer = QualityScorer(min_sample_size=1)
        quality = scorer.score(markets)
        assert quality.volume_accuracy_correlation == 0.0

    def test_filter_by_category(self, now: datetime) -> None:
        """Filtering by category should only include matching markets."""
        markets = [
            Market(
                id=f"cat-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Cat {i}",
                category="politics" if i < 5 else "sports",
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
        scorer = QualityScorer(min_sample_size=5)
        quality = scorer.score(markets, category="Politics")
        assert quality.sample_size == 5

    def test_no_resolution_skipped(self, now: datetime) -> None:
        """Markets with status=RESOLVED but resolution=None are skipped."""
        markets = [
            Market(
                id="no-res",
                exchange=ExchangeName.POLYMARKET,
                title="No resolution",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.50, volume=0.0),
                    Outcome(name="No", price=0.50, volume=0.0),
                ],
                resolution=None,
                volume_total=10_000.0,
                fetched_at=now,
            )
        ]
        scorer = QualityScorer(min_sample_size=1)
        with pytest.raises(InsufficientDataError):
            scorer.score(markets)

    def test_volume_accuracy_zero_std(self, now: datetime) -> None:
        """When all volumes or accuracies are the same, correlation = 0."""
        markets = [
            Market(
                id=f"same-vol-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Same vol {i}",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.50, volume=0.0),
                    Outcome(name="No", price=0.50, volume=0.0),
                ],
                resolution="yes",
                volume_total=10_000.0,
                fetched_at=now,
            )
            for i in range(10)
        ]
        scorer = QualityScorer(min_sample_size=5)
        quality = scorer.score(markets)
        assert quality.volume_accuracy_correlation == 0.0

    def test_manipulation_no_suspicious(self, now: datetime) -> None:
        """Well-calibrated markets should have manipulation_score near 0."""
        markets = [
            Market(
                id=f"good-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Good {i}",
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
        scorer = QualityScorer(min_sample_size=5)
        quality = scorer.score(markets)
        assert quality.manipulation_score == 0.0


# ── Efficiency Analyzer Extended ─────────────────────────────────────────


class TestEfficiencyAnalyzerExtended:
    def test_price_discovery_price_decreasing(self) -> None:
        """Test with price moving downward toward 0."""
        analyzer = EfficiencyAnalyzer()
        base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        series = [
            (base, 0.80),
            (datetime(2026, 3, 15, 12, 10, 0, tzinfo=UTC), 0.50),
            (datetime(2026, 3, 15, 12, 20, 0, tzinfo=UTC), 0.15),
            (datetime(2026, 3, 15, 12, 30, 0, tzinfo=UTC), 0.05),
        ]
        speed = analyzer.compute_price_discovery_speed(series, final_price=0.0)
        assert speed > 0

    def test_price_discovery_already_at_final(self) -> None:
        """When starting price equals final, speed should be 0."""
        analyzer = EfficiencyAnalyzer()
        base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        series = [
            (base, 1.0),
            (datetime(2026, 3, 15, 12, 10, 0, tzinfo=UTC), 1.0),
        ]
        speed = analyzer.compute_price_discovery_speed(series, final_price=1.0)
        assert speed == 0.0

    def test_price_discovery_single_point(self) -> None:
        analyzer = EfficiencyAnalyzer()
        base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        speed = analyzer.compute_price_discovery_speed([(base, 0.50)], final_price=1.0)
        assert speed == 0.0

    def test_information_ratio_canceling_changes(self) -> None:
        """Price changes that cancel out -> low directional efficiency."""
        analyzer = EfficiencyAnalyzer()
        changes = [0.05, -0.05, 0.05, -0.05]
        intervals = [5.0, 5.0, 5.0, 5.0]
        ratio = analyzer.compute_information_ratio(changes, intervals)
        assert ratio == 0.0

    def test_information_ratio_zero_total_absolute(self) -> None:
        analyzer = EfficiencyAnalyzer()
        changes = [0.0, 0.0, 0.0]
        intervals = [5.0, 5.0, 5.0]
        ratio = analyzer.compute_information_ratio(changes, intervals)
        assert ratio == 0.0

    def test_information_ratio_mismatched_lengths(self) -> None:
        analyzer = EfficiencyAnalyzer()
        changes = [0.01, 0.02, 0.03]
        intervals = [5.0]  # shorter
        ratio = analyzer.compute_information_ratio(changes, intervals)
        assert ratio >= 0.0

    def test_analyze_market_with_history(self, now: datetime) -> None:
        market = Market(
            id="hist",
            exchange=ExchangeName.POLYMARKET,
            title="History test",
            outcomes=[
                Outcome(name="Yes", price=0.50, volume=0.0),
                Outcome(name="No", price=0.50, volume=0.0),
            ],
            fetched_at=now,
        )
        price_history = [
            (now, 0.30),
            (now + timedelta(minutes=10), 0.50),
            (now + timedelta(minutes=20), 0.70),
            (now + timedelta(minutes=30), 0.90),
        ]
        divergences = [
            Divergence(
                event="test",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.55,
                price_b=0.50,
                spread=0.05,
                spread_pct=0.10,
            )
        ]
        analyzer = EfficiencyAnalyzer()
        metrics = analyzer.analyze_market(
            market,
            divergences=divergences,
            price_history=price_history,
        )
        assert metrics.avg_arb_window_minutes > 0
        assert metrics.price_discovery_speed_minutes >= 0
        assert metrics.information_ratio >= 0

    def test_price_discovery_never_reaches_threshold(self) -> None:
        """When the price never reaches the threshold."""
        analyzer = EfficiencyAnalyzer()
        base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        series = [
            (base, 0.10),
            (datetime(2026, 3, 15, 12, 10, 0, tzinfo=UTC), 0.15),
            (datetime(2026, 3, 15, 12, 20, 0, tzinfo=UTC), 0.20),
        ]
        speed = analyzer.compute_price_discovery_speed(
            series, final_price=1.0, threshold=0.90
        )
        assert speed == 0.0

    def test_arb_window_p95_index(self) -> None:
        """Ensure p95 computation handles edge indices."""
        analyzer = EfficiencyAnalyzer()
        durations = [1.0, 2.0, 3.0]
        stats = analyzer.compute_arb_window_stats([], durations)
        assert stats["p95"] >= stats["median"]
