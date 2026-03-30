"""Tests for arbiter.analytics -- divergence, violations, liquidity, quality, efficiency."""

from __future__ import annotations

from datetime import UTC, datetime

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
    Outcome,
)

# ── Divergence Detection ─────────────────────────────────────────────


class TestDivergenceDetector:
    """Tests for DivergenceDetector."""

    def test_detect_binary_divergence(self, market_pair: MarketPair) -> None:
        detector = DivergenceDetector(min_spread=0.01)
        divergences = detector.detect([market_pair])
        assert len(divergences) == 1

        d = divergences[0]
        assert d.spread == pytest.approx(0.05)
        assert d.exchange_a == ExchangeName.POLYMARKET
        assert d.exchange_b == ExchangeName.KALSHI
        assert d.outcome == "Yes"

    def test_no_divergence_below_threshold(self, market_pair: MarketPair) -> None:
        detector = DivergenceDetector(min_spread=0.10)
        divergences = detector.detect([market_pair])
        assert len(divergences) == 0

    def test_multi_outcome_divergence(self, now: datetime) -> None:
        ma = Market(
            id="pm-1",
            exchange=ExchangeName.POLYMARKET,
            title="Who wins?",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="Alice", price=0.40, volume=10_000.0),
                Outcome(name="Bob", price=0.35, volume=8_000.0),
                Outcome(name="Other", price=0.25, volume=5_000.0),
            ],
            fetched_at=now,
        )
        mb = Market(
            id="kx-1",
            exchange=ExchangeName.KALSHI,
            title="Who wins?",
            contract_type=ContractType.MULTI_OUTCOME,
            outcomes=[
                Outcome(name="Alice", price=0.45, volume=6_000.0),
                Outcome(name="Bob", price=0.30, volume=4_000.0),
                Outcome(name="Other", price=0.25, volume=3_000.0),
            ],
            fetched_at=now,
        )
        pair = MarketPair(market_a=ma, market_b=mb, similarity_score=0.9)
        detector = DivergenceDetector(min_spread=0.03)
        divs = detector.detect([pair])
        assert len(divs) == 2  # Alice (0.05 spread), Bob (0.05 spread)

    def test_empty_pairs(self) -> None:
        detector = DivergenceDetector()
        assert detector.detect([]) == []

    def test_divergences_sorted_by_spread(self, now: datetime) -> None:
        markets_data = [
            (0.50, 0.42, "Small spread"),
            (0.70, 0.55, "Big spread"),
        ]
        pairs = []
        for i, (pa, pb, title) in enumerate(markets_data):
            ma = Market(
                id=f"pm-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=title,
                outcomes=[
                    Outcome(name="Yes", price=pa, volume=0.0),
                    Outcome(name="No", price=1.0 - pa, volume=0.0),
                ],
                fetched_at=now,
            )
            mb = Market(
                id=f"kx-{i}",
                exchange=ExchangeName.KALSHI,
                title=title,
                outcomes=[
                    Outcome(name="Yes", price=pb, volume=0.0),
                    Outcome(name="No", price=1.0 - pb, volume=0.0),
                ],
                fetched_at=now,
            )
            pairs.append(MarketPair(market_a=ma, market_b=mb, similarity_score=0.9))

        detector = DivergenceDetector(min_spread=0.01)
        divs = detector.detect(pairs)
        assert len(divs) == 2
        assert divs[0].spread > divs[1].spread


# ── Violation Detection ──────────────────────────────────────────────


class TestViolationDetector:
    """Tests for ViolationDetector."""

    def test_binary_violation_overpriced(
        self, binary_market_polymarket: Market
    ) -> None:
        # YES=0.35 + NO=0.68 = 1.03 > 1.0
        detector = ViolationDetector(binary_tolerance=0.02)
        violations = detector.detect_binary_violations([binary_market_polymarket])
        assert len(violations) == 1
        v = violations[0]
        assert v.price_sum == pytest.approx(1.03)
        assert v.implied_arb == pytest.approx(0.03)

    def test_no_violation_within_tolerance(self, now: datetime) -> None:
        market = Market(
            id="test",
            exchange=ExchangeName.POLYMARKET,
            title="Fair market",
            outcomes=[
                Outcome(name="Yes", price=0.52, volume=0.0),
                Outcome(name="No", price=0.49, volume=0.0),
            ],
            fetched_at=now,
        )
        detector = ViolationDetector(binary_tolerance=0.03)
        assert len(detector.detect_binary_violations([market])) == 0

    def test_multi_outcome_violation(self, multi_outcome_market: Market) -> None:
        # Sum = 0.22 + 0.31 + 0.15 + 0.08 + 0.28 = 1.04
        detector = ViolationDetector(multi_tolerance=0.03)
        violations = detector.detect_multi_outcome_violations([multi_outcome_market])
        assert len(violations) == 1
        assert violations[0].price_sum == pytest.approx(1.04)
        assert violations[0].deviation == pytest.approx(0.04)

    def test_detect_all_combined(
        self,
        binary_market_polymarket: Market,
        multi_outcome_market: Market,
    ) -> None:
        detector = ViolationDetector(binary_tolerance=0.02, multi_tolerance=0.03)
        binary_v, multi_v = detector.detect_all(
            [binary_market_polymarket, multi_outcome_market]
        )
        assert len(binary_v) == 1
        assert len(multi_v) == 1

    def test_skips_non_binary_for_binary_check(
        self, multi_outcome_market: Market
    ) -> None:
        detector = ViolationDetector()
        assert len(detector.detect_binary_violations([multi_outcome_market])) == 0

    def test_sorted_by_implied_arb(self, now: datetime) -> None:
        markets = [
            Market(
                id=f"m-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Market {i}",
                outcomes=[
                    Outcome(name="Yes", price=0.55 + i * 0.02, volume=0.0),
                    Outcome(name="No", price=0.55 + i * 0.02, volume=0.0),
                ],
                fetched_at=now,
            )
            for i in range(3)
        ]
        detector = ViolationDetector(binary_tolerance=0.05)
        violations = detector.detect_binary_violations(markets)
        if len(violations) >= 2:
            assert violations[0].implied_arb >= violations[1].implied_arb


# ── Liquidity Analysis ───────────────────────────────────────────────


class TestLiquidityAnalyzer:
    """Tests for LiquidityAnalyzer."""

    def test_analyze_with_order_book(
        self, order_book: OrderBook, now: datetime
    ) -> None:
        market = Market(
            id="test-liq",
            exchange=ExchangeName.KALSHI,
            title="Test liquidity",
            outcomes=[
                Outcome(
                    name="Yes",
                    price=0.73,
                    order_book=order_book,
                    volume=100_000.0,
                ),
                Outcome(name="No", price=0.27, volume=50_000.0),
            ],
            fetched_at=now,
        )
        analyzer = LiquidityAnalyzer()
        profile = analyzer.analyze(market)
        assert profile.best_bid == 0.72
        assert profile.best_ask == 0.74
        assert profile.spread == pytest.approx(0.02)
        assert profile.depth_at_1pct > 0
        assert profile.depth_at_5pct >= profile.depth_at_1pct
        assert len(profile.estimated_impact) == 5  # default trade sizes

    def test_analyze_without_order_book(self, now: datetime) -> None:
        market = Market(
            id="test-no-ob",
            exchange=ExchangeName.MANIFOLD,
            title="No order book",
            outcomes=[
                Outcome(name="Yes", price=0.65, volume=10_000.0),
                Outcome(name="No", price=0.35, volume=8_000.0),
            ],
            fetched_at=now,
        )
        analyzer = LiquidityAnalyzer()
        profile = analyzer.analyze(market)
        assert profile.best_bid is None
        assert profile.spread is None

    def test_impact_increases_with_size(self, order_book: OrderBook) -> None:
        analyzer = LiquidityAnalyzer(trade_sizes=[1_000, 10_000, 100_000])
        market = Market(
            id="test",
            exchange=ExchangeName.POLYMARKET,
            title="Impact test",
            outcomes=[
                Outcome(
                    name="Yes",
                    price=0.73,
                    order_book=order_book,
                    volume=0.0,
                ),
                Outcome(name="No", price=0.27, volume=0.0),
            ],
            fetched_at=datetime.now(UTC),
        )
        profile = analyzer.analyze(market)
        impacts = list(profile.estimated_impact.values())
        # Larger trades should have >= impact than smaller ones
        for i in range(1, len(impacts)):
            assert impacts[i] >= impacts[i - 1]

    def test_analyze_empty_book(self) -> None:
        book = OrderBook()
        analyzer = LiquidityAnalyzer()
        profile = analyzer.analyze_order_book(
            book, market_title="Test", market_id="x", exchange="kalshi"
        )
        assert profile.best_bid is None
        assert profile.estimated_impact == {s: 0.0 for s in analyzer.trade_sizes}


# ── Quality Scoring ──────────────────────────────────────────────────


class TestQualityScorer:
    """Tests for QualityScorer."""

    def test_score_resolved_markets(self, resolved_markets: list[Market]) -> None:
        scorer = QualityScorer(min_sample_size=5)
        quality = scorer.score(resolved_markets, category="Politics")
        assert 0.0 <= quality.brier_score <= 1.0
        assert 0.0 <= quality.calibration_error <= 1.0
        assert quality.sample_size == 12

    def test_insufficient_data_raises(self) -> None:
        scorer = QualityScorer(min_sample_size=100)
        with pytest.raises(InsufficientDataError):
            scorer.score([], category="all")

    def test_brier_score_perfect(self, now: datetime) -> None:
        """Markets that are perfectly calibrated should have low Brier score."""
        markets = [
            Market(
                id=f"perf-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Perfect {i}",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=1.0, volume=0.0),
                    Outcome(name="No", price=0.0, volume=0.0),
                ],
                resolution="yes",
                volume_total=10_000.0,
                fetched_at=now,
            )
            for i in range(10)
        ]
        scorer = QualityScorer(min_sample_size=5)
        quality = scorer.score(markets)
        assert quality.brier_score == pytest.approx(0.0)

    def test_manipulation_score(self, now: datetime) -> None:
        """Markets with extreme confidence that resolved wrong should flag."""
        markets = [
            Market(
                id=f"manip-{i}",
                exchange=ExchangeName.POLYMARKET,
                title=f"Suspicious {i}",
                contract_type=ContractType.BINARY,
                status=MarketStatus.RESOLVED,
                outcomes=[
                    Outcome(name="Yes", price=0.95, volume=0.0),
                    Outcome(name="No", price=0.05, volume=0.0),
                ],
                resolution="no",  # 95% confident YES, resolved NO
                volume_total=10_000.0,
                fetched_at=now,
            )
            for i in range(10)
        ]
        scorer = QualityScorer(min_sample_size=5)
        quality = scorer.score(markets)
        assert quality.manipulation_score > 0.5

    def test_filter_by_exchange(
        self, resolved_markets: list[Market], now: datetime
    ) -> None:
        # Add a Kalshi market
        kalshi_market = Market(
            id="kalshi-test",
            exchange=ExchangeName.KALSHI,
            title="Kalshi event",
            contract_type=ContractType.BINARY,
            status=MarketStatus.RESOLVED,
            outcomes=[
                Outcome(name="Yes", price=0.80, volume=0.0),
                Outcome(name="No", price=0.20, volume=0.0),
            ],
            resolution="yes",
            volume_total=50_000.0,
            fetched_at=now,
        )
        all_markets = [*resolved_markets, kalshi_market]
        scorer = QualityScorer(min_sample_size=1)
        quality = scorer.score(all_markets, exchange=ExchangeName.KALSHI)
        assert quality.sample_size == 1


# ── Efficiency Analysis ──────────────────────────────────────────────


class TestEfficiencyAnalyzer:
    """Tests for EfficiencyAnalyzer."""

    def test_arb_window_stats(self) -> None:
        analyzer = EfficiencyAnalyzer()
        divergences = [
            Divergence(
                event=f"Event {i}",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.50,
                price_b=0.50 - 0.01 * (i + 1),
                spread=0.01 * (i + 1),
                spread_pct=0.02 * (i + 1),
            )
            for i in range(10)
        ]
        stats = analyzer.compute_arb_window_stats(divergences)
        assert stats["mean"] > 0
        assert stats["median"] > 0
        assert stats["p95"] >= stats["median"]

    def test_arb_window_with_durations(self) -> None:
        analyzer = EfficiencyAnalyzer()
        durations = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = analyzer.compute_arb_window_stats([], durations)
        assert stats["mean"] == pytest.approx(30.0)
        assert stats["median"] == 30.0

    def test_arb_window_empty(self) -> None:
        analyzer = EfficiencyAnalyzer()
        stats = analyzer.compute_arb_window_stats([])
        assert stats["mean"] == 0.0

    def test_price_discovery_speed(self) -> None:
        analyzer = EfficiencyAnalyzer()
        base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        series = [
            (base, 0.50),
            (datetime(2026, 3, 15, 12, 10, 0, tzinfo=UTC), 0.70),
            (datetime(2026, 3, 15, 12, 20, 0, tzinfo=UTC), 0.90),
            (datetime(2026, 3, 15, 12, 30, 0, tzinfo=UTC), 0.95),
        ]
        speed = analyzer.compute_price_discovery_speed(series, final_price=1.0)
        assert speed > 0

    def test_price_discovery_empty(self) -> None:
        analyzer = EfficiencyAnalyzer()
        assert analyzer.compute_price_discovery_speed([], 1.0) == 0.0

    def test_information_ratio(self) -> None:
        analyzer = EfficiencyAnalyzer()
        # All positive changes -> high directional efficiency
        changes = [0.01, 0.02, 0.01, 0.03]
        intervals = [5.0, 5.0, 5.0, 5.0]
        ratio = analyzer.compute_information_ratio(changes, intervals)
        assert ratio > 0

    def test_information_ratio_empty(self) -> None:
        analyzer = EfficiencyAnalyzer()
        assert analyzer.compute_information_ratio([], []) == 0.0

    def test_analyze_market(self, binary_market_polymarket: Market) -> None:
        analyzer = EfficiencyAnalyzer()
        metrics = analyzer.analyze_market(binary_market_polymarket)
        assert metrics.market == binary_market_polymarket.title
        assert metrics.exchange == ExchangeName.POLYMARKET
