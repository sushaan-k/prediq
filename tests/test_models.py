"""Tests for arbiter.models -- Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arbiter.models import (
    Alert,
    Divergence,
    EfficiencyMetrics,
    ExchangeConfig,
    ExchangeName,
    LiquidityProfile,
    Market,
    MarketPair,
    MarketQuality,
    MarketStatus,
    MultiOutcomeViolation,
    OrderBook,
    OrderBookLevel,
    Outcome,
    ProbabilityViolation,
    Side,
)


class TestOrderBookLevel:
    """Tests for OrderBookLevel model."""

    def test_valid_level(self) -> None:
        level = OrderBookLevel(price=0.72, quantity=5000.0)
        assert level.price == 0.72
        assert level.quantity == 5000.0

    def test_price_bounds(self) -> None:
        OrderBookLevel(price=0.0, quantity=1.0)
        OrderBookLevel(price=1.0, quantity=1.0)

        with pytest.raises(ValidationError):
            OrderBookLevel(price=-0.01, quantity=1.0)
        with pytest.raises(ValidationError):
            OrderBookLevel(price=1.01, quantity=1.0)

    def test_quantity_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            OrderBookLevel(price=0.5, quantity=0.0)
        with pytest.raises(ValidationError):
            OrderBookLevel(price=0.5, quantity=-1.0)

    def test_frozen(self) -> None:
        level = OrderBookLevel(price=0.5, quantity=100.0)
        with pytest.raises(ValidationError):
            level.price = 0.6  # type: ignore[misc]


class TestOrderBook:
    """Tests for OrderBook model."""

    def test_empty_book(self) -> None:
        book = OrderBook()
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.spread is None
        assert book.midpoint is None

    def test_with_levels(self, order_book: OrderBook) -> None:
        assert order_book.best_bid == 0.72
        assert order_book.best_ask == 0.74
        assert order_book.spread == pytest.approx(0.02)
        assert order_book.midpoint == pytest.approx(0.73)

    def test_one_side_only(self) -> None:
        book = OrderBook(
            bids=[OrderBookLevel(price=0.50, quantity=100.0)],
        )
        assert book.best_bid == 0.50
        assert book.best_ask is None
        assert book.spread is None
        assert book.midpoint is None


class TestOutcome:
    """Tests for Outcome model."""

    def test_creation(self) -> None:
        o = Outcome(name="Yes", price=0.65, volume=10_000.0)
        assert o.name == "Yes"
        assert o.price == 0.65
        assert o.volume == 10_000.0
        assert o.order_book is None

    def test_price_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Outcome(name="Yes", price=1.5, volume=0.0)


class TestMarket:
    """Tests for Market model."""

    def test_binary_market_prices(self, binary_market_polymarket: Market) -> None:
        m = binary_market_polymarket
        assert m.yes_price == 0.35
        assert m.no_price == 0.68

    def test_multi_outcome_no_yes_no(self, multi_outcome_market: Market) -> None:
        assert multi_outcome_market.yes_price is None
        assert multi_outcome_market.no_price is None

    def test_minimum_two_outcomes(self) -> None:
        with pytest.raises(ValidationError):
            Market(
                id="bad",
                exchange=ExchangeName.POLYMARKET,
                title="Bad market",
                outcomes=[Outcome(name="Yes", price=0.5, volume=0.0)],
            )

    def test_exchange_name_enum(self) -> None:
        assert ExchangeName.POLYMARKET.value == "polymarket"
        assert ExchangeName.KALSHI.value == "kalshi"

    def test_market_status_enum(self) -> None:
        assert MarketStatus.ACTIVE.value == "active"
        assert MarketStatus.RESOLVED.value == "resolved"


class TestMarketPair:
    """Tests for MarketPair model."""

    def test_creation(self, market_pair: MarketPair) -> None:
        assert market_pair.similarity_score == 0.85
        assert market_pair.market_a.exchange == ExchangeName.POLYMARKET
        assert market_pair.market_b.exchange == ExchangeName.KALSHI

    def test_similarity_bounds(
        self,
        binary_market_polymarket: Market,
        binary_market_kalshi: Market,
    ) -> None:
        with pytest.raises(ValidationError):
            MarketPair(
                market_a=binary_market_polymarket,
                market_b=binary_market_kalshi,
                similarity_score=1.5,
            )


class TestDivergence:
    """Tests for Divergence model."""

    def test_creation(self) -> None:
        d = Divergence(
            event="Test Event",
            outcome="Yes",
            exchange_a=ExchangeName.POLYMARKET,
            exchange_b=ExchangeName.KALSHI,
            price_a=0.35,
            price_b=0.30,
            spread=0.05,
            spread_pct=0.1538,
            liquidity_a=100_000.0,
            liquidity_b=50_000.0,
            net_arb_profit_estimate=1_200.0,
        )
        assert d.spread == 0.05
        assert d.exchange_a == ExchangeName.POLYMARKET

    def test_serialization_roundtrip(self) -> None:
        d = Divergence(
            event="Test",
            outcome="Yes",
            exchange_a=ExchangeName.POLYMARKET,
            exchange_b=ExchangeName.KALSHI,
            price_a=0.50,
            price_b=0.45,
            spread=0.05,
            spread_pct=0.1053,
        )
        data = d.model_dump(mode="json")
        restored = Divergence.model_validate(data)
        assert restored.spread == d.spread


class TestProbabilityViolation:
    """Tests for ProbabilityViolation model."""

    def test_overpriced_violation(self) -> None:
        v = ProbabilityViolation(
            market="Test",
            market_id="test-1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.55,
            no_price=0.52,
            price_sum=1.07,
            implied_arb=0.07,
            volume_available=50_000.0,
        )
        assert v.price_sum > 1.0
        assert v.implied_arb == 0.07


class TestMultiOutcomeViolation:
    """Tests for MultiOutcomeViolation model."""

    def test_creation(self) -> None:
        v = MultiOutcomeViolation(
            market="Election",
            market_id="elec-1",
            exchange=ExchangeName.KALSHI,
            outcomes={"A": 0.30, "B": 0.40, "C": 0.35},
            price_sum=1.05,
            deviation=0.05,
        )
        assert len(v.outcomes) == 3
        assert v.deviation == 0.05


class TestLiquidityProfile:
    """Tests for LiquidityProfile model."""

    def test_empty_profile(self) -> None:
        lp = LiquidityProfile(
            market="Test",
            market_id="test-1",
            exchange=ExchangeName.KALSHI,
        )
        assert lp.best_bid is None
        assert lp.spread is None
        assert lp.estimated_impact == {}


class TestMarketQuality:
    """Tests for MarketQuality model."""

    def test_score_bounds(self) -> None:
        mq = MarketQuality(
            exchange=ExchangeName.POLYMARKET,
            brier_score=0.18,
            calibration_error=0.03,
            manipulation_score=0.12,
            volume_accuracy_correlation=0.67,
            sample_size=100,
        )
        assert 0.0 <= mq.brier_score <= 2.0
        assert 0.0 <= mq.calibration_error <= 1.0


class TestEfficiencyMetrics:
    """Tests for EfficiencyMetrics model."""

    def test_creation(self) -> None:
        em = EfficiencyMetrics(
            market="Test",
            market_id="test-1",
            exchange=ExchangeName.POLYMARKET,
            price_discovery_speed_minutes=15.0,
            avg_arb_window_minutes=45.0,
            information_ratio=0.85,
        )
        assert em.price_discovery_speed_minutes == 15.0


class TestAlert:
    """Tests for Alert model."""

    def test_creation(self) -> None:
        a = Alert(
            alert_type="divergence",
            severity="high",
            message="Big spread detected",
        )
        assert a.alert_type == "divergence"
        assert a.data == {}


class TestExchangeConfig:
    """Tests for ExchangeConfig model."""

    def test_defaults(self) -> None:
        cfg = ExchangeConfig(name=ExchangeName.POLYMARKET)
        assert cfg.api_key is None
        assert cfg.rate_limit_per_second == 5.0
        assert cfg.enabled is True

    def test_rate_limit_positive(self) -> None:
        with pytest.raises(ValidationError):
            ExchangeConfig(
                name=ExchangeName.KALSHI,
                rate_limit_per_second=0.0,
            )


class TestSideEnum:
    """Tests for Side enum."""

    def test_values(self) -> None:
        assert Side.BID.value == "bid"
        assert Side.ASK.value == "ask"
