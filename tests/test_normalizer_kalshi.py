"""Tests for Kalshi price normalization in PriceNormalizer.

Kalshi uses cents (0-100) while Polymarket, Manifold, and Metaculus use
probability (0.0-1.0). These tests verify correct handling across the
full range of Kalshi price formats.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arbiter.matching.normalizer import PriceNormalizer
from arbiter.models import (
    ContractType,
    ExchangeName,
    Market,
    MarketStatus,
    Outcome,
)


class TestKalshiPriceNormalization:
    """Tests for Kalshi-specific price normalization."""

    def setup_method(self) -> None:
        self.normalizer = PriceNormalizer()

    def test_cents_above_one_normalized(self) -> None:
        """Kalshi price of 65 (cents) should become 0.65."""
        result = self.normalizer.normalize_price(65.0, ExchangeName.KALSHI)
        assert result == pytest.approx(0.65)

    def test_cents_at_99_normalized(self) -> None:
        """Kalshi price of 99 should become 0.99."""
        result = self.normalizer.normalize_price(99.0, ExchangeName.KALSHI)
        assert result == pytest.approx(0.99)

    def test_cents_at_1_stays_as_probability(self) -> None:
        """Kalshi price of 1.0 is ambiguous; treat as probability 1.0."""
        result = self.normalizer.normalize_price(1.0, ExchangeName.KALSHI)
        assert result == pytest.approx(1.0)

    def test_already_probability_unchanged(self) -> None:
        """Kalshi price already in 0-1 range stays unchanged."""
        result = self.normalizer.normalize_price(0.65, ExchangeName.KALSHI)
        assert result == pytest.approx(0.65)

    def test_zero_stays_zero(self) -> None:
        result = self.normalizer.normalize_price(0.0, ExchangeName.KALSHI)
        assert result == pytest.approx(0.0)

    def test_cents_at_50_normalized(self) -> None:
        """50 cents should become 0.50."""
        result = self.normalizer.normalize_price(50.0, ExchangeName.KALSHI)
        assert result == pytest.approx(0.50)

    def test_polymarket_not_affected(self) -> None:
        """Polymarket prices should never be divided by 100."""
        result = self.normalizer.normalize_price(0.65, ExchangeName.POLYMARKET)
        assert result == pytest.approx(0.65)

    def test_manifold_not_affected(self) -> None:
        result = self.normalizer.normalize_price(0.42, ExchangeName.MANIFOLD)
        assert result == pytest.approx(0.42)

    def test_normalize_kalshi_market(self) -> None:
        """Full market normalization with Kalshi cents-scale outcomes."""
        market = Market(
            id="KXTEST",
            exchange=ExchangeName.KALSHI,
            title="Test Kalshi Market",
            contract_type=ContractType.BINARY,
            status=MarketStatus.ACTIVE,
            outcomes=[
                Outcome(name="Yes", price=65.0 / 100.0, volume=0.0),
                Outcome(name="No", price=35.0 / 100.0, volume=0.0),
            ],
            fetched_at=datetime(2026, 3, 15, tzinfo=UTC),
        )
        normalized = self.normalizer.normalize_market(market)
        assert normalized.yes_price == pytest.approx(0.65)
        assert normalized.no_price == pytest.approx(0.35)

    def test_normalize_market_preserves_exchange(self) -> None:
        """Normalization should not change the exchange field."""
        market = Market(
            id="KX1",
            exchange=ExchangeName.KALSHI,
            title="Kalshi Test",
            outcomes=[
                Outcome(name="Yes", price=0.70, volume=0.0),
                Outcome(name="No", price=0.30, volume=0.0),
            ],
            fetched_at=datetime(2026, 3, 15, tzinfo=UTC),
        )
        normalized = self.normalizer.normalize_market(market)
        assert normalized.exchange == ExchangeName.KALSHI

    def test_is_cents_scale_boundary_values(self) -> None:
        """Test the cents-scale detection heuristic directly."""
        assert PriceNormalizer._is_cents_scale(50.0) is True
        assert PriceNormalizer._is_cents_scale(100.0) is True
        assert PriceNormalizer._is_cents_scale(1.5) is True
        assert PriceNormalizer._is_cents_scale(0.5) is False
        assert PriceNormalizer._is_cents_scale(0.0) is False
        assert PriceNormalizer._is_cents_scale(0.99) is False
