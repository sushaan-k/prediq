"""Tests for arbiter.matching -- semantic matching and normalization."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arbiter.matching.normalizer import PriceNormalizer
from arbiter.matching.semantic import SemanticMatcher, _normalize_text, _token_overlap
from arbiter.models import (
    ExchangeName,
    Market,
    Outcome,
)


class TestNormalizeText:
    """Tests for the _normalize_text helper."""

    def test_lowercase(self) -> None:
        assert _normalize_text("HELLO") == "hello"

    def test_strips_punctuation(self) -> None:
        assert _normalize_text("Will BTC hit $200K?") == "will btc hit 200k"

    def test_collapses_whitespace(self) -> None:
        assert _normalize_text("  foo   bar  ") == "foo bar"


class TestTokenOverlap:
    """Tests for Jaccard token overlap."""

    def test_identical(self) -> None:
        assert _token_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert _token_overlap("cat dog", "fish bird") == 0.0

    def test_partial_overlap(self) -> None:
        score = _token_overlap("Will Bitcoin hit 200K", "Bitcoin to reach 200000")
        assert 0.0 < score < 1.0

    def test_empty_strings(self) -> None:
        assert _token_overlap("", "hello") == 0.0
        assert _token_overlap("", "") == 0.0


class TestSemanticMatcher:
    """Tests for the SemanticMatcher class."""

    def test_identical_titles_match(self) -> None:
        matcher = SemanticMatcher(similarity_threshold=0.5)
        score = matcher.similarity(
            "Will Bitcoin hit $200K?",
            "Will Bitcoin hit $200K?",
        )
        assert score > 0.8

    def test_similar_titles_match(self) -> None:
        matcher = SemanticMatcher(similarity_threshold=0.4)
        score = matcher.similarity(
            "Will Bitcoin hit $200K by December 2026?",
            "Bitcoin to reach $200,000 by end of 2026?",
        )
        assert score > 0.3

    def test_unrelated_titles_no_match(self) -> None:
        matcher = SemanticMatcher(similarity_threshold=0.6)
        score = matcher.similarity(
            "Will Bitcoin hit $200K?",
            "Who will win the Super Bowl?",
        )
        assert score < 0.3

    def test_find_matches(
        self,
        binary_market_polymarket: Market,
        binary_market_kalshi: Market,
    ) -> None:
        matcher = SemanticMatcher(similarity_threshold=0.3)
        pairs = matcher.find_matches(
            [binary_market_polymarket],
            [binary_market_kalshi],
        )
        assert len(pairs) == 1
        assert pairs[0].market_a.id == binary_market_polymarket.id
        assert pairs[0].market_b.id == binary_market_kalshi.id

    def test_find_matches_threshold_blocks(self) -> None:
        """With a very high threshold, dissimilar markets should not match."""
        matcher = SemanticMatcher(similarity_threshold=0.99)
        now = datetime(2026, 1, 1, tzinfo=UTC)
        m1 = Market(
            id="a",
            exchange=ExchangeName.POLYMARKET,
            title="Will it rain tomorrow?",
            outcomes=[
                Outcome(name="Yes", price=0.5, volume=0.0),
                Outcome(name="No", price=0.5, volume=0.0),
            ],
            fetched_at=now,
        )
        m2 = Market(
            id="b",
            exchange=ExchangeName.KALSHI,
            title="Who will win the election?",
            outcomes=[
                Outcome(name="Yes", price=0.5, volume=0.0),
                Outcome(name="No", price=0.5, volume=0.0),
            ],
            fetched_at=now,
        )
        pairs = matcher.find_matches([m1], [m2])
        assert len(pairs) == 0

    def test_same_exchange_not_matched(self) -> None:
        """Markets from the same exchange should not match."""
        matcher = SemanticMatcher(similarity_threshold=0.0)
        now = datetime(2026, 1, 1, tzinfo=UTC)
        m1 = Market(
            id="a",
            exchange=ExchangeName.POLYMARKET,
            title="Test market",
            outcomes=[
                Outcome(name="Yes", price=0.5, volume=0.0),
                Outcome(name="No", price=0.5, volume=0.0),
            ],
            fetched_at=now,
        )
        m2 = Market(
            id="b",
            exchange=ExchangeName.POLYMARKET,
            title="Test market",
            outcomes=[
                Outcome(name="Yes", price=0.5, volume=0.0),
                Outcome(name="No", price=0.5, volume=0.0),
            ],
            fetched_at=now,
        )
        pairs = matcher.find_matches([m1], [m2])
        assert len(pairs) == 0


class TestPriceNormalizer:
    """Tests for the PriceNormalizer class."""

    def test_kalshi_cents_to_decimal(self) -> None:
        normalizer = PriceNormalizer()
        assert normalizer.normalize_price(72.0, ExchangeName.KALSHI) == pytest.approx(
            0.72
        )

    def test_kalshi_already_normalized(self) -> None:
        normalizer = PriceNormalizer()
        assert normalizer.normalize_price(0.72, ExchangeName.KALSHI) == pytest.approx(
            0.72
        )

    def test_polymarket_passthrough(self) -> None:
        normalizer = PriceNormalizer()
        assert normalizer.normalize_price(
            0.65, ExchangeName.POLYMARKET
        ) == pytest.approx(0.65)

    def test_clamp_to_bounds(self) -> None:
        normalizer = PriceNormalizer()
        assert normalizer.normalize_price(-0.1, ExchangeName.POLYMARKET) == 0.0
        assert normalizer.normalize_price(1.5, ExchangeName.POLYMARKET) == 1.0

    def test_normalize_market(self, binary_market_polymarket: Market) -> None:
        normalizer = PriceNormalizer()
        normalized = normalizer.normalize_market(binary_market_polymarket)
        assert normalized.outcomes[0].price == 0.35
        assert normalized.fetched_at.tzinfo is not None

    def test_estimate_fee(self) -> None:
        normalizer = PriceNormalizer()
        fee = normalizer.estimate_fee(ExchangeName.POLYMARKET, 10_000.0)
        assert fee == pytest.approx(200.0)

        fee_k = normalizer.estimate_fee(ExchangeName.KALSHI, 10_000.0)
        assert fee_k == pytest.approx(100.0)

    def test_clean_title(self) -> None:
        assert PriceNormalizer.clean_title("  Hello, World! ") == "hello world"
