"""Extended tests for arbiter.matching -- semantic edge cases and cross-exchange pairing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arbiter.matching.semantic import SemanticMatcher, _normalize_text, _token_overlap
from arbiter.models import (
    ExchangeName,
    Market,
    Outcome,
)


class TestSemanticMatcherEdgeCases:
    """Edge cases for semantic matching."""

    def test_empty_titles(self) -> None:
        matcher = SemanticMatcher(similarity_threshold=0.1)
        score = matcher.similarity("", "")
        assert score >= 0.0

    def test_very_short_titles(self) -> None:
        matcher = SemanticMatcher(similarity_threshold=0.5)
        score = matcher.similarity("rain", "rain")
        assert score > 0.8

    def test_completely_different_titles(self) -> None:
        matcher = SemanticMatcher(similarity_threshold=0.5)
        score = matcher.similarity(
            "Will it rain in Seattle tomorrow?",
            "Global semiconductor production forecast Q4 2027",
        )
        assert score < 0.3

    def test_rephrased_titles(self) -> None:
        matcher = SemanticMatcher(similarity_threshold=0.3)
        score = matcher.similarity(
            "Will the Federal Reserve cut interest rates in March 2026?",
            "Fed rate decision March 2026 - will rates be lowered?",
        )
        assert score > 0.2

    def test_find_matches_multiple_pairs(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        matcher = SemanticMatcher(similarity_threshold=0.3)

        markets_a = [
            Market(
                id="pm-1",
                exchange=ExchangeName.POLYMARKET,
                title="Will Bitcoin hit $200K by Dec 2026?",
                outcomes=[
                    Outcome(name="Yes", price=0.35, volume=0.0),
                    Outcome(name="No", price=0.65, volume=0.0),
                ],
                fetched_at=now,
            ),
            Market(
                id="pm-2",
                exchange=ExchangeName.POLYMARKET,
                title="Will it rain in NYC tomorrow?",
                outcomes=[
                    Outcome(name="Yes", price=0.70, volume=0.0),
                    Outcome(name="No", price=0.30, volume=0.0),
                ],
                fetched_at=now,
            ),
        ]
        markets_b = [
            Market(
                id="kx-1",
                exchange=ExchangeName.KALSHI,
                title="Bitcoin to reach $200,000 by end of 2026?",
                outcomes=[
                    Outcome(name="Yes", price=0.30, volume=0.0),
                    Outcome(name="No", price=0.70, volume=0.0),
                ],
                fetched_at=now,
            ),
            Market(
                id="kx-2",
                exchange=ExchangeName.KALSHI,
                title="Rain in New York City tomorrow?",
                outcomes=[
                    Outcome(name="Yes", price=0.65, volume=0.0),
                    Outcome(name="No", price=0.35, volume=0.0),
                ],
                fetched_at=now,
            ),
        ]

        pairs = matcher.find_matches(markets_a, markets_b)
        assert len(pairs) >= 1

    def test_find_matches_no_double_matching(self) -> None:
        """Each market_b should only be matched once."""
        now = datetime(2026, 1, 1, tzinfo=UTC)
        matcher = SemanticMatcher(similarity_threshold=0.1)

        markets_a = [
            Market(
                id=f"pm-{i}",
                exchange=ExchangeName.POLYMARKET,
                title="Will Bitcoin hit $200K?",
                outcomes=[
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ],
                fetched_at=now,
            )
            for i in range(3)
        ]
        markets_b = [
            Market(
                id="kx-1",
                exchange=ExchangeName.KALSHI,
                title="Bitcoin to reach $200K?",
                outcomes=[
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ],
                fetched_at=now,
            )
        ]

        pairs = matcher.find_matches(markets_a, markets_b)
        # Only one pair should be returned since there's only one market_b
        assert len(pairs) <= 1

    def test_find_matches_empty_lists(self) -> None:
        matcher = SemanticMatcher()
        assert matcher.find_matches([], []) == []

    def test_find_matches_empty_a(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        matcher = SemanticMatcher()
        m = Market(
            id="x",
            exchange=ExchangeName.KALSHI,
            title="Test",
            outcomes=[
                Outcome(name="Yes", price=0.5, volume=0.0),
                Outcome(name="No", price=0.5, volume=0.0),
            ],
            fetched_at=now,
        )
        assert matcher.find_matches([], [m]) == []

    def test_embedding_similarity_fallback(self) -> None:
        """When use_embeddings=True but sentence-transformers is missing."""
        matcher = SemanticMatcher(use_embeddings=True)
        # Should fall back to fuzzy matching
        score = matcher.similarity("test text", "test text")
        assert score > 0.5

    def test_fuzzy_similarity_identical(self) -> None:
        matcher = SemanticMatcher()
        score = matcher._fuzzy_similarity("hello world", "hello world")
        assert score == pytest.approx(1.0)

    def test_fuzzy_similarity_different(self) -> None:
        matcher = SemanticMatcher()
        score = matcher._fuzzy_similarity("apple banana cherry", "dog elephant fish")
        assert score < 0.3


class TestTokenOverlapEdgeCases:
    def test_single_word_match(self) -> None:
        score = _token_overlap("hello", "hello")
        assert score == pytest.approx(1.0)

    def test_superset(self) -> None:
        score = _token_overlap("hello world foo", "hello world")
        assert 0.5 < score < 1.0

    def test_unicode_text(self) -> None:
        score = _token_overlap("café résumé", "cafe resume")
        # Different characters, low overlap
        assert isinstance(score, float)


class TestNormalizeTextEdgeCases:
    def test_numbers_preserved(self) -> None:
        assert _normalize_text("BTC $200K") == "btc 200k"

    def test_multiple_spaces(self) -> None:
        assert _normalize_text("a    b    c") == "a b c"

    def test_only_punctuation(self) -> None:
        result = _normalize_text("!@#$%")
        assert result == ""
