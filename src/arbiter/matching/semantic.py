"""Semantic market matching across exchanges.

Uses text similarity to identify when two markets on different exchanges
refer to the same real-world event. Supports both simple fuzzy matching
(no ML dependencies) and optional sentence-transformer embeddings.

Note: For significantly better matching accuracy, install the ``[matching]``
extra (``pip install arbiter[matching]``) which provides sentence-transformer
embedding-based similarity.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from importlib import import_module
from typing import Any, cast

from arbiter.models import Market, MarketPair

logger = logging.getLogger(__name__)

# Synonyms are expanded *before* tokenization so that both sides of a
# comparison use the same canonical form.  Keys should be lowercase.
_SYNONYMS: dict[str, str] = {
    "100k": "100000",
    "200k": "200000",
    "500k": "500000",
    "1m": "1000000",
    "1b": "1000000000",
    "btc": "bitcoin",
    "eth": "ethereum",
    "wins": "win",
    "winning": "win",
    "won": "win",
    "elect": "election",
    "elected": "election",
    "pres": "president",
    "presidential": "president",
    "gov": "governor",
    "dem": "democrat",
    "gop": "republican",
    "rep": "republican",
}


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _expand_synonyms(text: str) -> str:
    """Replace known synonyms in *text* with their canonical forms."""
    tokens = text.split()
    expanded = [_SYNONYMS.get(t, t) for t in tokens]
    return " ".join(expanded)


def _token_overlap(a: str, b: str) -> float:
    """Compute Jaccard similarity between word token sets.

    Synonym expansion is applied before tokenisation so that semantically
    equivalent terms are treated as identical.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Jaccard similarity in [0, 1].
    """
    norm_a = _expand_synonyms(_normalize_text(a))
    norm_b = _expand_synonyms(_normalize_text(b))
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


class SemanticMatcher:
    """Matches equivalent markets across different exchanges.

    Uses a combination of fuzzy string matching and token overlap.
    When sentence-transformers is installed, can optionally use
    embedding-based similarity for higher accuracy.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        use_embeddings: bool = False,
    ) -> None:
        """Initialize the matcher.

        Args:
            similarity_threshold: Minimum similarity score to consider
                two markets as matching. Range [0, 1].  The default of
                0.4 works well for fuzzy matching; when using embeddings
                a higher threshold (e.g. 0.6) is recommended.
            use_embeddings: If True, attempt to use sentence-transformers
                for embedding-based matching. Falls back to fuzzy if unavailable.
        """
        self.threshold = similarity_threshold
        self._encoder = None

        if use_embeddings:
            try:
                sentence_transformers = cast(
                    Any, import_module("sentence_transformers")
                )
                self._encoder = sentence_transformers.SentenceTransformer(
                    "all-MiniLM-L6-v2"
                )
                logger.info("Loaded sentence-transformer for semantic matching")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using fuzzy matching. Install with: pip install arbiter[matching]"
                )

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two market titles.

        Uses embedding cosine similarity if available, otherwise
        combines SequenceMatcher ratio with token overlap.

        Args:
            text_a: First market title.
            text_b: Second market title.

        Returns:
            Similarity score in [0, 1].
        """
        if self._encoder is not None:
            return self._embedding_similarity(text_a, text_b)
        return self._fuzzy_similarity(text_a, text_b)

    def _fuzzy_similarity(self, text_a: str, text_b: str) -> float:
        """Compute fuzzy similarity using string-level heuristics.

        Combines SequenceMatcher ratio (edit distance based) with
        Jaccard token overlap for better matching across different
        phrasings of the same question.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Blended similarity score in [0, 1].
        """
        norm_a = _normalize_text(text_a)
        norm_b = _normalize_text(text_b)

        seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
        token_sim = _token_overlap(text_a, text_b)

        # Weighted blend: sequence matching captures word order,
        # token overlap captures content regardless of phrasing
        return 0.4 * seq_ratio + 0.6 * token_sim

    def _embedding_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity using sentence embeddings.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Cosine similarity in [0, 1].
        """
        if self._encoder is None:
            return self._fuzzy_similarity(text_a, text_b)

        embeddings = self._encoder.encode([text_a, text_b])
        # Cosine similarity
        dot = sum(a * b for a, b in zip(embeddings[0], embeddings[1], strict=True))
        norm_a = sum(a * a for a in embeddings[0]) ** 0.5
        norm_b = sum(b * b for b in embeddings[1]) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        sim = dot / (norm_a * norm_b)
        return float(max(0.0, min(1.0, sim)))

    def find_matches(
        self,
        markets_a: list[Market],
        markets_b: list[Market],
    ) -> list[MarketPair]:
        """Find matching markets between two sets from different exchanges.

        For each market in set A, finds the best matching market in set B
        above the similarity threshold.

        Args:
            markets_a: Markets from exchange A.
            markets_b: Markets from exchange B.

        Returns:
            List of MarketPair objects for matched markets.
        """
        pairs: list[MarketPair] = []
        used_b: set[str] = set()

        for ma in markets_a:
            best_score = 0.0
            best_match: Market | None = None

            for mb in markets_b:
                if mb.id in used_b:
                    continue
                if ma.exchange == mb.exchange:
                    continue

                score = self.similarity(ma.title, mb.title)
                if score > best_score:
                    best_score = score
                    best_match = mb

            if best_match is not None and best_score >= self.threshold:
                used_b.add(best_match.id)
                pairs.append(
                    MarketPair(
                        market_a=ma,
                        market_b=best_match,
                        similarity_score=best_score,
                    )
                )
                logger.debug(
                    "Matched: '%s' <-> '%s' (score=%.3f)",
                    ma.title,
                    best_match.title,
                    best_score,
                )

        return pairs
