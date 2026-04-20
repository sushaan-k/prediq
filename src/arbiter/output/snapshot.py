"""Portable market snapshots and offline replay analysis.

Snapshots capture normalized markets exactly as arbiter saw them so research
workflows can be replayed without hitting live exchange APIs again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self, cast

from pydantic import BaseModel, Field

from arbiter.analytics.consensus import ConsensusAnalyzer
from arbiter.analytics.divergence import DivergenceDetector
from arbiter.analytics.violations import ViolationDetector
from arbiter.matching.semantic import SemanticMatcher
from arbiter.models import (
    Divergence,
    Market,
    MarketPair,
    MultiOutcomeViolation,
    ProbabilityViolation,
)


class MarketSnapshot(BaseModel):
    """A portable JSON snapshot of normalized markets grouped by exchange."""

    schema_version: int = Field(default=1, ge=1)
    generated_at: str = Field(description="UTC timestamp when the snapshot was built")
    markets: dict[str, list[Market]] = Field(default_factory=dict)

    @classmethod
    def from_markets(
        cls,
        markets: dict[str, list[Market]],
        *,
        generated_at: str,
    ) -> Self:
        """Build a snapshot from markets grouped by exchange name."""
        return cls(generated_at=generated_at, markets=markets)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Load and validate a snapshot JSON file."""
        return cls.model_validate_json(Path(path).read_text())

    def write(self, path: str | Path) -> Path:
        """Write the snapshot as stable, indented JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n")
        return output_path

    @property
    def exchange_counts(self) -> dict[str, int]:
        """Return market counts by exchange."""
        return {exchange: len(markets) for exchange, markets in self.markets.items()}

    @property
    def market_count(self) -> int:
        """Return the total number of markets in the snapshot."""
        return sum(self.exchange_counts.values())

    @property
    def all_markets(self) -> list[Market]:
        """Return all markets flattened into one list."""
        return [market for markets in self.markets.values() for market in markets]


class SnapshotAnalysis(BaseModel):
    """Offline analytics results computed from a market snapshot."""

    market_count: int
    exchange_counts: dict[str, int]
    pair_count: int
    divergences: list[Divergence]
    binary_violations: list[ProbabilityViolation]
    multi_outcome_violations: list[MultiOutcomeViolation]
    consensus: list[dict[str, object]]

    def to_markdown(self) -> str:
        """Render a compact Markdown report for human review."""
        lines = [
            "# Arbiter Snapshot Replay",
            "",
            "## Summary",
            "",
            f"- Markets: {self.market_count}",
            f"- Matched pairs: {self.pair_count}",
            f"- Divergences: {len(self.divergences)}",
            f"- Binary violations: {len(self.binary_violations)}",
            f"- Multi-outcome violations: {len(self.multi_outcome_violations)}",
            f"- Consensus disagreements: {len(self.consensus)}",
            "",
            "## Exchanges",
            "",
        ]
        for exchange, count in sorted(self.exchange_counts.items()):
            lines.append(f"- {exchange}: {count}")

        if self.divergences:
            lines.extend(["", "## Top Divergences", ""])
            for div in self.divergences[:10]:
                lines.append(
                    "- "
                    f"{div.event} ({div.outcome}): "
                    f"{div.exchange_a.value} {div.price_a:.1%} vs "
                    f"{div.exchange_b.value} {div.price_b:.1%} "
                    f"(spread {div.spread:.1%})"
                )

        if self.consensus:
            lines.extend(["", "## Top Consensus Disagreements", ""])
            for row in self.consensus[:10]:
                disagreement = cast(float, row["disagreement_band"])
                consensus_yes = cast(float, row["consensus_yes_price"])
                lines.append(
                    "- "
                    f"{row['event']}: consensus {consensus_yes:.1%}, "
                    f"disagreement {disagreement:.1%}"
                )

        return "\n".join(lines) + "\n"


class SnapshotAnalyzer:
    """Run arbiter analytics against an already captured market snapshot."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.6,
        min_spread: float = 0.02,
        min_liquidity: float = 0.0,
        min_disagreement: float = 0.05,
        consensus_limit: int | None = None,
    ) -> None:
        self._matcher = SemanticMatcher(similarity_threshold=similarity_threshold)
        self._divergence_detector = DivergenceDetector(
            min_spread=min_spread,
            min_liquidity=min_liquidity,
        )
        self._violation_detector = ViolationDetector()
        self._consensus_analyzer = ConsensusAnalyzer()
        self._min_disagreement = min_disagreement
        self._consensus_limit = consensus_limit

    def analyze(self, snapshot: MarketSnapshot) -> SnapshotAnalysis:
        """Analyze a snapshot without contacting exchanges."""
        pairs = self._match_markets(snapshot.markets)
        divergences = self._divergence_detector.detect(pairs)
        binary_violations, multi_violations = self._violation_detector.detect_all(
            snapshot.all_markets
        )
        consensus = self._consensus_analyzer.outliers(
            pairs,
            min_disagreement=self._min_disagreement,
            limit=self._consensus_limit,
        )

        return SnapshotAnalysis(
            market_count=snapshot.market_count,
            exchange_counts=snapshot.exchange_counts,
            pair_count=len(pairs),
            divergences=divergences,
            binary_violations=binary_violations,
            multi_outcome_violations=multi_violations,
            consensus=consensus,
        )

    def _match_markets(
        self, markets_by_exchange: dict[str, list[Market]]
    ) -> list[MarketPair]:
        exchange_names = list(markets_by_exchange)
        pairs: list[MarketPair] = []

        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                pairs.extend(
                    self._matcher.find_matches(
                        markets_by_exchange[exchange_names[i]],
                        markets_by_exchange[exchange_names[j]],
                    )
                )

        return pairs


def analyze_snapshot(
    snapshot: MarketSnapshot,
    *,
    similarity_threshold: float = 0.6,
    min_spread: float = 0.02,
    min_liquidity: float = 0.0,
    min_disagreement: float = 0.05,
    consensus_limit: int | None = None,
) -> SnapshotAnalysis:
    """Convenience wrapper for one-shot snapshot analysis."""
    analyzer = SnapshotAnalyzer(
        similarity_threshold=similarity_threshold,
        min_spread=min_spread,
        min_liquidity=min_liquidity,
        min_disagreement=min_disagreement,
        consensus_limit=consensus_limit,
    )
    return analyzer.analyze(snapshot)
