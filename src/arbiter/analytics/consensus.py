"""Cross-exchange consensus pricing analytics."""

from __future__ import annotations

from statistics import mean
from typing import cast

from arbiter.models import MarketPair


class ConsensusAnalyzer:
    """Compute consensus prices and disagreement bands for matched markets."""

    def summarize_pair(self, pair: MarketPair) -> dict[str, object]:
        """Return a consensus summary for a matched binary market pair."""
        market_a = pair.market_a
        market_b = pair.market_b
        price_a = market_a.yes_price
        price_b = market_b.yes_price
        if price_a is None or price_b is None:
            raise ValueError("Consensus analysis requires binary YES prices")

        liquidity_a = max(market_a.volume_total, 1.0)
        liquidity_b = max(market_b.volume_total, 1.0)
        total_liquidity = liquidity_a + liquidity_b
        weighted_yes = (
            (price_a * liquidity_a) + (price_b * liquidity_b)
        ) / total_liquidity
        disagreement = abs(price_a - price_b)

        return {
            "event": market_a.title,
            "consensus_yes_price": round(weighted_yes, 4),
            "simple_average_yes_price": round(mean([price_a, price_b]), 4),
            "disagreement_band": round(disagreement, 4),
            "exchange_count": 2,
            "total_liquidity": round(total_liquidity, 2),
        }

    def summarize_pairs(self, pairs: list[MarketPair]) -> list[dict[str, object]]:
        """Return consensus summaries for all binary pairs."""
        summaries: list[dict[str, object]] = []
        for pair in pairs:
            if pair.market_a.yes_price is None or pair.market_b.yes_price is None:
                continue
            summaries.append(self.summarize_pair(pair))
        return summaries

    def outliers(
        self,
        pairs: list[MarketPair],
        *,
        min_disagreement: float = 0.05,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        """Return consensus summaries with large cross-exchange disagreement."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        summaries = [
            summary
            for summary in self.summarize_pairs(pairs)
            if cast(float, summary["disagreement_band"]) >= min_disagreement
        ]
        summaries.sort(
            key=lambda item: cast(float, item["disagreement_band"]), reverse=True
        )
        return summaries[:limit] if limit is not None else summaries
