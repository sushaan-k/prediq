"""Main orchestration engine for arbiter.

The Arbiter class is the primary entry point. It coordinates exchange
connectors, market matching, analytics engines, and output to provide
a unified interface for cross-exchange prediction market analysis.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from arbiter.analytics.divergence import DivergenceDetector
from arbiter.analytics.efficiency import EfficiencyAnalyzer
from arbiter.analytics.liquidity import LiquidityAnalyzer
from arbiter.analytics.quality import QualityScorer
from arbiter.analytics.violations import ViolationDetector
from arbiter.exceptions import ArbiterError, ConfigError
from arbiter.exchanges.base import BaseExchange
from arbiter.matching.normalizer import PriceNormalizer
from arbiter.matching.semantic import SemanticMatcher
from arbiter.models import (
    Alert,
    Divergence,
    ExchangeName,
    LiquidityProfile,
    Market,
    MarketPair,
    MarketQuality,
    MultiOutcomeViolation,
    ProbabilityViolation,
)
from arbiter.output.alerts import AlertManager
from arbiter.output.export import DataExporter
from arbiter.storage import Storage

logger = logging.getLogger(__name__)


class Arbiter:
    """Cross-exchange prediction market analytics engine.

    The main orchestrator that ties together exchange connectors,
    market matching, analytics, and output. Designed for both
    interactive use and long-running monitoring.

    Example::

        from arbiter import Arbiter
        from arbiter.exchanges import PolymarketExchange, KalshiExchange

        async with Arbiter(exchanges=[
            PolymarketExchange(),
            KalshiExchange(api_key="..."),
        ]) as arb:
            divergences = await arb.divergences(min_spread=0.03)
            for d in divergences:
                print(f"{d.event}: {d.spread_pct:.1%} spread")
    """

    def __init__(
        self,
        exchanges: list[BaseExchange] | None = None,
        similarity_threshold: float = 0.6,
        storage_path: str | None = None,
    ) -> None:
        """Initialize the arbiter engine.

        Args:
            exchanges: List of exchange connector instances.
            similarity_threshold: Minimum similarity for market matching.
            storage_path: Path for DuckDB storage. None for in-memory.
        """
        self._exchanges: list[BaseExchange] = exchanges or []
        self._matcher = SemanticMatcher(similarity_threshold=similarity_threshold)
        self._normalizer = PriceNormalizer()
        self._divergence_detector = DivergenceDetector()
        self._violation_detector = ViolationDetector()
        self._liquidity_analyzer = LiquidityAnalyzer()
        self._quality_scorer = QualityScorer()
        self._efficiency_analyzer = EfficiencyAnalyzer()
        self._alert_manager = AlertManager()
        self._exporter = DataExporter()
        self._storage = Storage(storage_path)
        self._market_cache: dict[str, list[Market]] = {}

    def add_exchange(self, exchange: BaseExchange) -> None:
        """Add an exchange connector.

        Args:
            exchange: Exchange connector instance.
        """
        self._exchanges.append(exchange)
        logger.info("Added exchange: %s", exchange.name.value)

    async def fetch_all_markets(
        self, active_only: bool = True, limit: int = 100
    ) -> dict[str, list[Market]]:
        """Fetch markets from all configured exchanges in parallel.

        Args:
            active_only: Only fetch active markets.
            limit: Per-exchange market limit.

        Returns:
            Dictionary mapping exchange name -> list of markets.
        """
        if not self._exchanges:
            raise ConfigError("exchanges", "No exchanges configured")

        tasks = [
            self._fetch_exchange_markets(ex, active_only, limit)
            for ex in self._exchanges
        ]
        results: list[list[Market] | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        all_markets: dict[str, list[Market]] = {}
        for exchange, result in zip(self._exchanges, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "Failed to fetch from %s: %s",
                    exchange.name.value,
                    result,
                )
                all_markets[exchange.name.value] = []
            else:
                all_markets[exchange.name.value] = result

        self._market_cache = all_markets

        # Persist to storage
        for markets in all_markets.values():
            if markets:
                try:
                    self._storage.insert_markets(markets)
                except Exception:
                    logger.debug("Failed to persist markets to storage", exc_info=True)

        return all_markets

    async def _fetch_exchange_markets(
        self,
        exchange: BaseExchange,
        active_only: bool,
        limit: int,
    ) -> list[Market]:
        """Fetch and normalize markets from a single exchange.

        Args:
            exchange: The exchange connector.
            active_only: Only active markets.
            limit: Maximum markets.

        Returns:
            List of normalized markets.
        """
        raw_markets = await exchange.fetch_markets(active_only=active_only, limit=limit)
        return [self._normalizer.normalize_market(m) for m in raw_markets]

    def match_markets(
        self,
        markets_by_exchange: dict[str, list[Market]] | None = None,
    ) -> list[MarketPair]:
        """Find matching markets across all exchange pairs.

        Args:
            markets_by_exchange: Exchange name -> markets mapping.
                Uses cached data from fetch_all_markets if None.

        Returns:
            List of matched market pairs.
        """
        data = markets_by_exchange or self._market_cache
        exchange_names = list(data.keys())
        all_pairs: list[MarketPair] = []

        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                markets_a = data[exchange_names[i]]
                markets_b = data[exchange_names[j]]
                pairs = self._matcher.find_matches(markets_a, markets_b)
                all_pairs.extend(pairs)

        logger.info("Found %d matched market pairs", len(all_pairs))
        return all_pairs

    async def divergences(
        self,
        min_spread: float = 0.02,
        min_liquidity: float = 0.0,
        limit: int = 100,
    ) -> list[Divergence]:
        """Find current cross-exchange price divergences.

        Fetches markets from all exchanges, matches them, and detects
        price discrepancies above the threshold.

        Args:
            min_spread: Minimum absolute spread to report.
            min_liquidity: Minimum dollar liquidity to consider.
            limit: Maximum markets to fetch per exchange.

        Returns:
            List of Divergence objects sorted by spread.
        """
        self._divergence_detector.min_spread = min_spread
        self._divergence_detector.min_liquidity = min_liquidity

        markets_by_exchange = await self.fetch_all_markets(limit=limit)
        pairs = self.match_markets(markets_by_exchange)
        divergences = self._divergence_detector.detect(pairs)

        # Persist divergences to storage
        if divergences:
            try:
                self._storage.insert_divergences(divergences)
            except Exception:
                logger.debug("Failed to persist divergences to storage", exc_info=True)

        return divergences

    async def violations(
        self,
    ) -> tuple[list[ProbabilityViolation], list[MultiOutcomeViolation]]:
        """Find probability violations across all exchanges.

        Returns:
            Tuple of (binary_violations, multi_outcome_violations).
        """
        if not self._market_cache:
            await self.fetch_all_markets()

        all_markets: list[Market] = []
        for markets in self._market_cache.values():
            all_markets.extend(markets)

        result = self._violation_detector.detect_all(all_markets)

        # Persist binary violations to storage
        binary_violations = result[0]
        if binary_violations:
            try:
                self._storage.insert_violations(binary_violations)
            except Exception:
                logger.debug("Failed to persist violations to storage", exc_info=True)

        return result

    async def liquidity(self, exchange: str, market_id: str) -> LiquidityProfile:
        """Analyze liquidity for a specific market.

        Args:
            exchange: Exchange name (e.g. 'polymarket').
            market_id: Exchange-native market identifier.

        Returns:
            LiquidityProfile with depth and impact analysis.
        """
        connector = self._get_exchange(exchange)
        market = await connector.fetch_market(market_id)
        liquidity_id = market_id
        if connector.name == ExchangeName.POLYMARKET:
            token_ids = market.metadata.get("clobTokenIds") or market.metadata.get(
                "tokens"
            )
            if isinstance(token_ids, list) and token_ids:
                liquidity_id = str(token_ids[0])
            elif isinstance(token_ids, str) and token_ids:
                liquidity_id = token_ids

        try:
            order_book = await connector.fetch_order_book(liquidity_id)
            from arbiter.models import Outcome

            updated_outcomes = []
            for i, outcome in enumerate(market.outcomes):
                if i == 0:
                    updated_outcomes.append(
                        Outcome(
                            name=outcome.name,
                            price=outcome.price,
                            order_book=order_book,
                            volume=outcome.volume,
                        )
                    )
                else:
                    updated_outcomes.append(outcome)

            market = market.model_copy(update={"outcomes": updated_outcomes})
        except Exception:
            logger.warning("Could not fetch order book for %s", market_id)

        return self._liquidity_analyzer.analyze(market)

    async def quality(
        self,
        exchange: str | None = None,
        category: str = "all",
    ) -> MarketQuality:
        """Compute market quality scores.

        Args:
            exchange: Specific exchange, or None for all.
            category: Market category filter.

        Returns:
            MarketQuality metrics.
        """
        exchange_enum = None
        if exchange:
            exchange_enum = self._get_exchange(exchange).name

        if not self._market_cache:
            await self.fetch_all_markets(active_only=False)

        all_markets: list[Market] = []
        for markets in self._market_cache.values():
            all_markets.extend(markets)

        return self._quality_scorer.score(all_markets, exchange_enum, category)

    async def monitor(
        self,
        divergence_threshold: float = 0.03,
        interval_seconds: float = 30.0,
    ) -> AsyncIterator[Alert]:
        """Monitor exchanges for divergences and yield alerts.

        Continuously polls exchanges, detects divergences, and yields
        Alert objects when thresholds are exceeded.

        Args:
            divergence_threshold: Minimum spread to alert on.
            interval_seconds: Polling interval.

        Yields:
            Alert objects for significant market events.
        """
        while True:
            try:
                divs = await self.divergences(min_spread=divergence_threshold)
                for div in divs:
                    alert = self._alert_manager.alert_from_divergence(
                        div, divergence_threshold
                    )
                    if alert:
                        await self._alert_manager.emit(alert)
                        yield alert

                binary_v, _ = await self.violations()
                for v in binary_v:
                    alert = self._alert_manager.alert_from_violation(v)
                    await self._alert_manager.emit(alert)
                    yield alert

            except ArbiterError as exc:
                logger.error("Monitor cycle error: %s", exc)

            await asyncio.sleep(interval_seconds)

    async def export_dataset(
        self,
        path: str,
        exchanges: list[str] | None = None,
        include_orderbook_snapshots: bool = False,
    ) -> str:
        """Export a research-ready dataset to Parquet.

        Args:
            path: Output file path.
            exchanges: Filter to specific exchanges.
            include_orderbook_snapshots: Whether to include order book data.

        Returns:
            Path to the exported file.
        """
        if not self._market_cache:
            await self.fetch_all_markets(active_only=False)

        all_markets: list[Market] = []
        for ex_name, markets in self._market_cache.items():
            if exchanges and ex_name not in exchanges:
                continue
            all_markets.extend(markets)

        result_path = self._exporter.export_markets_to_parquet(all_markets, path)
        return str(result_path)

    def _get_exchange(self, name: str) -> BaseExchange:
        """Look up an exchange connector by name.

        Args:
            name: Exchange name string.

        Returns:
            The matching exchange connector.

        Raises:
            ConfigError: If the exchange is not configured.
        """
        for ex in self._exchanges:
            if ex.name.value == name.lower():
                return ex
        raise ConfigError(
            "exchange",
            f"Exchange '{name}' not configured. "
            f"Available: {[e.name.value for e in self._exchanges]}",
        )

    async def close(self) -> None:
        """Close all exchange connections and storage."""
        for ex in self._exchanges:
            await ex.close()
        self._storage.close()

    async def __aenter__(self) -> Arbiter:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()
