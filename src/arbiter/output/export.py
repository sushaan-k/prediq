"""Data export for research and analysis.

Exports prediction market data to Parquet and CSV formats,
producing research-ready datasets for academic analysis.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any

from arbiter.models import Divergence, Market

logger = logging.getLogger(__name__)


class DataExporter:
    """Exports prediction market data to files.

    Supports Parquet (via pyarrow) and CSV formats. Parquet is preferred
    for large datasets due to columnar compression and fast analytical
    queries.
    """

    MARKET_EXPORT_FIELDS = [
        "id",
        "exchange",
        "title",
        "category",
        "contract_type",
        "status",
        "yes_price",
        "no_price",
        "volume_total",
        "url",
        "created_at",
        "closes_at",
        "resolved_at",
        "resolution",
        "fetched_at",
    ]

    DIVERGENCE_EXPORT_FIELDS = [
        "event",
        "outcome",
        "exchange_a",
        "exchange_b",
        "price_a",
        "price_b",
        "spread",
        "spread_pct",
        "liquidity_a",
        "liquidity_b",
        "net_arb_profit_estimate",
        "window_opened",
        "market_a_id",
        "market_b_id",
    ]

    def export_markets_to_parquet(
        self,
        markets: list[Market],
        path: str | Path,
    ) -> Path:
        """Export market data to a Parquet file.

        Args:
            markets: List of markets to export.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        records = [self._market_to_record(m) for m in markets]

        if not records:
            logger.warning("No markets to export")
            table = pa.Table.from_pylist([], schema=self._market_schema())
        else:
            table = pa.Table.from_pylist(records)

        pq.write_table(table, str(path), compression="snappy")
        logger.info("Exported %d markets to %s", len(records), path)
        return path

    def export_markets_to_csv(
        self,
        markets: list[Market],
        path: str | Path,
    ) -> Path:
        """Export market data to a CSV file.

        Args:
            markets: List of markets to export.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        path = Path(path)
        records = [self._market_to_record(m) for m in markets]

        if not records:
            logger.warning("No markets to export")
            path.write_text(self._csv_header(self.MARKET_EXPORT_FIELDS) + "\n")
            return path

        fieldnames = self.MARKET_EXPORT_FIELDS
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info("Exported %d markets to %s", len(records), path)
        return path

    def export_divergences_to_parquet(
        self,
        divergences: list[Divergence],
        path: str | Path,
    ) -> Path:
        """Export divergence data to a Parquet file.

        Args:
            divergences: List of divergences to export.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        records = [self._divergence_to_record(d) for d in divergences]

        if not records:
            table = pa.Table.from_pylist([], schema=self._divergence_schema())
        else:
            table = pa.Table.from_pylist(records)

        pq.write_table(table, str(path), compression="snappy")
        logger.info("Exported %d divergences to %s", len(records), path)
        return path

    def markets_to_csv_string(self, markets: list[Market]) -> str:
        """Export markets to a CSV string (for API responses).

        Args:
            markets: List of markets.

        Returns:
            CSV-formatted string.
        """
        records = [self._market_to_record(m) for m in markets]
        if not records:
            return self._csv_header(self.MARKET_EXPORT_FIELDS) + "\n"

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.MARKET_EXPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()

    @staticmethod
    def _market_to_record(market: Market) -> dict[str, Any]:
        """Convert a Market to a flat dictionary for export.

        Args:
            market: The market to flatten.

        Returns:
            Flat dictionary suitable for tabular export.
        """
        return {
            "id": market.id,
            "exchange": market.exchange.value,
            "title": market.title,
            "category": market.category,
            "contract_type": market.contract_type.value,
            "status": market.status.value,
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "volume_total": market.volume_total,
            "url": market.url,
            "created_at": market.created_at.isoformat() if market.created_at else None,
            "closes_at": market.closes_at.isoformat() if market.closes_at else None,
            "resolved_at": market.resolved_at.isoformat()
            if market.resolved_at
            else None,
            "resolution": market.resolution,
            "fetched_at": market.fetched_at.isoformat(),
        }

    @staticmethod
    def _divergence_to_record(div: Divergence) -> dict[str, Any]:
        """Convert a Divergence to a flat dictionary for export.

        Args:
            div: The divergence to flatten.

        Returns:
            Flat dictionary suitable for tabular export.
        """
        return {
            "event": div.event,
            "outcome": div.outcome,
            "exchange_a": div.exchange_a.value,
            "exchange_b": div.exchange_b.value,
            "price_a": div.price_a,
            "price_b": div.price_b,
            "spread": div.spread,
            "spread_pct": div.spread_pct,
            "liquidity_a": div.liquidity_a,
            "liquidity_b": div.liquidity_b,
            "net_arb_profit_estimate": div.net_arb_profit_estimate,
            "window_opened": div.window_opened.isoformat(),
            "market_a_id": div.market_a_id,
            "market_b_id": div.market_b_id,
        }

    @classmethod
    def _market_schema(cls) -> Any:
        """Build the Parquet schema for market exports."""
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("exchange", pa.string()),
                pa.field("title", pa.string()),
                pa.field("category", pa.string()),
                pa.field("contract_type", pa.string()),
                pa.field("status", pa.string()),
                pa.field("yes_price", pa.float64()),
                pa.field("no_price", pa.float64()),
                pa.field("volume_total", pa.float64()),
                pa.field("url", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("closes_at", pa.string()),
                pa.field("resolved_at", pa.string()),
                pa.field("resolution", pa.string()),
                pa.field("fetched_at", pa.string()),
            ]
        )

    @classmethod
    def _divergence_schema(cls) -> Any:
        """Build the Parquet schema for divergence exports."""
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("event", pa.string()),
                pa.field("outcome", pa.string()),
                pa.field("exchange_a", pa.string()),
                pa.field("exchange_b", pa.string()),
                pa.field("price_a", pa.float64()),
                pa.field("price_b", pa.float64()),
                pa.field("spread", pa.float64()),
                pa.field("spread_pct", pa.float64()),
                pa.field("liquidity_a", pa.float64()),
                pa.field("liquidity_b", pa.float64()),
                pa.field("net_arb_profit_estimate", pa.float64()),
                pa.field("window_opened", pa.string()),
                pa.field("market_a_id", pa.string()),
                pa.field("market_b_id", pa.string()),
            ]
        )

    @staticmethod
    def _csv_header(fieldnames: list[str]) -> str:
        """Serialize just the CSV header row for empty exports."""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        return output.getvalue().strip("\r\n")
