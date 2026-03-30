"""DuckDB storage layer for persistent analytics data.

Provides schema management, insertion, and analytical queries for
markets, divergences, violations, and liquidity profiles. Uses DuckDB
for fast OLAP queries and supports Parquet export for research.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb

from arbiter.exceptions import StorageError
from arbiter.models import Divergence, Market, ProbabilityViolation

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS markets (
    id VARCHAR,
    exchange VARCHAR,
    title VARCHAR,
    category VARCHAR,
    contract_type VARCHAR,
    status VARCHAR,
    yes_price DOUBLE,
    no_price DOUBLE,
    volume_total DOUBLE,
    url VARCHAR,
    created_at TIMESTAMP,
    closes_at TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution VARCHAR,
    fetched_at TIMESTAMP,
    PRIMARY KEY (id, exchange, fetched_at)
);

CREATE TABLE IF NOT EXISTS divergences (
    event VARCHAR,
    outcome VARCHAR,
    exchange_a VARCHAR,
    exchange_b VARCHAR,
    price_a DOUBLE,
    price_b DOUBLE,
    spread DOUBLE,
    spread_pct DOUBLE,
    liquidity_a DOUBLE,
    liquidity_b DOUBLE,
    net_arb_profit_estimate DOUBLE,
    window_opened TIMESTAMP,
    market_a_id VARCHAR,
    market_b_id VARCHAR
);

CREATE TABLE IF NOT EXISTS violations (
    market VARCHAR,
    market_id VARCHAR,
    exchange VARCHAR,
    yes_price DOUBLE,
    no_price DOUBLE,
    price_sum DOUBLE,
    implied_arb DOUBLE,
    volume_available DOUBLE,
    detected_at TIMESTAMP
);
"""


class Storage:
    """DuckDB-backed storage for arbiter analytics data.

    Manages schema creation, data insertion, and analytical queries.
    Supports both in-memory (for testing) and file-backed databases.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the storage layer.

        Args:
            db_path: Path to DuckDB file. None for in-memory database.
        """
        self._db_path = str(db_path) if db_path else ":memory:"
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> None:
        """Open the database connection and create schema.

        Raises:
            StorageError: If the connection fails.
        """
        try:
            self._conn = duckdb.connect(self._db_path)
            self._conn.execute(_SCHEMA_SQL)
            logger.info("Connected to storage: %s", self._db_path)
        except Exception as exc:
            raise StorageError(f"Failed to connect to {self._db_path}: {exc}") from exc

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get the active connection, creating if needed.

        Returns:
            Active DuckDB connection.

        Raises:
            StorageError: If no connection is available.
        """
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def insert_markets(self, markets: list[Market]) -> int:
        """Insert market snapshots into the database.

        Args:
            markets: List of markets to store.

        Returns:
            Number of rows inserted.
        """
        if not markets:
            return 0

        rows = []
        for m in markets:
            rows.append(
                (
                    m.id,
                    m.exchange.value,
                    m.title,
                    m.category,
                    m.contract_type.value,
                    m.status.value,
                    m.yes_price,
                    m.no_price,
                    m.volume_total,
                    m.url,
                    m.created_at,
                    m.closes_at,
                    m.resolved_at,
                    m.resolution,
                    m.fetched_at,
                )
            )

        self.conn.executemany(
            "INSERT INTO markets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        logger.debug("Inserted %d market snapshots", len(rows))
        return len(rows)

    def insert_divergences(self, divergences: list[Divergence]) -> int:
        """Insert divergence records into the database.

        Args:
            divergences: List of divergences to store.

        Returns:
            Number of rows inserted.
        """
        if not divergences:
            return 0

        rows = []
        for d in divergences:
            rows.append(
                (
                    d.event,
                    d.outcome,
                    d.exchange_a.value,
                    d.exchange_b.value,
                    d.price_a,
                    d.price_b,
                    d.spread,
                    d.spread_pct,
                    d.liquidity_a,
                    d.liquidity_b,
                    d.net_arb_profit_estimate,
                    d.window_opened,
                    d.market_a_id,
                    d.market_b_id,
                )
            )

        self.conn.executemany(
            "INSERT INTO divergences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        logger.debug("Inserted %d divergences", len(rows))
        return len(rows)

    def insert_violations(self, violations: list[ProbabilityViolation]) -> int:
        """Insert binary probability violation records into the database.

        Args:
            violations: List of violations to store.

        Returns:
            Number of rows inserted.
        """
        if not violations:
            return 0

        rows = []
        for v in violations:
            rows.append(
                (
                    v.market,
                    v.market_id,
                    v.exchange.value,
                    v.yes_price,
                    v.no_price,
                    v.price_sum,
                    v.implied_arb,
                    v.volume_available,
                    v.detected_at,
                )
            )

        self.conn.executemany(
            "INSERT INTO violations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        logger.debug("Inserted %d violations", len(rows))
        return len(rows)

    def query_markets(
        self,
        exchange: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query stored market snapshots.

        Args:
            exchange: Filter by exchange name.
            status: Filter by market status.
            limit: Maximum results.

        Returns:
            List of market records as dictionaries.
        """
        query = "SELECT * FROM markets WHERE 1=1"
        params: list[Any] = []

        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY fetched_at DESC LIMIT ?"
        params.append(limit)

        result = self.conn.execute(query, params)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row, strict=False)) for row in result.fetchall()]

    def query_divergences(
        self,
        min_spread: float = 0.0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query stored divergence records.

        Args:
            min_spread: Minimum spread filter.
            limit: Maximum results.

        Returns:
            List of divergence records as dictionaries.
        """
        result = self.conn.execute(
            "SELECT * FROM divergences WHERE spread >= ? "
            "ORDER BY window_opened DESC LIMIT ?",
            [min_spread, limit],
        )
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row, strict=False)) for row in result.fetchall()]

    _VALID_TABLES: frozenset[str] = frozenset({"markets", "divergences", "violations"})

    def export_to_parquet(self, table: str, path: str | Path) -> Path:
        """Export a table to Parquet format.

        Args:
            table: Table name to export.  Must be one of
                ``"markets"``, ``"divergences"``, or ``"violations"``.
            path: Output file path.

        Returns:
            Path to the written Parquet file.

        Raises:
            StorageError: If the table name is invalid or the export fails.
        """
        if table not in self._VALID_TABLES:
            raise StorageError(
                f"Invalid table name '{table}'. "
                f"Allowed: {', '.join(sorted(self._VALID_TABLES))}"
            )
        path = Path(path)
        try:
            self.conn.execute(
                f"COPY {table} TO '{path}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
            )
            logger.info("Exported table '%s' to %s", table, path)
            return path
        except Exception as exc:
            raise StorageError(f"Failed to export {table}: {exc}") from exc

    def __enter__(self) -> Storage:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
