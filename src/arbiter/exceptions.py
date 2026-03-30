"""Custom exception hierarchy for arbiter.

All arbiter-specific exceptions inherit from ArbiterError, making it
easy to catch any library error with a single except clause.
"""

from __future__ import annotations


class ArbiterError(Exception):
    """Base exception for all arbiter errors."""


class ExchangeError(ArbiterError):
    """Error communicating with a prediction market exchange."""

    def __init__(self, exchange: str, message: str) -> None:
        self.exchange = exchange
        super().__init__(f"[{exchange}] {message}")


class ExchangeAuthError(ExchangeError):
    """Authentication or authorization failure with an exchange."""

    def __init__(self, exchange: str) -> None:
        super().__init__(exchange, "Authentication failed. Check API credentials.")


class ExchangeRateLimitError(ExchangeError):
    """Rate limit exceeded on an exchange API."""

    def __init__(self, exchange: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        msg = "Rate limit exceeded"
        if retry_after is not None:
            msg += f" (retry after {retry_after:.1f}s)"
        super().__init__(exchange, msg)


class ExchangeConnectionError(ExchangeError):
    """Network-level connection failure to an exchange."""

    def __init__(self, exchange: str, reason: str = "Connection failed") -> None:
        super().__init__(exchange, reason)


class MarketNotFoundError(ExchangeError):
    """Requested market does not exist on the exchange."""

    def __init__(self, exchange: str, market_id: str) -> None:
        self.market_id = market_id
        super().__init__(exchange, f"Market '{market_id}' not found")


class MatchingError(ArbiterError):
    """Error during cross-exchange market matching."""


class StorageError(ArbiterError):
    """Error in the storage layer (DuckDB / Parquet)."""


class ConfigError(ArbiterError):
    """Invalid or missing configuration."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"Configuration error for '{field}': {message}")


class InsufficientDataError(ArbiterError):
    """Not enough data to perform the requested analysis."""

    def __init__(self, analysis: str, required: int, available: int) -> None:
        self.analysis = analysis
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient data for {analysis}: "
            f"need {required} data points, have {available}"
        )
