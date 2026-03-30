"""Tests for arbiter.exceptions."""

from __future__ import annotations

from arbiter.exceptions import (
    ArbiterError,
    ConfigError,
    ExchangeAuthError,
    ExchangeConnectionError,
    ExchangeError,
    ExchangeRateLimitError,
    InsufficientDataError,
    MarketNotFoundError,
    MatchingError,
    StorageError,
)


class TestExceptionHierarchy:
    """All exceptions should inherit from ArbiterError."""

    def test_base_exception(self) -> None:
        err = ArbiterError("test")
        assert isinstance(err, Exception)

    def test_exchange_error(self) -> None:
        err = ExchangeError("polymarket", "something broke")
        assert isinstance(err, ArbiterError)
        assert err.exchange == "polymarket"
        assert "[polymarket]" in str(err)

    def test_auth_error(self) -> None:
        err = ExchangeAuthError("kalshi")
        assert isinstance(err, ExchangeError)
        assert "Authentication failed" in str(err)

    def test_rate_limit_error(self) -> None:
        err = ExchangeRateLimitError("polymarket", retry_after=5.0)
        assert isinstance(err, ExchangeError)
        assert err.retry_after == 5.0
        assert "5.0s" in str(err)

    def test_rate_limit_error_no_retry(self) -> None:
        err = ExchangeRateLimitError("kalshi")
        assert err.retry_after is None

    def test_connection_error(self) -> None:
        err = ExchangeConnectionError("manifold", "DNS failure")
        assert isinstance(err, ExchangeError)
        assert "DNS failure" in str(err)

    def test_market_not_found(self) -> None:
        err = MarketNotFoundError("kalshi", "KXFOO")
        assert isinstance(err, ExchangeError)
        assert err.market_id == "KXFOO"
        assert "KXFOO" in str(err)

    def test_matching_error(self) -> None:
        err = MatchingError("failed to match")
        assert isinstance(err, ArbiterError)

    def test_storage_error(self) -> None:
        err = StorageError("db locked")
        assert isinstance(err, ArbiterError)

    def test_config_error(self) -> None:
        err = ConfigError("api_key", "must not be empty")
        assert isinstance(err, ArbiterError)
        assert err.field == "api_key"

    def test_insufficient_data_error(self) -> None:
        err = InsufficientDataError("quality", required=10, available=3)
        assert isinstance(err, ArbiterError)
        assert err.required == 10
        assert err.available == 3
        assert "10" in str(err) and "3" in str(err)
