"""Abstract base class for prediction market exchange connectors.

Every exchange connector must implement this interface. The base class
provides common HTTP client management, rate limiting, and lifecycle hooks.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from arbiter.exceptions import ExchangeConnectionError, ExchangeRateLimitError
from arbiter.models import (
    ExchangeConfig,
    ExchangeName,
    Market,
    OrderBook,
)

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Async token-bucket rate limiter.

    Enforces a maximum rate of ``rate`` requests per second by tracking
    available tokens and refilling them over time.
    """

    def __init__(self, rate: float) -> None:
        """
        Args:
            rate: Maximum requests per second (must be > 0).
        """
        self._rate = max(rate, 0.1)
        self._max_tokens = max(1.0, rate)
        self._tokens = self._max_tokens
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._max_tokens, self._tokens + elapsed * self._rate
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Sleep until at least one token is available
                wait = (1.0 - self._tokens) / self._rate
                # Release the lock while sleeping so other callers can
                # check too -- but since we need to re-check, just sleep
                # a short interval.
                await asyncio.sleep(wait)


class BaseExchange(abc.ABC):
    """Abstract base for all exchange connectors.

    Manages an httpx async client with automatic rate limiting
    and provides the interface that analytics engines depend on.
    """

    name: ExchangeName

    def __init__(self, config: ExchangeConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = TokenBucketRateLimiter(config.rate_limit_per_second)
        self._closed = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the shared HTTP client, creating it if needed."""
        if self._client is None or self._client.is_closed:
            headers = self._build_headers()
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    def _build_headers(self) -> dict[str, str]:
        """Build default request headers. Override for auth."""
        return {
            "User-Agent": "arbiter/0.1.0",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a rate-limited HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: URL path relative to base_url.
            **kwargs: Forwarded to httpx.AsyncClient.request.

        Returns:
            The HTTP response.

        Raises:
            ExchangeRateLimitError: If the exchange returns 429.
            ExchangeConnectionError: On network failures.
        """
        await self._rate_limiter.acquire()

        client = await self._get_client()
        try:
            response = await client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ExchangeConnectionError(self.name.value, str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise ExchangeConnectionError(
                self.name.value, f"Request timed out: {exc}"
            ) from exc

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise ExchangeRateLimitError(
                self.name.value,
                float(retry_after) if retry_after else None,
            )

        response.raise_for_status()
        return response

    async def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for GET requests."""
        return await self._request("GET", path, **kwargs)

    @abc.abstractmethod
    async def fetch_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[Market]:
        """Fetch available markets from the exchange.

        Args:
            active_only: If True, only return markets currently open for trading.
            limit: Maximum number of markets to return.

        Returns:
            List of normalized Market objects.
        """

    @abc.abstractmethod
    async def fetch_market(self, market_id: str) -> Market:
        """Fetch a single market by its exchange-native ID.

        Args:
            market_id: The exchange-specific market identifier.

        Returns:
            A normalized Market object.

        Raises:
            MarketNotFoundError: If the market does not exist.
        """

    @abc.abstractmethod
    async def fetch_order_book(self, market_id: str) -> OrderBook:
        """Fetch the current order book for a market.

        Args:
            market_id: The exchange-specific market identifier.

        Returns:
            An OrderBook snapshot.
        """

    async def stream_prices(self, market_ids: list[str]) -> AsyncIterator[Market]:
        """Stream real-time price updates via WebSocket.

        Default implementation polls via REST. Exchange connectors
        with WebSocket support should override this.

        Args:
            market_ids: List of market IDs to subscribe to.

        Yields:
            Updated Market objects as prices change.
        """
        while not self._closed:
            for mid in market_ids:
                try:
                    market = await self.fetch_market(mid)
                    yield market
                except Exception:
                    logger.warning(
                        "Failed to fetch market %s from %s",
                        mid,
                        self.name.value,
                    )
            await asyncio.sleep(5.0)

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._closed = True
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> BaseExchange:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()
