"""Connection pooling for exchange HTTP clients.

Reuses httpx.AsyncClient instances across requests instead of creating
new ones per-request. Provides a context manager for clean lifecycle
management and supports configurable pool sizes per base URL.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType

import httpx

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Manages a pool of reusable httpx.AsyncClient instances.

    Each unique base URL gets its own client. Clients are created lazily
    on first request and reused for all subsequent requests to the same
    base URL. The pool handles graceful shutdown of all clients.

    Example::

        async with ConnectionPool() as pool:
            client = await pool.get_client(
                "https://api.polymarket.com",
                headers={"Accept": "application/json"},
            )
            response = await client.get("/markets")
    """

    def __init__(
        self,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        """Initialize the connection pool.

        Args:
            timeout: Default request timeout in seconds.
            connect_timeout: TCP connection timeout in seconds.
            max_connections: Maximum total connections across all hosts.
            max_keepalive_connections: Maximum idle keep-alive connections.
        """
        self._timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def get_client(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
    ) -> httpx.AsyncClient:
        """Get or create a client for the given base URL.

        If a client already exists for this base URL and is not closed,
        it is returned. Otherwise, a new client is created.

        Args:
            base_url: The API base URL to connect to.
            headers: Default headers for the client.

        Returns:
            A reusable httpx.AsyncClient instance.

        Raises:
            RuntimeError: If the pool has been closed.
        """
        if self._closed:
            raise RuntimeError("ConnectionPool is closed")

        async with self._lock:
            existing = self._clients.get(base_url)
            if existing is not None and not existing.is_closed:
                return existing

            client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers or {},
                timeout=self._timeout,
                limits=self._limits,
            )
            self._clients[base_url] = client
            logger.debug("Created pooled client for %s", base_url)
            return client

    async def close_client(self, base_url: str) -> None:
        """Close and remove the client for a specific base URL.

        Args:
            base_url: The base URL whose client should be closed.
        """
        async with self._lock:
            client = self._clients.pop(base_url, None)
            if client and not client.is_closed:
                await client.aclose()

    async def close(self) -> None:
        """Close all pooled clients and mark the pool as closed."""
        self._closed = True
        async with self._lock:
            for base_url, client in self._clients.items():
                if not client.is_closed:
                    try:
                        await client.aclose()
                    except Exception:
                        logger.debug(
                            "Error closing client for %s", base_url, exc_info=True
                        )
            self._clients.clear()

    @property
    def active_connections(self) -> int:
        """Number of active (non-closed) clients in the pool."""
        return sum(
            1 for c in self._clients.values() if not c.is_closed
        )

    @property
    def is_closed(self) -> bool:
        """Whether the pool has been closed."""
        return self._closed

    async def __aenter__(self) -> ConnectionPool:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit -- closes all connections."""
        await self.close()
