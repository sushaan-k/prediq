"""Tests for arbiter.exchanges.pool -- connection pooling."""

from __future__ import annotations

import pytest

from arbiter.exchanges.pool import ConnectionPool


@pytest.mark.asyncio
class TestConnectionPool:
    """Tests for the ConnectionPool class."""

    async def test_get_client_creates_client(self) -> None:
        async with ConnectionPool() as pool:
            client = await pool.get_client("https://example.com")
            assert client is not None
            assert not client.is_closed

    async def test_get_client_reuses_client(self) -> None:
        async with ConnectionPool() as pool:
            c1 = await pool.get_client("https://example.com")
            c2 = await pool.get_client("https://example.com")
            assert c1 is c2

    async def test_different_urls_get_different_clients(self) -> None:
        async with ConnectionPool() as pool:
            c1 = await pool.get_client("https://a.example.com")
            c2 = await pool.get_client("https://b.example.com")
            assert c1 is not c2

    async def test_active_connections_count(self) -> None:
        async with ConnectionPool() as pool:
            assert pool.active_connections == 0
            await pool.get_client("https://a.example.com")
            assert pool.active_connections == 1
            await pool.get_client("https://b.example.com")
            assert pool.active_connections == 2
            # Same URL doesn't add a new connection
            await pool.get_client("https://a.example.com")
            assert pool.active_connections == 2

    async def test_close_client_removes_from_pool(self) -> None:
        async with ConnectionPool() as pool:
            await pool.get_client("https://example.com")
            assert pool.active_connections == 1
            await pool.close_client("https://example.com")
            assert pool.active_connections == 0

    async def test_close_marks_pool_as_closed(self) -> None:
        pool = ConnectionPool()
        assert not pool.is_closed
        await pool.close()
        assert pool.is_closed

    async def test_get_client_after_close_raises(self) -> None:
        pool = ConnectionPool()
        await pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            await pool.get_client("https://example.com")

    async def test_context_manager_closes_pool(self) -> None:
        pool = ConnectionPool()
        async with pool:
            await pool.get_client("https://example.com")
            assert pool.active_connections == 1
        assert pool.is_closed
        assert pool.active_connections == 0

    async def test_closed_client_gets_recreated(self) -> None:
        async with ConnectionPool() as pool:
            c1 = await pool.get_client("https://example.com")
            await c1.aclose()
            c2 = await pool.get_client("https://example.com")
            assert c2 is not c1
            assert not c2.is_closed

    async def test_custom_timeout_and_limits(self) -> None:
        async with ConnectionPool(
            timeout=15.0,
            connect_timeout=5.0,
            max_connections=50,
            max_keepalive_connections=10,
        ) as pool:
            client = await pool.get_client("https://example.com")
            assert client is not None

    async def test_close_client_nonexistent_url(self) -> None:
        """Closing a client for an unknown URL should not raise."""
        async with ConnectionPool() as pool:
            await pool.close_client("https://nonexistent.example.com")

    async def test_headers_passed_to_client(self) -> None:
        async with ConnectionPool() as pool:
            headers = {"Authorization": "Bearer test123"}
            client = await pool.get_client("https://example.com", headers=headers)
            assert client.headers.get("authorization") == "Bearer test123"
