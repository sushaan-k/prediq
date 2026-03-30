"""Exchange connectors for prediction market platforms.

Each connector normalizes exchange-specific data into the canonical
``Market`` model defined in ``arbiter.models``.
"""

from arbiter.exchanges.base import BaseExchange
from arbiter.exchanges.kalshi import KalshiExchange
from arbiter.exchanges.manifold import ManifoldExchange
from arbiter.exchanges.metaculus import MetaculusExchange
from arbiter.exchanges.polymarket import PolymarketExchange

__all__ = [
    "BaseExchange",
    "KalshiExchange",
    "ManifoldExchange",
    "MetaculusExchange",
    "PolymarketExchange",
]
