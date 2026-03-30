"""Core data models for the arbiter prediction market analytics platform.

All domain objects are defined as Pydantic models for validation, serialization,
and schema generation. These models represent the normalized view of prediction
market data across multiple exchanges.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExchangeName(enum.StrEnum):
    """Supported prediction market exchanges."""

    POLYMARKET = "polymarket"
    KALSHI = "kalshi"
    METACULUS = "metaculus"
    MANIFOLD = "manifold"


class MarketStatus(enum.StrEnum):
    """Lifecycle status of a prediction market."""

    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class ContractType(enum.StrEnum):
    """Type of prediction market contract."""

    BINARY = "binary"
    MULTI_OUTCOME = "multi_outcome"
    RANGED = "ranged"


class Side(enum.StrEnum):
    """Order side in an order book."""

    BID = "bid"
    ASK = "ask"


class OrderBookLevel(BaseModel):
    """Single price level in an order book."""

    price: float = Field(ge=0.0, le=1.0, description="Price in [0, 1]")
    quantity: float = Field(gt=0.0, description="Available quantity in dollars")

    model_config = {"frozen": True}


class OrderBook(BaseModel):
    """Full order book snapshot for a single outcome."""

    bids: list[OrderBookLevel] = Field(
        default_factory=list, description="Bid levels, best first"
    )
    asks: list[OrderBookLevel] = Field(
        default_factory=list, description="Ask levels, best first"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def best_bid(self) -> float | None:
        """Highest bid price, or None if no bids."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Lowest ask price, or None if no asks."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        """Bid-ask spread, or None if either side is empty."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def midpoint(self) -> float | None:
        """Midpoint price between best bid and ask."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None


class Outcome(BaseModel):
    """A single outcome within a prediction market."""

    name: str = Field(description="Outcome label, e.g. 'Yes', 'DeSantis'")
    price: float = Field(ge=0.0, le=1.0, description="Last traded price in [0, 1]")
    order_book: OrderBook | None = Field(
        default=None, description="Order book if available"
    )
    volume: float = Field(default=0.0, ge=0.0, description="Total volume in dollars")

    model_config = {"frozen": True}


class Market(BaseModel):
    """Normalized prediction market from any exchange.

    This is the canonical internal representation. Exchange connectors
    convert their native formats into this schema.
    """

    id: str = Field(description="Exchange-native market identifier")
    exchange: ExchangeName
    title: str = Field(description="Human-readable market question")
    description: str = Field(default="", description="Longer description or rules")
    category: str = Field(default="", description="Market category, e.g. 'Politics'")
    contract_type: ContractType = Field(default=ContractType.BINARY)
    status: MarketStatus = Field(default=MarketStatus.ACTIVE)
    outcomes: list[Outcome] = Field(
        min_length=2, description="At least two outcomes (Yes/No for binary)"
    )
    url: str = Field(default="", description="Link to the market on the exchange")
    volume_total: float = Field(default=0.0, ge=0.0)
    created_at: datetime | None = None
    closes_at: datetime | None = None
    resolved_at: datetime | None = None
    resolution: str | None = Field(
        default=None, description="Winning outcome name if resolved"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def yes_price(self) -> float | None:
        """Price of the YES outcome for binary markets."""
        if self.contract_type != ContractType.BINARY:
            return None
        for o in self.outcomes:
            if o.name.lower() in ("yes", "y"):
                return o.price
        return self.outcomes[0].price if self.outcomes else None

    @property
    def no_price(self) -> float | None:
        """Price of the NO outcome for binary markets."""
        if self.contract_type != ContractType.BINARY:
            return None
        for o in self.outcomes:
            if o.name.lower() in ("no", "n"):
                return o.price
        return self.outcomes[1].price if len(self.outcomes) > 1 else None


class MarketPair(BaseModel):
    """A matched pair of markets across two exchanges representing the same event."""

    market_a: Market
    market_b: Market
    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Semantic similarity between the two markets"
    )
    matched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Divergence(BaseModel):
    """Price divergence between two markets on different exchanges.

    Represents a potential arbitrage opportunity where the same event
    is priced differently across exchanges.
    """

    event: str = Field(description="Canonical event description")
    outcome: str = Field(description="The outcome being compared")
    exchange_a: ExchangeName
    exchange_b: ExchangeName
    price_a: float = Field(ge=0.0, le=1.0)
    price_b: float = Field(ge=0.0, le=1.0)
    spread: float = Field(description="Absolute price difference")
    spread_pct: float = Field(description="Spread relative to midpoint")
    liquidity_a: float = Field(default=0.0, ge=0.0)
    liquidity_b: float = Field(default=0.0, ge=0.0)
    net_arb_profit_estimate: float = Field(
        default=0.0, description="Estimated profit after fees and slippage"
    )
    window_opened: datetime = Field(default_factory=lambda: datetime.now(UTC))
    market_a_id: str = Field(default="")
    market_b_id: str = Field(default="")

    @field_validator("spread", mode="before")
    @classmethod
    def compute_spread(cls, v: float, info: Any) -> float:
        """Allow spread to be auto-computed from prices."""
        return v


class ProbabilityViolation(BaseModel):
    """A market where YES + NO prices violate probability axioms.

    For binary markets: YES + NO should equal ~1.0 (minus exchange fees).
    When the sum exceeds 1.0, a risk-free arbitrage exists by shorting both sides.
    When the sum is well below 1.0, a risk-free profit exists by buying both sides.
    """

    market: str = Field(description="Market title")
    market_id: str
    exchange: ExchangeName
    yes_price: float = Field(ge=0.0, le=1.0)
    no_price: float = Field(ge=0.0, le=1.0)
    price_sum: float = Field(description="YES + NO price sum")
    implied_arb: float = Field(description="Risk-free profit per contract pair")
    volume_available: float = Field(default=0.0, ge=0.0)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MultiOutcomeViolation(BaseModel):
    """A multi-outcome market where outcome probabilities don't sum to ~1.0."""

    market: str
    market_id: str
    exchange: ExchangeName
    outcomes: dict[str, float] = Field(description="Outcome name -> price mapping")
    price_sum: float
    deviation: float = Field(description="How far the sum deviates from 1.0")
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LiquidityProfile(BaseModel):
    """Liquidity analysis for a single market on a single exchange."""

    market: str
    market_id: str
    exchange: ExchangeName
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    depth_at_1pct: float = Field(
        default=0.0, description="Dollar depth within 1% of midpoint"
    )
    depth_at_5pct: float = Field(
        default=0.0, description="Dollar depth within 5% of midpoint"
    )
    estimated_impact: dict[int, float] = Field(
        default_factory=dict,
        description="Trade size ($) -> estimated price impact",
    )
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MarketQuality(BaseModel):
    """Quality and accuracy metrics for a set of markets."""

    exchange: ExchangeName
    category: str = Field(default="all")
    brier_score: float = Field(
        ge=0.0, le=2.0, description="Brier score (lower is better)"
    )
    calibration_error: float = Field(
        ge=0.0, le=1.0, description="Mean absolute calibration error"
    )
    avg_resolution_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Average hours between event and resolution",
    )
    manipulation_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated manipulation severity (0=none, 1=severe)",
    )
    volume_accuracy_correlation: float = Field(
        ge=-1.0,
        le=1.0,
        description="Correlation between volume and prediction accuracy",
    )
    sample_size: int = Field(
        default=0, ge=0, description="Number of resolved markets in sample"
    )
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EfficiencyMetrics(BaseModel):
    """Metrics measuring how efficiently a market incorporates information."""

    market: str
    market_id: str
    exchange: ExchangeName
    price_discovery_speed_minutes: float = Field(
        default=0.0,
        ge=0.0,
        description="Minutes for price to reach 90% of final value after news",
    )
    avg_arb_window_minutes: float = Field(
        default=0.0,
        ge=0.0,
        description="Average duration of arbitrage windows in minutes",
    )
    information_ratio: float = Field(
        default=0.0,
        ge=0.0,
        description="Rate at which new information is priced in",
    )
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Alert(BaseModel):
    """Real-time alert for significant market events."""

    alert_type: str = Field(description="e.g. 'divergence', 'violation'")
    severity: str = Field(description="'low', 'medium', 'high', 'critical'")
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExchangeConfig(BaseModel):
    """Configuration for connecting to an exchange."""

    name: ExchangeName
    api_key: str | None = None
    api_secret: str | None = None
    base_url: str = ""
    ws_url: str = ""
    rate_limit_per_second: float = Field(default=5.0, gt=0.0)
    enabled: bool = True
