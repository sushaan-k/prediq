"""Webhook alert system for real-time notifications.

Sends alerts when significant market events occur (divergences,
violations, liquidity changes) via HTTP webhooks or callback functions.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from arbiter.models import Alert, Divergence, ProbabilityViolation

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alert rules and webhook delivery.

    Supports both webhook URLs (HTTP POST) and in-process callback
    functions for maximum flexibility.
    """

    def __init__(self) -> None:
        """Initialize the alert manager."""
        self._webhook_urls: list[str] = []
        self._callbacks: list[Callable[[Alert], Any]] = []
        self._alert_history: list[Alert] = []
        self._max_history = 1000

    def add_webhook(self, url: str) -> None:
        """Register a webhook URL for alert delivery.

        Args:
            url: HTTP(S) URL that will receive POST requests
                with alert JSON payloads.
        """
        self._webhook_urls.append(url)
        logger.info("Registered webhook: %s", url)

    def add_callback(self, callback: Callable[[Alert], Any]) -> None:
        """Register an in-process callback for alerts.

        Args:
            callback: Function that receives Alert objects.
        """
        self._callbacks.append(callback)

    def alert_from_divergence(
        self, divergence: Divergence, threshold: float = 0.05
    ) -> Alert | None:
        """Create an alert from a price divergence if significant.

        Args:
            divergence: The detected divergence.
            threshold: Minimum spread to trigger an alert.

        Returns:
            Alert if the divergence exceeds threshold, None otherwise.
        """
        if divergence.spread < threshold:
            return None

        severity = "low"
        if divergence.spread >= 0.10:
            severity = "critical"
        elif divergence.spread >= 0.07:
            severity = "high"
        elif divergence.spread >= 0.04:
            severity = "medium"

        return Alert(
            alert_type="divergence",
            severity=severity,
            message=(
                f"Price divergence on '{divergence.event}' "
                f"({divergence.outcome}): "
                f"{divergence.exchange_a.value} @ {divergence.price_a:.3f} vs "
                f"{divergence.exchange_b.value} @ {divergence.price_b:.3f} "
                f"(spread: {divergence.spread:.3f}, "
                f"{divergence.spread_pct:.1%})"
            ),
            data=divergence.model_dump(mode="json"),
            created_at=datetime.now(UTC),
        )

    def alert_from_violation(self, violation: ProbabilityViolation) -> Alert:
        """Create an alert from a probability violation.

        Args:
            violation: The detected violation.

        Returns:
            Alert describing the violation.
        """
        severity = "medium"
        if violation.implied_arb >= 0.10:
            severity = "critical"
        elif violation.implied_arb >= 0.05:
            severity = "high"

        return Alert(
            alert_type="violation",
            severity=severity,
            message=(
                f"Probability violation on '{violation.market}' "
                f"({violation.exchange.value}): "
                f"YES={violation.yes_price:.3f} + NO={violation.no_price:.3f} "
                f"= {violation.price_sum:.3f} "
                f"(arb: {violation.implied_arb:.3f})"
            ),
            data=violation.model_dump(mode="json"),
            created_at=datetime.now(UTC),
        )

    async def emit(self, alert: Alert) -> None:
        """Emit an alert to all registered webhooks and callbacks.

        Args:
            alert: The alert to send.
        """
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history :]

        logger.info(
            "Alert [%s] %s: %s", alert.severity, alert.alert_type, alert.message
        )

        for callback in self._callbacks:
            try:
                result = callback(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Alert callback failed")

        if self._webhook_urls:
            payload = alert.model_dump(mode="json")
            async with httpx.AsyncClient(timeout=10.0) as client:
                for url in self._webhook_urls:
                    try:
                        await client.post(url, json=payload)
                    except Exception:
                        logger.warning("Failed to send webhook to %s", url)

    async def monitor_divergences(
        self,
        divergence_stream: AsyncIterator[Divergence],
        threshold: float = 0.03,
    ) -> AsyncIterator[Alert]:
        """Monitor a divergence stream and yield alerts.

        Args:
            divergence_stream: Async iterator producing divergences.
            threshold: Minimum spread to alert on.

        Yields:
            Alert objects for significant divergences.
        """
        async for divergence in divergence_stream:
            alert = self.alert_from_divergence(divergence, threshold)
            if alert is not None:
                await self.emit(alert)
                yield alert

    @property
    def history(self) -> list[Alert]:
        """Return recent alert history."""
        return list(self._alert_history)
