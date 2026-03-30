"""Extended tests for arbiter.output.alerts -- webhook delivery, retry, async callbacks."""

from __future__ import annotations

import httpx
import pytest
import respx

from arbiter.models import (
    Alert,
    Divergence,
    ExchangeName,
    ProbabilityViolation,
)
from arbiter.output.alerts import AlertManager


class TestAlertManagerWebhookDelivery:
    @respx.mock
    @pytest.mark.asyncio
    async def test_webhook_delivery_success(self) -> None:
        route = respx.post("https://example.com/hook").mock(
            return_value=httpx.Response(200)
        )

        manager = AlertManager()
        manager.add_webhook("https://example.com/hook")

        alert = Alert(
            alert_type="divergence",
            severity="high",
            message="Test alert",
        )
        await manager.emit(alert)
        assert route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_webhook_delivery_failure(self) -> None:
        respx.post("https://example.com/fail").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        manager = AlertManager()
        manager.add_webhook("https://example.com/fail")

        alert = Alert(
            alert_type="divergence",
            severity="high",
            message="Test alert",
        )
        # Should not raise even if webhook fails
        await manager.emit(alert)
        assert len(manager.history) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_multiple_webhooks(self) -> None:
        route1 = respx.post("https://example.com/hook1").mock(
            return_value=httpx.Response(200)
        )
        route2 = respx.post("https://example.com/hook2").mock(
            return_value=httpx.Response(200)
        )

        manager = AlertManager()
        manager.add_webhook("https://example.com/hook1")
        manager.add_webhook("https://example.com/hook2")

        alert = Alert(
            alert_type="violation",
            severity="critical",
            message="Both hooks",
        )
        await manager.emit(alert)
        assert route1.called
        assert route2.called


class TestAlertManagerHistory:
    @pytest.mark.asyncio
    async def test_history_capped_at_max(self) -> None:
        manager = AlertManager()
        manager._max_history = 5

        for i in range(10):
            alert = Alert(
                alert_type="test",
                severity="low",
                message=f"Alert {i}",
            )
            await manager.emit(alert)

        assert len(manager.history) == 5
        # Should contain the most recent alerts
        assert "Alert 9" in manager.history[-1].message

    @pytest.mark.asyncio
    async def test_history_preserves_order(self) -> None:
        manager = AlertManager()
        for i in range(3):
            alert = Alert(
                alert_type="test",
                severity="low",
                message=f"Alert {i}",
            )
            await manager.emit(alert)

        messages = [a.message for a in manager.history]
        assert messages == ["Alert 0", "Alert 1", "Alert 2"]


class TestAlertManagerCallbacks:
    @pytest.mark.asyncio
    async def test_async_callback(self) -> None:
        received = []

        async def async_handler(alert: Alert) -> None:
            received.append(alert)

        manager = AlertManager()
        manager.add_callback(async_handler)

        alert = Alert(
            alert_type="test",
            severity="low",
            message="Async callback test",
        )
        await manager.emit(alert)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_callback_exception_handled(self) -> None:
        def bad_callback(alert: Alert) -> None:
            raise ValueError("callback error")

        manager = AlertManager()
        manager.add_callback(bad_callback)

        alert = Alert(
            alert_type="test",
            severity="low",
            message="Bad callback test",
        )
        # Should not raise
        await manager.emit(alert)
        assert len(manager.history) == 1

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self) -> None:
        results_a: list[Alert] = []
        results_b: list[Alert] = []

        manager = AlertManager()
        manager.add_callback(lambda a: results_a.append(a))
        manager.add_callback(lambda a: results_b.append(a))

        alert = Alert(
            alert_type="test",
            severity="medium",
            message="Multi callback",
        )
        await manager.emit(alert)
        assert len(results_a) == 1
        assert len(results_b) == 1


class TestAlertFromDivergenceEdgeCases:
    def test_severity_boundary_at_004(self) -> None:
        """Spread of exactly 0.04 should be 'medium'."""
        manager = AlertManager()
        div = Divergence(
            event="Test",
            outcome="Yes",
            exchange_a=ExchangeName.POLYMARKET,
            exchange_b=ExchangeName.KALSHI,
            price_a=0.52,
            price_b=0.48,
            spread=0.04,
            spread_pct=0.08,
        )
        alert = manager.alert_from_divergence(div, threshold=0.01)
        assert alert is not None
        assert alert.severity == "medium"

    def test_severity_boundary_at_007(self) -> None:
        """Spread of exactly 0.07 should be 'high'."""
        manager = AlertManager()
        div = Divergence(
            event="Test",
            outcome="Yes",
            exchange_a=ExchangeName.POLYMARKET,
            exchange_b=ExchangeName.KALSHI,
            price_a=0.535,
            price_b=0.465,
            spread=0.07,
            spread_pct=0.14,
        )
        alert = manager.alert_from_divergence(div, threshold=0.01)
        assert alert is not None
        assert alert.severity == "high"


class TestAlertFromViolationEdgeCases:
    def test_violation_severity_critical(self) -> None:
        manager = AlertManager()
        v = ProbabilityViolation(
            market="Critical",
            market_id="crit-1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.60,
            no_price=0.55,
            price_sum=1.15,
            implied_arb=0.15,
        )
        alert = manager.alert_from_violation(v)
        assert alert.severity == "critical"

    def test_violation_severity_high(self) -> None:
        manager = AlertManager()
        v = ProbabilityViolation(
            market="High",
            market_id="high-1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.55,
            no_price=0.52,
            price_sum=1.07,
            implied_arb=0.07,
        )
        alert = manager.alert_from_violation(v)
        assert alert.severity == "high"

    def test_violation_severity_medium(self) -> None:
        manager = AlertManager()
        v = ProbabilityViolation(
            market="Medium",
            market_id="med-1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.52,
            no_price=0.51,
            price_sum=1.03,
            implied_arb=0.03,
        )
        alert = manager.alert_from_violation(v)
        assert alert.severity == "medium"


class TestAlertManagerMonitorDivergences:
    @pytest.mark.asyncio
    async def test_monitor_yields_alerts(self) -> None:
        manager = AlertManager()

        async def mock_stream():
            yield Divergence(
                event="Monitored",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.60,
                price_b=0.50,
                spread=0.10,
                spread_pct=0.20,
            )

        alerts = []
        async for alert in manager.monitor_divergences(mock_stream(), threshold=0.05):
            alerts.append(alert)

        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_monitor_skips_below_threshold(self) -> None:
        manager = AlertManager()

        async def mock_stream():
            yield Divergence(
                event="Small",
                outcome="Yes",
                exchange_a=ExchangeName.POLYMARKET,
                exchange_b=ExchangeName.KALSHI,
                price_a=0.51,
                price_b=0.50,
                spread=0.01,
                spread_pct=0.02,
            )

        alerts = []
        async for alert in manager.monitor_divergences(mock_stream(), threshold=0.05):
            alerts.append(alert)

        assert len(alerts) == 0
