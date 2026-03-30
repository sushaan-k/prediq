"""Tests for arbiter.cli -- Typer CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from arbiter.cli import app
from arbiter.models import (
    Divergence,
    ExchangeName,
    Market,
    Outcome,
    ProbabilityViolation,
)

runner = CliRunner()


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_divergence(spread: float = 0.05) -> Divergence:
    return Divergence(
        event="Will BTC hit 200K?",
        outcome="Yes",
        exchange_a=ExchangeName.POLYMARKET,
        exchange_b=ExchangeName.KALSHI,
        price_a=0.50 + spread / 2,
        price_b=0.50 - spread / 2,
        spread=spread,
        spread_pct=spread * 2,
    )


def _make_mock_arbiter(**overrides):
    """Create a mock Arbiter that works as an async context manager."""
    mock_arb = AsyncMock()
    mock_arb.__aenter__ = AsyncMock(return_value=mock_arb)
    mock_arb.__aexit__ = AsyncMock(return_value=False)
    for key, value in overrides.items():
        setattr(mock_arb, key, value)
    return mock_arb


# ── version command ──────────────────────────────────────────────────────


class TestVersionCommand:
    def test_version_output(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


# ── scan command ─────────────────────────────────────────────────────────


class TestScanCommand:
    @patch("arbiter.engine.Arbiter")
    def test_scan_with_divergences(self, mock_arbiter_cls: AsyncMock) -> None:
        mock_arb = _make_mock_arbiter(
            divergences=AsyncMock(
                return_value=[_make_divergence(0.05), _make_divergence(0.08)]
            )
        )
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["scan", "--min-spread", "0.01"])
        assert result.exit_code == 0
        assert "Price Divergences" in result.output

        mock_arb.divergences.assert_awaited_once_with(min_spread=0.01, limit=50)

    @patch("arbiter.engine.Arbiter")
    def test_scan_limit_controls_fetch_size(self, mock_arbiter_cls: AsyncMock) -> None:
        mock_arb = _make_mock_arbiter(
            divergences=AsyncMock(return_value=[_make_divergence(0.05)])
        )
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["scan", "--limit", "2"])
        assert result.exit_code == 0
        mock_arb.divergences.assert_awaited_once_with(min_spread=0.02, limit=2)

    @patch("arbiter.engine.Arbiter")
    def test_scan_no_divergences(self, mock_arbiter_cls: AsyncMock) -> None:
        mock_arb = _make_mock_arbiter(divergences=AsyncMock(return_value=[]))
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        assert "No divergences" in result.output

    @patch("arbiter.engine.Arbiter")
    def test_scan_json_output(self, mock_arbiter_cls: AsyncMock) -> None:
        mock_arb = _make_mock_arbiter(
            divergences=AsyncMock(return_value=[_make_divergence(0.06)])
        )
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["scan", "--json"])
        assert result.exit_code == 0
        assert "0.06" in result.output

    @patch("arbiter.engine.Arbiter")
    def test_scan_error_handling(self, mock_arbiter_cls: AsyncMock) -> None:
        mock_arb = _make_mock_arbiter(
            divergences=AsyncMock(side_effect=RuntimeError("API down"))
        )
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        assert "Error" in result.output


# ── violations command ───────────────────────────────────────────────────


class TestViolationsCommand:
    @patch("arbiter.engine.Arbiter")
    def test_violations_with_binary(self, mock_arbiter_cls: AsyncMock) -> None:
        binary_v = ProbabilityViolation(
            market="Test market",
            market_id="t-1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.55,
            no_price=0.52,
            price_sum=1.07,
            implied_arb=0.07,
        )
        mock_arb = _make_mock_arbiter(
            violations=AsyncMock(return_value=([binary_v], []))
        )
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["violations"])
        assert result.exit_code == 0
        assert "Binary Probability Violations" in result.output

    @patch("arbiter.engine.Arbiter")
    def test_violations_no_results(self, mock_arbiter_cls: AsyncMock) -> None:
        mock_arb = _make_mock_arbiter(violations=AsyncMock(return_value=([], [])))
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["violations"])
        assert result.exit_code == 0
        assert "No violations" in result.output

    @patch("arbiter.engine.Arbiter")
    def test_violations_json_output(self, mock_arbiter_cls: AsyncMock) -> None:
        binary_v = ProbabilityViolation(
            market="Test",
            market_id="t-1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.55,
            no_price=0.52,
            price_sum=1.07,
            implied_arb=0.07,
        )
        mock_arb = _make_mock_arbiter(
            violations=AsyncMock(return_value=([binary_v], []))
        )
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["violations", "--json"])
        assert result.exit_code == 0
        assert "binary" in result.output

    @patch("arbiter.engine.Arbiter")
    def test_violations_error_handling(self, mock_arbiter_cls: AsyncMock) -> None:
        mock_arb = _make_mock_arbiter(
            violations=AsyncMock(side_effect=RuntimeError("fail"))
        )
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["violations"])
        assert result.exit_code == 0
        assert "Error" in result.output

    @patch("arbiter.engine.Arbiter")
    def test_violations_with_multi_outcome(self, mock_arbiter_cls: AsyncMock) -> None:
        from arbiter.models import MultiOutcomeViolation

        multi_v = MultiOutcomeViolation(
            market="Election winner",
            market_id="elec-1",
            exchange=ExchangeName.KALSHI,
            outcomes={"A": 0.40, "B": 0.35, "C": 0.30},
            price_sum=1.05,
            deviation=0.05,
        )
        mock_arb = _make_mock_arbiter(
            violations=AsyncMock(return_value=([], [multi_v]))
        )
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["violations"])
        assert result.exit_code == 0
        assert "Multi-Outcome Violations" in result.output

    @patch("arbiter.engine.Arbiter")
    def test_violations_both_types(self, mock_arbiter_cls: AsyncMock) -> None:
        from arbiter.models import MultiOutcomeViolation

        binary_v = ProbabilityViolation(
            market="Binary test",
            market_id="b-1",
            exchange=ExchangeName.POLYMARKET,
            yes_price=0.55,
            no_price=0.52,
            price_sum=1.07,
            implied_arb=0.07,
        )
        multi_v = MultiOutcomeViolation(
            market="Multi test",
            market_id="m-1",
            exchange=ExchangeName.KALSHI,
            outcomes={"A": 0.40, "B": 0.65},
            price_sum=1.05,
            deviation=0.05,
        )
        mock_arb = _make_mock_arbiter(
            violations=AsyncMock(return_value=([binary_v], [multi_v]))
        )
        mock_arbiter_cls.return_value = mock_arb

        result = runner.invoke(app, ["violations"])
        assert result.exit_code == 0
        assert "Binary Probability Violations" in result.output
        assert "Multi-Outcome Violations" in result.output


# ── export command ───────────────────────────────────────────────────────


class TestExportCommand:
    @patch("arbiter.engine.Arbiter")
    def test_export_csv(
        self,
        mock_arbiter_cls: AsyncMock,
        tmp_path: Path,
    ) -> None:
        from datetime import UTC, datetime

        markets = [
            Market(
                id="m-1",
                exchange=ExchangeName.MANIFOLD,
                title="Test",
                outcomes=[
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ],
                fetched_at=datetime.now(UTC),
            )
        ]
        mock_exporter = AsyncMock()
        mock_exporter.export_markets_to_csv = lambda mlist, path: None
        mock_arb = _make_mock_arbiter(
            fetch_all_markets=AsyncMock(return_value=None),
            _market_cache={"manifold": markets},
            _exporter=mock_exporter,
        )
        mock_arbiter_cls.return_value = mock_arb

        out = str(tmp_path / "output.csv")
        result = runner.invoke(app, ["export", out, "--format", "csv"])
        assert result.exit_code == 0
        assert "Exported" in result.output

    @patch("arbiter.engine.Arbiter")
    def test_export_parquet(
        self,
        mock_arbiter_cls: AsyncMock,
        tmp_path: Path,
    ) -> None:
        from datetime import UTC, datetime

        markets = [
            Market(
                id="m-1",
                exchange=ExchangeName.MANIFOLD,
                title="Test",
                outcomes=[
                    Outcome(name="Yes", price=0.5, volume=0.0),
                    Outcome(name="No", price=0.5, volume=0.0),
                ],
                fetched_at=datetime.now(UTC),
            )
        ]
        mock_exporter = AsyncMock()
        mock_exporter.export_markets_to_parquet = lambda mlist, path: None
        mock_arb = _make_mock_arbiter(
            fetch_all_markets=AsyncMock(return_value=None),
            _market_cache={"manifold": markets},
            _exporter=mock_exporter,
        )
        mock_arbiter_cls.return_value = mock_arb

        out = str(tmp_path / "output.parquet")
        result = runner.invoke(app, ["export", out])
        assert result.exit_code == 0
        assert "Exported" in result.output

    @patch("arbiter.engine.Arbiter")
    def test_export_error_handling(
        self,
        mock_arbiter_cls: AsyncMock,
        tmp_path: Path,
    ) -> None:
        mock_arb = _make_mock_arbiter(
            fetch_all_markets=AsyncMock(side_effect=RuntimeError("fetch failed"))
        )
        mock_arbiter_cls.return_value = mock_arb

        out = str(tmp_path / "output.parquet")
        result = runner.invoke(app, ["export", out])
        assert result.exit_code == 0
        assert "Error" in result.output


# ── serve command ────────────────────────────────────────────────────────


class TestServeCommand:
    @patch("uvicorn.run")
    def test_serve_invokes_uvicorn(self, mock_run: AsyncMock) -> None:
        result = runner.invoke(app, ["serve", "--port", "9000"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("uvicorn.run")
    def test_serve_default_params(self, mock_run: AsyncMock) -> None:
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_run.assert_called_once()


# ── no-args ──────────────────────────────────────────────────────────────


class TestNoArgs:
    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # no_args_is_help=True causes exit code 0 or 2 depending on Typer version
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "arbiter" in result.output
