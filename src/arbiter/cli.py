"""Command-line interface for arbiter.

Provides a typer-based CLI for common analytics tasks without
writing Python code.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from arbiter.exchanges.base import BaseExchange

app = typer.Typer(
    name="arbiter",
    help="Cross-exchange prediction market analytics engine",
    no_args_is_help=True,
)
console = Console()

# Registry of known exchange constructors keyed by lowercase name.
_EXCHANGE_REGISTRY: dict[str, str] = {
    "polymarket": "arbiter.exchanges.polymarket.PolymarketExchange",
    "manifold": "arbiter.exchanges.manifold.ManifoldExchange",
    "metaculus": "arbiter.exchanges.metaculus.MetaculusExchange",
    "kalshi": "arbiter.exchanges.kalshi.KalshiExchange",
}


def _build_exchanges(names: list[str] | None = None) -> list[BaseExchange]:
    """Create exchange connector instances from a list of names.

    Args:
        names: Exchange names (e.g. ``["polymarket", "manifold"]``).
            Defaults to ``["polymarket", "manifold"]`` when *None*.

    Returns:
        List of exchange connector instances.
    """
    import importlib

    if names is None:
        names = ["polymarket", "manifold"]

    exchanges: list[BaseExchange] = []
    for name in names:
        key = name.strip().lower()
        if key not in _EXCHANGE_REGISTRY:
            console.print(
                f"[red]Unknown exchange '{name}'. "
                f"Available: {', '.join(sorted(_EXCHANGE_REGISTRY))}[/red]"
            )
            raise typer.Exit(code=1)
        module_path, class_name = _EXCHANGE_REGISTRY[key].rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        exchanges.append(cls())
    return exchanges


T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async function synchronously."""
    return asyncio.run(coro)


def _parse_exchange_names(raw: str | None) -> list[str] | None:
    """Split a comma-separated exchange string into a list, or return None."""
    if raw is None:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


@app.command()
def scan(
    min_spread: float = typer.Option(
        0.02, "--min-spread", "-s", help="Minimum spread to report"
    ),
    limit: int = typer.Option(
        50, "--limit", "-n", help="Max markets to fetch per exchange"
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    exchanges: str | None = typer.Option(
        None,
        "--exchanges",
        "-e",
        help="Comma-separated exchange names (default: polymarket,manifold)",
    ),
) -> None:
    """Scan for cross-exchange price divergences."""

    async def _scan() -> None:
        from arbiter.engine import Arbiter

        exchange_list = _build_exchanges(_parse_exchange_names(exchanges))

        async with Arbiter(exchanges=exchange_list) as arb:
            console.print("[bold]Fetching markets...[/bold]")

            try:
                divergences = await arb.divergences(
                    min_spread=min_spread,
                    limit=limit,
                )
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                return

            if output_json:
                data = [d.model_dump(mode="json") for d in divergences]
                console.print(json.dumps(data, indent=2))
                return

            if not divergences:
                console.print("[yellow]No divergences found above threshold.[/yellow]")
                return

            table = Table(title="Price Divergences")
            table.add_column("Event", style="cyan", max_width=40)
            table.add_column("Outcome", style="green")
            table.add_column("Exchange A")
            table.add_column("Price A", justify="right")
            table.add_column("Exchange B")
            table.add_column("Price B", justify="right")
            table.add_column("Spread", justify="right", style="bold red")
            table.add_column("Spread %", justify="right")

            for d in divergences:
                table.add_row(
                    d.event[:40],
                    d.outcome,
                    d.exchange_a.value,
                    f"{d.price_a:.3f}",
                    d.exchange_b.value,
                    f"{d.price_b:.3f}",
                    f"{d.spread:.3f}",
                    f"{d.spread_pct:.1%}",
                )

            console.print(table)

    _run_async(_scan())


@app.command()
def violations(
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    exchanges: str | None = typer.Option(
        None,
        "--exchanges",
        "-e",
        help="Comma-separated exchange names (default: polymarket,manifold)",
    ),
) -> None:
    """Scan for probability violations across exchanges."""

    async def _violations() -> None:
        from arbiter.engine import Arbiter

        exchange_list = _build_exchanges(_parse_exchange_names(exchanges))

        async with Arbiter(exchanges=exchange_list) as arb:
            console.print("[bold]Scanning for violations...[/bold]")

            try:
                binary_v, multi_v = await arb.violations()
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                return

            if output_json:
                data = {
                    "binary": [v.model_dump(mode="json") for v in binary_v],
                    "multi_outcome": [v.model_dump(mode="json") for v in multi_v],
                }
                console.print(json.dumps(data, indent=2))
                return

            if not binary_v and not multi_v:
                console.print("[green]No violations detected.[/green]")
                return

            if binary_v:
                table = Table(title="Binary Probability Violations")
                table.add_column("Market", style="cyan", max_width=40)
                table.add_column("Exchange")
                table.add_column("YES", justify="right")
                table.add_column("NO", justify="right")
                table.add_column("Sum", justify="right", style="bold red")
                table.add_column("Arb", justify="right", style="bold green")

                for v in binary_v:
                    table.add_row(
                        v.market[:40],
                        v.exchange.value,
                        f"{v.yes_price:.3f}",
                        f"{v.no_price:.3f}",
                        f"{v.price_sum:.3f}",
                        f"{v.implied_arb:.3f}",
                    )
                console.print(table)

            if multi_v:
                table = Table(title="Multi-Outcome Violations")
                table.add_column("Market", style="cyan", max_width=40)
                table.add_column("Exchange")
                table.add_column("Sum", justify="right", style="bold red")
                table.add_column("Deviation", justify="right")

                for mv in multi_v:
                    table.add_row(
                        mv.market[:40],
                        mv.exchange.value,
                        f"{mv.price_sum:.3f}",
                        f"{mv.deviation:.3f}",
                    )
                console.print(table)

    _run_async(_violations())


@app.command()
def export(
    output_path: str = typer.Argument(help="Output file path (.parquet or .csv)"),
    output_format: str = typer.Option(
        "parquet", "--format", "-f", help="Export format: parquet or csv"
    ),
    exchanges: str | None = typer.Option(
        None,
        "--exchanges",
        "-e",
        help="Comma-separated exchange names (default: polymarket,manifold)",
    ),
) -> None:
    """Export market data to a file."""

    async def _export() -> None:
        from arbiter.engine import Arbiter

        exchange_list = _build_exchanges(_parse_exchange_names(exchanges))

        async with Arbiter(exchanges=exchange_list) as arb:
            console.print("[bold]Fetching markets for export...[/bold]")

            try:
                await arb.fetch_all_markets(active_only=False)
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                return

            all_markets = []
            for markets in arb._market_cache.values():
                all_markets.extend(markets)

            exporter = arb._exporter
            if output_format.lower() == "csv":
                exporter.export_markets_to_csv(all_markets, output_path)
            else:
                exporter.export_markets_to_parquet(all_markets, output_path)

            console.print(
                f"[green]Exported {len(all_markets)} markets to {output_path}[/green]"
            )

    _run_async(_export())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
) -> None:
    """Start the arbiter REST API server."""
    import uvicorn

    from arbiter.output.api import app as api_app

    console.print(f"[bold]Starting arbiter API on {host}:{port}[/bold]")
    uvicorn.run(api_app, host=host, port=port)


@app.command()
def version() -> None:
    """Show the arbiter version."""
    console.print("arbiter 0.1.0")


if __name__ == "__main__":
    app()
