"""Command-line interface for arbiter.

Provides a typer-based CLI for common analytics tasks without
writing Python code.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar, cast

import typer
from rich.console import Console
from rich.table import Table

from arbiter.exchanges.base import BaseExchange
from arbiter.output.snapshot import MarketSnapshot, SnapshotAnalysis, analyze_snapshot

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


def _print_snapshot_analysis(analysis: SnapshotAnalysis) -> None:
    """Render a concise replay summary to the console."""
    summary = Table(title="Snapshot Replay Summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Markets", str(analysis.market_count))
    summary.add_row("Matched pairs", str(analysis.pair_count))
    summary.add_row("Divergences", str(len(analysis.divergences)))
    summary.add_row("Binary violations", str(len(analysis.binary_violations)))
    summary.add_row(
        "Multi-outcome violations", str(len(analysis.multi_outcome_violations))
    )
    summary.add_row("Consensus disagreements", str(len(analysis.consensus)))
    console.print(summary)

    if analysis.exchange_counts:
        exchanges_table = Table(title="Snapshot Exchanges")
        exchanges_table.add_column("Exchange", style="cyan")
        exchanges_table.add_column("Markets", justify="right")
        for exchange, count in sorted(analysis.exchange_counts.items()):
            exchanges_table.add_row(exchange, str(count))
        console.print(exchanges_table)

    if analysis.divergences:
        div_table = Table(title="Top Snapshot Divergences")
        div_table.add_column("Event", style="cyan", max_width=44)
        div_table.add_column("Outcome")
        div_table.add_column("Exchange A")
        div_table.add_column("Price A", justify="right")
        div_table.add_column("Exchange B")
        div_table.add_column("Price B", justify="right")
        div_table.add_column("Spread", justify="right", style="bold red")

        for div in analysis.divergences[:10]:
            div_table.add_row(
                div.event[:44],
                div.outcome,
                div.exchange_a.value,
                f"{div.price_a:.1%}",
                div.exchange_b.value,
                f"{div.price_b:.1%}",
                f"{div.spread:.1%}",
            )
        console.print(div_table)


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
def consensus(
    min_disagreement: float = typer.Option(
        0.05,
        "--min-disagreement",
        "-d",
        help="Minimum YES-price disagreement to include",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum consensus rows to display",
    ),
    market_limit: int = typer.Option(
        50,
        "--market-limit",
        help="Max markets to fetch per exchange before matching",
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    exchanges: str | None = typer.Option(
        None,
        "--exchanges",
        "-e",
        help="Comma-separated exchange names (default: polymarket,manifold)",
    ),
) -> None:
    """Rank matched markets by cross-exchange consensus disagreement."""

    async def _consensus() -> None:
        from arbiter.engine import Arbiter

        exchange_list = _build_exchanges(_parse_exchange_names(exchanges))

        async with Arbiter(exchanges=exchange_list) as arb:
            console.print("[bold]Building consensus price view...[/bold]")

            try:
                rows = await arb.consensus(
                    min_disagreement=min_disagreement,
                    limit=limit,
                    market_limit=market_limit,
                )
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                return

            if output_json:
                console.print(json.dumps(rows, indent=2))
                return

            if not rows:
                console.print(
                    "[yellow]No consensus disagreements found above threshold.[/yellow]"
                )
                return

            table = Table(title="Consensus Price Disagreements")
            table.add_column("Event", style="cyan", max_width=44)
            table.add_column("Consensus YES", justify="right", style="green")
            table.add_column("Avg YES", justify="right")
            table.add_column("Disagreement", justify="right", style="bold red")
            table.add_column("Liquidity", justify="right")

            for row in rows:
                consensus_yes = cast(float, row["consensus_yes_price"])
                average_yes = cast(float, row["simple_average_yes_price"])
                disagreement = cast(float, row["disagreement_band"])
                total_liquidity = cast(float, row["total_liquidity"])
                table.add_row(
                    str(row["event"])[:44],
                    f"{consensus_yes:.1%}",
                    f"{average_yes:.1%}",
                    f"{disagreement:.1%}",
                    f"${total_liquidity:,.0f}",
                )

            console.print(table)

    _run_async(_consensus())


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
def snapshot(
    output_path: str = typer.Argument(help="Output snapshot JSON path"),
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        help="Max markets to fetch per exchange",
    ),
    include_inactive: bool = typer.Option(
        False,
        "--include-inactive",
        help="Include closed/resolved markets when exchanges support it",
    ),
    exchanges: str | None = typer.Option(
        None,
        "--exchanges",
        "-e",
        help="Comma-separated exchange names (default: polymarket,manifold)",
    ),
) -> None:
    """Write normalized live markets to a portable replay snapshot."""

    async def _snapshot() -> None:
        from arbiter.engine import Arbiter

        exchange_list = _build_exchanges(_parse_exchange_names(exchanges))

        async with Arbiter(exchanges=exchange_list) as arb:
            console.print("[bold]Fetching markets for snapshot...[/bold]")

            try:
                market_snapshot = await arb.market_snapshot(
                    active_only=not include_inactive,
                    limit=limit,
                )
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                return

            path = market_snapshot.write(output_path)
            console.print(
                "[green]"
                f"Wrote {market_snapshot.market_count} markets from "
                f"{len(market_snapshot.exchange_counts)} exchanges to {path}"
                "[/green]"
            )

    _run_async(_snapshot())


@app.command()
def replay(
    snapshot_path: str = typer.Argument(help="Snapshot JSON path to replay"),
    min_spread: float = typer.Option(
        0.02,
        "--min-spread",
        "-s",
        help="Minimum divergence spread to include",
    ),
    min_disagreement: float = typer.Option(
        0.05,
        "--min-disagreement",
        "-d",
        help="Minimum consensus disagreement to include",
    ),
    consensus_limit: int | None = typer.Option(
        20,
        "--consensus-limit",
        help="Maximum consensus rows to include; use 0 for none",
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    markdown_output: str | None = typer.Option(
        None,
        "--markdown-output",
        help="Optional Markdown report output path",
    ),
) -> None:
    """Replay analytics from a saved snapshot without calling exchanges."""

    try:
        market_snapshot = MarketSnapshot.from_file(snapshot_path)
        limit = None if consensus_limit is None else max(0, consensus_limit)
        analysis = analyze_snapshot(
            market_snapshot,
            min_spread=min_spread,
            min_disagreement=min_disagreement,
            consensus_limit=limit,
        )
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if markdown_output is not None:
        output_path = Path(markdown_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(analysis.to_markdown())
        console.print(f"[green]Wrote replay report to {output_path}[/green]")

    if output_json:
        console.print(json.dumps(analysis.model_dump(mode="json"), indent=2))
        return

    _print_snapshot_analysis(analysis)


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
def exchange_status(
    exchanges: str | None = typer.Option(
        None,
        "--exchanges",
        "-e",
        help="Comma-separated exchange names (default: all configured)",
    ),
) -> None:
    """Check health and latency of configured exchanges."""

    async def _status() -> None:
        names = _parse_exchange_names(exchanges) or list(_EXCHANGE_REGISTRY.keys())
        exchange_list = _build_exchanges(names)

        table = Table(title="Exchange Health Status")
        table.add_column("Exchange", style="cyan")
        table.add_column("Status")
        table.add_column("Latency", justify="right")
        table.add_column("Error", style="dim")

        for ex in exchange_list:
            result = await ex.health_check()
            status_str = (
                "[green]OK[/green]" if result["status"] == "ok" else "[red]ERROR[/red]"
            )
            latency_str = f"{result['latency_ms']:.1f} ms"
            error_str = result.get("error", "")
            table.add_row(
                result["exchange"],
                status_str,
                latency_str,
                error_str[:60] if error_str else "",
            )
            await ex.close()

        console.print(table)

    _run_async(_status())


@app.command()
def version() -> None:
    """Show the arbiter version."""
    console.print("arbiter 0.1.0")


if __name__ == "__main__":
    app()
