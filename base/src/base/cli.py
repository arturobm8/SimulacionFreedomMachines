from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from config import SimConfig

app = typer.Typer(
    name="base",
    help="Greedy strategy for multi-robot warehouse simulation.",
    invoke_without_command=True,
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    seed: int | None = typer.Option(None, help="Random seed"),
    robots: int | None = typer.Option(None, help="Number of robots"),
    ticks: int | None = typer.Option(None, help="Simulation ticks"),
    scenario: str | None = typer.Option(None, help="Scenario name"),
    viz: bool = typer.Option(False, "--viz", help="Generate video + heatmaps"),
) -> None:
    """Run the full simulation pipeline (layout -> orders -> sim -> metrics)."""
    cfg = SimConfig.from_yaml(config)

    if seed is not None:
        cfg.seed = seed
    if robots is not None:
        cfg.robots = robots
    if ticks is not None:
        cfg.ticks = ticks
    if scenario is not None:
        cfg.scenario = scenario
    if viz:
        cfg.visualization.enabled = True

    console.print(
        f"[blue]Running pipeline:[/blue] scenario={cfg.scenario} "
        f"robots={cfg.robots} ticks={cfg.ticks} viz={cfg.visualization.enabled}"
    )

    from base.runner import run_full_pipeline

    m = run_full_pipeline(cfg, viz=cfg.visualization.enabled)

    table = Table(title="Simulation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in m.items():
        val = f"{v:.2f}" if isinstance(v, float) else str(v)
        table.add_row(k, val)

    console.print(table)

    if cfg.visualization.enabled:
        console.print(f"[green]Visualization saved to [bold]{cfg.scenario}[/bold][/green]")
