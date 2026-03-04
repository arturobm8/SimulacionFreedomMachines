from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from config import SimConfig
from moop.config import OptConfig

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def main(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config"
    ),
    headless: bool = typer.Option(
        False, "--headless", help="Run without TUI, print results to stdout"
    ),
    baseline: bool = typer.Option(
        False, "--baseline", help="Run baseline simulation from config and save metrics"
    ),
    workers: Optional[int] = typer.Option(
        None, "--workers", "-w", help="Number of parallel workers"
    ),
) -> None:
    """MOOP — Multi-Objective Optimization for warehouse simulation."""
    sim_cfg = SimConfig.from_yaml(config)

    if baseline:
        _run_baseline(sim_cfg)
        return

    cfg = sim_cfg.optimization
    seed = sim_cfg.seed
    orders_count = sim_cfg.orders.count

    if workers is not None:
        cfg.n_workers = workers

    if headless:
        _run_headless(cfg, seed, orders_count)
    else:
        _run_tui(cfg, seed, orders_count)


def _run_baseline(sim_cfg: "SimConfig") -> None:
    """Run a single simulation with baseline params and save metrics."""
    import json

    from moop.evaluator import evaluate_individual

    bl = sim_cfg.baseline
    ticks = sim_cfg.ticks

    typer.echo(
        f"Baseline — seed={bl.seed}  robots={bl.robots}  "
        f"grid={bl.width}×{bl.height}  stations={bl.stations}  "
        f"orders={bl.orders_count}  burst={bl.orders_burst}  ticks={ticks}"
    )

    metrics = evaluate_individual(
        robots=bl.robots,
        width=bl.width,
        height=bl.height,
        stations=bl.stations,
        orders_count=bl.orders_count,
        orders_burst=bl.orders_burst,
        strategy_window=bl.strategy_window,
        seed=bl.seed,
        ticks=ticks,
    )

    # Add config params to metrics for traceability
    metrics["width"] = bl.width
    metrics["height"] = bl.height
    metrics["stations"] = bl.stations
    metrics["orders_burst"] = bl.orders_burst
    metrics["orders_count"] = bl.orders_count

    out_dir = Path(sim_cfg.paths.output_dir) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    typer.echo()
    for k, v in metrics.items():
        typer.echo(f"  {k}: {v}")
    typer.echo(f"\nSaved to {out_path}")


def _run_headless(cfg: OptConfig, seed: int, orders_count: int) -> None:
    """Run optimization without TUI, printing progress to stdout."""
    from moop.optimizer import run_optimization

    typer.echo(
        f"MOOP headless — {cfg.algorithm.upper()}  "
        f"pop={cfg.population_size}  gen={cfg.generations}  "
        f"GPU (WebGPU)  ticks={cfg.eval_ticks}  seed={seed}"
    )
    typer.echo()

    def on_gen(
        gen: int,
        X: np.ndarray,
        F: np.ndarray,
        params_list: list[dict],
        metrics_list: list[dict],
    ) -> None:
        evals = gen * cfg.population_size
        typer.echo(f"  Gen {gen}/{cfg.generations}  evals={evals}")

    result = run_optimization(cfg, on_generation=on_gen, n_workers=cfg.n_workers, seed=seed, orders_count=orders_count)

    typer.echo()
    typer.echo(f"Done — {len(result.params)} Pareto solutions")
    typer.echo()

    # Derived params that are not decision variables but appear in output
    derived_keys = ["height"]
    param_keys = list(cfg.var_names) + derived_keys

    extra_metric_keys = ["total_distance_cells", "high_contention_events"]

    # Print Pareto front
    if result.params:
        header = (
            ["seed"] + param_keys + ["orders_count"]
            + [obj.name for obj in cfg.objectives]
            + ["completion_pct"] + extra_metric_keys
        )
        typer.echo("  ".join(f"{h:>15}" for h in header))
        typer.echo("  ".join("-" * 15 for _ in header))

        for params, metrics in zip(result.params, result.metrics):
            row = [str(params.get("seed", ""))]
            for k in param_keys:
                val = params.get(k, "")
                if k == "orders_burst":
                    row.append("yes" if val else "no")
                else:
                    row.append(str(val))
            row.append(str(params.get("orders_count", "")))
            for obj in cfg.objectives:
                val = metrics.get(obj.metric_key)
                if val is not None and isinstance(val, float):
                    row.append(f"{val:.1f}")
                else:
                    row.append(str(val))
            oc = params.get("orders_count", 0)
            completed = metrics.get("completed_orders", 0)
            row.append(f"{completed / oc * 100:.1f}" if oc else "")
            for k in extra_metric_keys:
                row.append(str(metrics.get(k, "")))
            typer.echo("  ".join(f"{v:>15}" for v in row))

    # Save CSV
    if cfg.save_pareto and result.params:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "pareto_front.csv"

        fieldnames = (
            ["seed"] + param_keys + ["orders_count"]
            + [obj.metric_key for obj in cfg.objectives]
            + ["completion_pct"] + extra_metric_keys
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for params, metrics in zip(result.params, result.metrics):
                row = {}
                row["seed"] = params.get("seed", "")
                for k in param_keys:
                    row[k] = params.get(k, "")
                row["orders_count"] = params.get("orders_count", "")
                for obj in cfg.objectives:
                    row[obj.metric_key] = metrics.get(obj.metric_key, "")
                oc = params.get("orders_count", 0)
                completed = metrics.get("completed_orders", 0)
                row["completion_pct"] = round(completed / oc * 100, 1) if oc else ""
                for k in extra_metric_keys:
                    row[k] = metrics.get(k, "")
                writer.writerow(row)

        typer.echo(f"\nPareto front saved to {csv_path}")


def _run_tui(cfg: OptConfig, seed: int, orders_count: int) -> None:
    """Launch the Textual TUI."""
    from moop.tui.app import MoopApp

    app = MoopApp(cfg, n_workers=cfg.n_workers, seed=seed, orders_count=orders_count)
    app.run()
