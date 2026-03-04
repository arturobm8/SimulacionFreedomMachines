from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog

from config import SimConfig, setup_logging
from core import (
    WarehouseSimulator,
    generate_layout,
    generate_orders,
    load_layout,
    load_orders,
    save_layout,
    save_orders,
)

from base.strategy import GreedyStrategy

log = structlog.get_logger()


def _load_sim(cfg: SimConfig) -> tuple[WarehouseSimulator, np.ndarray]:
    """Load layout, orders, and build a simulator from *cfg*."""
    grid, station_docks, shelf_homes, spawns = load_layout(
        cfg.scenario_path("layout.npy"),
        cfg.scenario_path("stations.json"),
        cfg.scenario_path("shelves.json"),
        cfg.scenario_path("spawn.json"),
    )
    orders = load_orders(cfg.scenario_path("orders.json"))
    strategy = GreedyStrategy()
    sim = WarehouseSimulator(
        grid=grid,
        station_docks=station_docks,
        shelf_homes=shelf_homes,
        robots=cfg.robots,
        spawn_points=spawns,
        orders=orders,
        seed=cfg.seed,
        strategy=strategy,
    )
    return sim, grid


def run_generate_layout(cfg: SimConfig) -> None:
    """Generate and save a warehouse layout using config defaults."""
    log.info("generate_layout.start", seed=cfg.seed, width=cfg.warehouse.width,
             height=cfg.warehouse.height, stations=cfg.warehouse.stations)

    data = generate_layout(
        seed=cfg.seed,
        width=cfg.warehouse.width,
        height=cfg.warehouse.height,
        stations=cfg.warehouse.stations,
    )

    out_dir = Path(cfg.paths.output_dir) / cfg.scenario
    save_layout(data, out_dir)

    log.info(
        "generate_layout.done",
        shelves=len(data.shelves),
        stations=len(data.stations),
        spawn_points=len(data.spawn_points),
        output_dir=str(out_dir),
    )


def run_generate_orders(cfg: SimConfig) -> None:
    """Generate and save orders using config defaults."""
    stations_path = cfg.scenario_path("stations.json")
    shelves_path = cfg.scenario_path("shelves.json")

    with open(stations_path, "r", encoding="utf-8") as f:
        stations_raw = json.load(f)
    with open(shelves_path, "r", encoding="utf-8") as f:
        shelves_raw = json.load(f)

    station_ids = [s["station_id"] for s in stations_raw]
    shelf_ids = [a["shelf_id"] for a in shelves_raw]

    log.info("generate_orders.start", seed=cfg.seed, count=cfg.orders.count,
             burst=cfg.orders.burst)

    orders = generate_orders(
        seed=cfg.seed,
        count=cfg.orders.count,
        burst=cfg.orders.burst,
        station_ids=station_ids,
        shelf_ids=shelf_ids,
    )

    out_path = cfg.scenario_path("orders.json")
    save_orders(orders, cfg.seed, out_path)

    log.info("generate_orders.done", count=len(orders), path=str(out_path))


def run_simulation(cfg: SimConfig) -> dict:
    """Run the greedy simulation and return metrics."""
    sim, _grid = _load_sim(cfg)

    log.info("simulation.start", seed=cfg.seed, robots=cfg.robots, ticks=cfg.ticks,
             scenario=cfg.scenario)

    sim.run(cfg.ticks)
    m = sim.metrics()

    metrics_path = cfg.scenario_path("metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    log.info("simulation.done", completed=m["completed_orders"],
             total=m["total_orders"], throughput=m["throughput_per_1000_ticks"])

    return m


def run_full_pipeline(cfg: SimConfig, viz: bool = False) -> dict:
    """End-to-end pipeline: layout -> orders -> simulation -> (optional) viz."""
    setup_logging(cfg.logging)

    # 1. Generate layout
    run_generate_layout(cfg)

    # 2. Generate orders
    run_generate_orders(cfg)

    # 3. Run simulation
    m = run_simulation(cfg)

    # 4. Optionally visualize
    if viz:
        from render import run_visualize

        sim_viz, grid = _load_sim(cfg)
        run_visualize(cfg, sim_viz, grid)

    return m
