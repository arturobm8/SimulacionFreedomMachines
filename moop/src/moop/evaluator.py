from __future__ import annotations

from typing import Any

from core.entities import Order
from core.layout_generator import generate_layout
from core.order_generator import generate_orders
from core.simulator import WarehouseSimulator
from base.strategy import GreedyStrategy


PENALTY_METRICS: dict[str, Any] = {
    "deadlocks": 9999,
    "throughput_per_1000_ticks": 0.0,
    "completed_orders": 0,
    "avg_order_time_ticks": 99999.0,
}


def evaluate_individual(
    robots: int,
    width: int,
    height: int,
    stations: int,
    orders_count: int,
    orders_burst: bool,
    strategy_window: int,
    seed: int,
    ticks: int,
) -> dict[str, Any]:
    """Run one full simulation in-memory and return its metrics.

    Returns penalty values if the configuration is infeasible (e.g. layout
    generation fails, not enough spawn points).
    """
    try:
        # Constraint: station row must fit
        if stations * 2 >= width - 2:
            return PENALTY_METRICS

        layout = generate_layout(
            seed=seed, width=width, height=height, stations=stations
        )

        station_ids = [s.station_id for s in layout.stations]
        shelf_ids = list(layout.shelves.keys())

        if not shelf_ids or not station_ids:
            return PENALTY_METRICS

        raw_orders = generate_orders(
            seed=seed,
            count=orders_count,
            burst=orders_burst,
            station_ids=station_ids,
            shelf_ids=shelf_ids,
        )

        orders = [
            Order(
                order_id=o["order_id"],
                shelf_id=o["shelf_id"],
                station_id=o["station_id"],
                tick_created=o["tick_created"],
            )
            for o in raw_orders
        ]

        strategy = GreedyStrategy(window=strategy_window)
        station_docks = {s.station_id: s.dock for s in layout.stations}

        sim = WarehouseSimulator(
            grid=layout.grid,
            station_docks=station_docks,
            shelf_homes=layout.shelves,
            robots=robots,
            spawn_points=layout.spawn_points,
            orders=orders,
            seed=seed,
            strategy=strategy,
        )

        sim.run(ticks)
        return sim.metrics()

    except Exception:
        return PENALTY_METRICS
