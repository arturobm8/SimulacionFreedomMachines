from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.entities import Order


def generate_orders(
    seed: int,
    count: int,
    burst: bool,
    station_ids: list[int],
    shelf_ids: list[int],
) -> list[dict]:
    """Generate discrete order events.

    Parameters
    ----------
    burst : bool
        If *True*, 70 % of orders are created in ticks 0-2000; the rest
        spread up to tick 10 000. Otherwise all orders start at tick 0.

    Returns a list of raw order dicts ready for JSON serialization.
    """
    rng = np.random.default_rng(seed)
    orders: list[dict] = []

    for i in range(count):
        station_id = int(rng.choice(station_ids))
        shelf_id = int(rng.choice(shelf_ids))

        if burst:
            if rng.random() < 0.70:
                tick_created = int(rng.integers(0, 2001))
            else:
                tick_created = int(rng.integers(0, 10001))
        else:
            tick_created = 0

        orders.append(
            {
                "order_id": i,
                "shelf_id": shelf_id,
                "station_id": station_id,
                "tick_created": tick_created,
            }
        )

    return orders


def save_orders(orders: list[dict], seed: int, path: Path) -> None:
    """Write orders to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "orders": orders}, f, indent=2, ensure_ascii=False)


def load_orders(path: str | Path) -> list[Order]:
    """Read orders from a JSON file and return ``Order`` objects."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Order(
            order_id=int(p["order_id"]),
            shelf_id=int(p["shelf_id"]),
            station_id=int(p["station_id"]),
            tick_created=int(p.get("tick_created", 0)),
        )
        for p in data.get("orders", [])
    ]
