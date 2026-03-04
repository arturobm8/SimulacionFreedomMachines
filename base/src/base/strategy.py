from __future__ import annotations

import numpy as np

from core.grid import choose_adjacent_target
from core.pathfinding import astar
from core.types import Assignment, Cell


class GreedyStrategy:
    """Greedy nearest-neighbour order assignment.

    For each idle robot, pick the closest pending order (by Manhattan
    distance to the shelf's pickup cell) among the first *window* candidates.
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window

    def assign(
        self,
        idle_robots: list[tuple[int, Cell]],
        pending_orders: list[tuple[int, int, Cell, Cell]],
        grid: np.ndarray,
        walkable: np.ndarray,
        shelf_homes: dict[int, Cell],
        station_docks: dict[int, Cell],
        tick: int,
    ) -> list[Assignment]:
        if not idle_robots or not pending_orders:
            return []

        assignments: list[Assignment] = []
        used_orders: set[int] = set()

        for robot_id, robot_pos in idle_robots:
            best_idx: int | None = None
            best_dist = 1 << 30
            best_pickup: Cell | None = None

            window = min(self.window, len(pending_orders))
            for i in range(window):
                order_idx, _order_id, shelf_home, _station_dock = pending_orders[i]
                if order_idx in used_orders:
                    continue

                pickup = choose_adjacent_target(grid, shelf_home, walkable)
                if pickup is None:
                    continue

                dist = abs(robot_pos[0] - pickup[0]) + abs(robot_pos[1] - pickup[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    best_pickup = pickup

            if best_idx is None or best_pickup is None:
                continue

            order_idx = pending_orders[best_idx][0]
            route = astar(walkable, robot_pos, best_pickup)
            if route is None:
                continue

            used_orders.add(order_idx)
            assignments.append(
                Assignment(
                    robot_id=robot_id,
                    order_index=order_idx,
                    pickup_cell=best_pickup,
                    route=route,
                )
            )

        return assignments
