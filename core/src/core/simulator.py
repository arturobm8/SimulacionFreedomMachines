from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import numpy as np

from core.entities import Order, Robot
from core.grid import adjacent_cells, choose_adjacent_target, walkable_mask
from core.pathfinding import astar
from core.reservation import ReservationTable
from core.types import AssignmentStrategy, Cell, RobotState


class WarehouseSimulator:
    """Discrete-event warehouse simulator.

    The assignment strategy is pluggable via the ``strategy`` argument.  The
    simulator owns the tick loop, movement proposals, and collision resolution.
    """

    def __init__(
        self,
        grid: np.ndarray,
        station_docks: dict[int, Cell],
        shelf_homes: dict[int, Cell],
        robots: int,
        spawn_points: list[Cell],
        orders: list[Order],
        seed: int,
        strategy: AssignmentStrategy,
    ) -> None:
        self.grid = grid
        self.walkable = walkable_mask(grid)
        self.station_docks = station_docks
        self.shelf_homes = shelf_homes
        self.orders = orders
        self.seed = seed
        self.tick: int = 0
        self.strategy = strategy

        self.reservation = ReservationTable()

        # -- cache adjacent pickup cells for each shelf --------------------
        self._pickup_cache: dict[Cell, Cell | None] = {}
        for home in shelf_homes.values():
            if home not in self._pickup_cache:
                self._pickup_cache[home] = choose_adjacent_target(
                    grid, home, self.walkable
                )

        # -- robots --------------------------------------------------------
        if len(spawn_points) < robots:
            raise RuntimeError("Not enough spawn points for the number of robots.")
        self.robots_list: list[Robot] = [
            Robot(robot_id=i, pos=spawn_points[i]) for i in range(robots)
        ]
        for r in self.robots_list:
            self.reservation.confirm_wait(r.robot_id, r.pos, 0)

        # -- metrics -------------------------------------------------------
        self.vertex_collisions: int = 0
        self.edge_swaps: int = 0
        self.deadlock_count: int = 0
        self.high_contention_events: int = 0

        # -- order management ----------------------------------------------
        self._order_by_id: dict[int, Order] = {o.order_id: o for o in orders}
        self.pending: list[int] = []  # indices into self.orders
        self._unreleased: deque[int] = deque(
            sorted(range(len(self.orders)), key=lambda i: self.orders[i].tick_created)
        )

    # -- order lifecycle ---------------------------------------------------

    def _release_orders(self) -> None:
        while self._unreleased and self.orders[self._unreleased[0]].tick_created <= self.tick:
            self.pending.append(self._unreleased.popleft())

    def _complete_order(self, order_id: int) -> None:
        self._order_by_id[order_id].tick_completed = self.tick

    # -- strategy delegation -----------------------------------------------

    def _assign_orders(self) -> None:
        idle = [
            (r.robot_id, r.pos)
            for r in self.robots_list
            if r.state == RobotState.IDLE
        ]
        if not idle or not self.pending:
            return

        pending_info: list[tuple[int, int, Cell, Cell]] = []
        for pi in self.pending:
            o = self.orders[pi]
            home = self.shelf_homes[o.shelf_id]
            dock = self.station_docks[o.station_id]
            pending_info.append((pi, o.order_id, home, dock))

        assignments = self.strategy.assign(
            idle_robots=idle,
            pending_orders=pending_info,
            grid=self.grid,
            walkable=self.walkable,
            shelf_homes=self.shelf_homes,
            station_docks=self.station_docks,
            tick=self.tick,
        )

        assigned_indices: set[int] = set()
        for a in assignments:
            robot = self.robots_list[a.robot_id]
            order = self.orders[a.order_index]
            order.tick_assigned = self.tick
            assigned_indices.add(a.order_index)

            robot.order_id = order.order_id
            robot.shelf_home = self.shelf_homes[order.shelf_id]
            robot.station_dock = self.station_docks[order.station_id]
            robot.state = RobotState.TO_PICKUP
            robot.route = a.route
            robot.route_idx = 0

        # Remove assigned from pending (preserve order)
        if assigned_indices:
            self.pending = [pi for pi in self.pending if pi not in assigned_indices]

    # -- route planning on arrival -----------------------------------------

    def _plan_next_leg(self, r: Robot) -> None:
        if not r.route or r.route_idx != len(r.route) - 1:
            return

        if r.state == RobotState.TO_PICKUP:
            r.state = RobotState.TO_STATION
            route = astar(self.walkable, r.pos, r.station_dock)
            if route is None:
                r.state = RobotState.TO_PICKUP
                return
            r.route = route
            r.route_idx = 0

        elif r.state == RobotState.TO_STATION:
            r.state = RobotState.RETURNING
            pickup = self._pickup_cache.get(r.shelf_home) if r.shelf_home else None
            if pickup is None:
                r.state = RobotState.TO_STATION
                return
            route = astar(self.walkable, r.pos, pickup)
            if route is None:
                r.state = RobotState.TO_STATION
                return
            r.route = route
            r.route_idx = 0

        elif r.state == RobotState.RETURNING:
            if r.order_id is not None:
                self._complete_order(r.order_id)
            r.state = RobotState.IDLE
            r.order_id = None
            r.shelf_home = None
            r.station_dock = None
            r.route = []
            r.route_idx = 0

    # -- main tick ---------------------------------------------------------

    def step(self) -> None:
        self._release_orders()
        self._assign_orders()

        # Propose movements
        proposals: dict[int, Cell] = {}
        for r in self.robots_list:
            if r.state != RobotState.IDLE:
                r.busy_ticks += 1

            self._plan_next_leg(r)

            if r.state == RobotState.IDLE or not r.route:
                proposals[r.robot_id] = r.pos
                continue

            if r.route_idx < len(r.route) - 1:
                proposals[r.robot_id] = r.route[r.route_idx + 1]
            else:
                proposals[r.robot_id] = r.pos

        next_tick = self.tick + 1
        anyone_moved = False

        # Deterministic order: ascending robot_id
        for r in self.robots_list:
            current = r.pos
            proposed = proposals[r.robot_id]

            if proposed == current:
                self.reservation.confirm_wait(r.robot_id, current, next_tick)
                continue

            if self.reservation.can_move(current, proposed, next_tick):
                self.reservation.confirm_move(r.robot_id, current, proposed, next_tick)
                r.pos = proposed
                r.route_idx += 1
                r.cells_moved += 1
                anyone_moved = True
            else:
                r.wait_ticks += 1
                self.high_contention_events += 1
                self.reservation.confirm_wait(r.robot_id, current, next_tick)

        if not anyone_moved and any(r.state != RobotState.IDLE for r in self.robots_list):
            self.deadlock_count += 1

        self.tick = next_tick

        # Periodic pruning to bound memory
        if self.tick % 500 == 0:
            self.reservation.prune(self.tick - 50)

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    async def arun(self, ticks: int, yield_every: int = 100) -> None:
        """Async run that yields control periodically for progress updates."""
        for i in range(ticks):
            self.step()
            if (i + 1) % yield_every == 0:
                await asyncio.sleep(0)

    # -- accessors ---------------------------------------------------------

    def robot_positions(self) -> list[Cell]:
        return [r.pos for r in self.robots_list]

    def robot_states(self) -> list[RobotState]:
        return [r.state for r in self.robots_list]

    def robot_ids(self) -> list[int]:
        return [r.robot_id for r in self.robots_list]

    # -- metrics -----------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        completed = [o for o in self.orders if o.tick_completed is not None]
        n_completed = len(completed)

        avg_order_time: float | None = None
        if n_completed > 0:
            avg_order_time = float(
                np.mean([o.tick_completed - o.tick_created for o in completed])
            )

        utilization = [r.busy_ticks / max(1, self.tick) for r in self.robots_list]
        waits = [r.wait_ticks for r in self.robots_list]

        throughput = 0.0
        if self.tick > 0:
            throughput = n_completed / (self.tick / 1000.0)

        return {
            "seed": self.seed,
            "final_tick": self.tick,
            "robots": len(self.robots_list),
            "total_orders": len(self.orders),
            "completed_orders": n_completed,
            "avg_order_time_ticks": avg_order_time,
            "throughput_per_1000_ticks": throughput,
            "avg_wait_ticks": float(np.mean(waits)) if waits else 0.0,
            "avg_utilization": float(np.mean(utilization)) if utilization else 0.0,
            "vertex_collisions": self.vertex_collisions,
            "edge_swaps": self.edge_swaps,
            "deadlocks": self.deadlock_count,
            "high_contention_events": self.high_contention_events,
            "total_distance_cells": int(sum(r.cells_moved for r in self.robots_list)),
        }
