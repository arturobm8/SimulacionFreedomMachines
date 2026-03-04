from __future__ import annotations

from dataclasses import dataclass, field

from core.types import Cell, RobotState


@dataclass(slots=True)
class Order:
    order_id: int
    shelf_id: int
    station_id: int
    tick_created: int
    tick_assigned: int | None = None
    tick_completed: int | None = None


@dataclass(slots=True)
class Robot:
    robot_id: int
    pos: Cell
    state: RobotState = RobotState.IDLE
    order_id: int | None = None
    shelf_home: Cell | None = None
    station_dock: Cell | None = None
    route: list[Cell] = field(default_factory=list)
    route_idx: int = 0
    wait_ticks: int = 0
    cells_moved: int = 0
    busy_ticks: int = 0
