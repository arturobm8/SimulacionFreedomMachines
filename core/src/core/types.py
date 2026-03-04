from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Grid cell constants
# ---------------------------------------------------------------------------
FREE: int = 0
SHELF: int = 1
STATION: int = 2
BLOCKED: int = 3

# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------
Cell = tuple[int, int]

# ---------------------------------------------------------------------------
# 4-connected directions
# ---------------------------------------------------------------------------
DIRECTIONS: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

# ---------------------------------------------------------------------------
# Robot state machine
# ---------------------------------------------------------------------------

class RobotState(StrEnum):
    IDLE = "IDLE"
    TO_PICKUP = "TO_PICKUP"
    TO_STATION = "TO_STATION"
    RETURNING = "RETURNING"


# ---------------------------------------------------------------------------
# Strategy protocol  (implemented by base, moop, etc.)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Assignment:
    """One robot-to-order assignment produced by a strategy."""
    robot_id: int
    order_index: int       # index into the simulator's pending list
    pickup_cell: Cell
    route: list[Cell]


@runtime_checkable
class AssignmentStrategy(Protocol):
    def assign(
        self,
        idle_robots: list[tuple[int, Cell]],          # (robot_id, pos)
        pending_orders: list[tuple[int, int, Cell, Cell]],  # (order_idx, order_id, shelf_home, station_dock)
        grid: np.ndarray,
        walkable: np.ndarray,
        shelf_homes: dict[int, Cell],
        station_docks: dict[int, Cell],
        tick: int,
    ) -> list[Assignment]: ...
