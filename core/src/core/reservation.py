from __future__ import annotations

from core.types import Cell


class ReservationTable:
    """Space-time reservation table for multi-robot collision avoidance.

    Prevents vertex collisions (two robots in the same cell at the same tick)
    and edge-swap collisions (A->B and B->A in the same tick).
    """

    __slots__ = ("_cells", "_edges")

    def __init__(self) -> None:
        # (x, y, tick) -> robot_id
        self._cells: dict[tuple[int, int, int], int] = {}
        # (x1, y1, x2, y2, tick) -> robot_id
        self._edges: dict[tuple[int, int, int, int, int], int] = {}

    # -- queries -------------------------------------------------------------

    def cell_free(self, cell: Cell, tick: int) -> bool:
        return (cell[0], cell[1], tick) not in self._cells

    def edge_free(self, a: Cell, b: Cell, tick: int) -> bool:
        return (a[0], a[1], b[0], b[1], tick) not in self._edges

    def can_move(self, current: Cell, next_cell: Cell, next_tick: int) -> bool:
        """Return *True* if moving ``current -> next_cell`` at *next_tick* is safe."""
        if not self.cell_free(next_cell, next_tick):
            return False
        # Check for edge-swap: the reverse edge must not be reserved
        if not self.edge_free(next_cell, current, next_tick):
            return False
        return True

    # -- reservations --------------------------------------------------------

    def confirm_move(
        self, robot_id: int, current: Cell, next_cell: Cell, next_tick: int
    ) -> None:
        """Reserve destination cell and forward edge."""
        x2, y2 = next_cell
        self._cells[(x2, y2, next_tick)] = robot_id
        x1, y1 = current
        self._edges[(x1, y1, x2, y2, next_tick)] = robot_id

    def confirm_wait(self, robot_id: int, cell: Cell, next_tick: int) -> None:
        """Reserve staying in *cell* during *next_tick*."""
        self._cells[(cell[0], cell[1], next_tick)] = robot_id

    # -- maintenance ---------------------------------------------------------

    def prune(self, before_tick: int) -> None:
        """Remove all entries for ticks strictly less than *before_tick*."""
        self._cells = {k: v for k, v in self._cells.items() if k[2] >= before_tick}
        self._edges = {k: v for k, v in self._edges.items() if k[4] >= before_tick}
