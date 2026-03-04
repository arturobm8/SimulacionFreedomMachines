from __future__ import annotations

import heapq

import numpy as np

from core.types import DIRECTIONS, Cell


def astar(
    walkable: np.ndarray,
    start: Cell,
    goal: Cell,
) -> list[Cell] | None:
    """A* on a 4-connected grid using a pre-computed boolean walkable mask.

    Parameters
    ----------
    walkable : np.ndarray[bool]
        ``True`` where a robot can traverse.
    start, goal : Cell
        ``(x, y)`` coordinates.

    Returns
    -------
    list[Cell] | None
        Path including *start* and *goal*, or ``None`` if unreachable.
    """
    h, w = walkable.shape
    sx, sy = start
    gx, gy = goal

    if (
        not (0 <= sx < w and 0 <= sy < h)
        or not (0 <= gx < w and 0 <= gy < h)
        or not walkable[sy, sx]
        or not walkable[gy, gx]
    ):
        return None

    # heap entries: (f, g, x, y)  — pure tuples for fast comparison
    open_heap: list[tuple[int, int, int, int]] = []
    h0 = abs(sx - gx) + abs(sy - gy)
    heapq.heappush(open_heap, (h0, 0, sx, sy))

    came_from: dict[Cell, Cell] = {}
    g_cost: dict[Cell, int] = {(sx, sy): 0}
    closed: set[Cell] = set()

    while open_heap:
        _, g_cur, cx, cy = heapq.heappop(open_heap)
        cur: Cell = (cx, cy)

        if cur in closed:
            continue
        closed.add(cur)

        if cx == gx and cy == gy:
            path: list[Cell] = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or not walkable[ny, nx]:
                continue
            new_g = g_cur + 1
            nbr: Cell = (nx, ny)
            if new_g < g_cost.get(nbr, 1 << 30):
                g_cost[nbr] = new_g
                came_from[nbr] = cur
                f = new_g + abs(nx - gx) + abs(ny - gy)
                heapq.heappush(open_heap, (f, new_g, nx, ny))

    return None
