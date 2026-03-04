from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from core.types import BLOCKED, DIRECTIONS, FREE, SHELF, STATION, Cell


def load_layout(
    grid_path: str | Path,
    stations_path: str | Path,
    shelves_path: str | Path,
    spawn_path: str | Path,
) -> tuple[np.ndarray, dict[int, Cell], dict[int, Cell], list[Cell]]:
    """Load a CEDIS layout from disk.

    Returns
    -------
    grid : np.ndarray
    station_docks : dict[station_id, (x, y)]
    shelf_homes   : dict[shelf_id, (x, y)]
    spawns        : list[(x, y)]
    """
    grid: np.ndarray = np.load(grid_path)

    with open(stations_path, "r", encoding="utf-8") as f:
        stations_raw = json.load(f)
    with open(shelves_path, "r", encoding="utf-8") as f:
        shelves_raw = json.load(f)
    with open(spawn_path, "r", encoding="utf-8") as f:
        spawns_raw = json.load(f)

    station_docks = {int(s["station_id"]): tuple(s["dock"]) for s in stations_raw}
    shelf_homes = {int(a["shelf_id"]): tuple(a["home"]) for a in shelves_raw}
    spawns = [(int(p[0]), int(p[1])) for p in spawns_raw]

    return grid, station_docks, shelf_homes, spawns


def walkable_mask(grid: np.ndarray) -> np.ndarray:
    """Pre-compute a boolean mask of cells a robot can traverse."""
    return (grid == FREE) | (grid == STATION)


def adjacent_cells(cell: Cell) -> list[Cell]:
    """Return 4-connected neighbours of *cell*."""
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in DIRECTIONS]


def choose_adjacent_target(
    grid: np.ndarray,
    shelf_cell: Cell,
    walkable: np.ndarray | None = None,
) -> Cell | None:
    """Pick the first walkable cell next to a shelf (which is an obstacle)."""
    h, w = grid.shape
    if walkable is None:
        walkable = walkable_mask(grid)

    for nx, ny in adjacent_cells(shelf_cell):
        if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx]:
            return (nx, ny)
    return None


def bfs_reachable(grid: np.ndarray, start_cells: list[Cell]) -> np.ndarray:
    """Return a bool array indicating cells reachable from *start_cells*."""
    h, w = grid.shape
    wk = walkable_mask(grid)
    seen = np.zeros((h, w), dtype=bool)

    queue: deque[Cell] = deque()
    for x, y in start_cells:
        if 0 <= x < w and 0 <= y < h and wk[y, x]:
            seen[y, x] = True
            queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] and wk[ny, nx]:
                seen[ny, nx] = True
                queue.append((nx, ny))

    return seen
