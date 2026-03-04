from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.types import BLOCKED, FREE, SHELF, STATION, Cell


@dataclass(slots=True)
class StationInfo:
    station_id: int
    dock: Cell
    cell: Cell


@dataclass(slots=True)
class LayoutData:
    seed: int
    width: int
    height: int
    grid: np.ndarray
    stations: list[StationInfo] = field(default_factory=list)
    shelves: dict[int, Cell] = field(default_factory=dict)
    spawn_points: list[Cell] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_rect(grid: np.ndarray, x0: int, y0: int, w: int, h: int, value: int) -> None:
    gh, gw = grid.shape
    x1 = max(0, min(gw, x0 + w))
    y1 = max(0, min(gh, y0 + h))
    x0 = max(0, min(gw, x0))
    y0 = max(0, min(gh, y0))
    grid[y0:y1, x0:x1] = value


def _in_bounds(grid: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_layout(seed: int, width: int, height: int, stations: int) -> LayoutData:
    """Generate a realistic CEDIS warehouse layout.

    Architecture:
    - Southern front: station row (y=H-2) + buffer lane (y=H-3)
    - Parking / charging zone: 12x8 cells (bottom-left)
    - Storage region: 3x2 macro-blocks with internal rack patterns
    - Cross-aisles: 2-cell corridors every ~10 rows
    """
    rng = np.random.default_rng(seed)
    grid = np.full((height, width), FREE, dtype=np.int8)

    # Border (1-cell BLOCKED perimeter)
    _fill_rect(grid, 0, 0, width, 1, BLOCKED)
    _fill_rect(grid, 0, height - 1, width, 1, BLOCKED)
    _fill_rect(grid, 0, 0, 1, height, BLOCKED)
    _fill_rect(grid, width - 1, 0, 1, height, BLOCKED)

    # Southern station row
    y_station = height - 2
    y_buffer = height - 3

    total_station_width = stations * 2
    x_start = (width - total_station_width) // 2
    station_list: list[StationInfo] = []

    # Buffer lane
    for x in range(1, width - 1):
        if grid[y_buffer, x] != BLOCKED:
            grid[y_buffer, x] = FREE

    for i in range(stations):
        sx = x_start + i * 2
        for dx in (0, 1):
            if _in_bounds(grid, sx + dx, y_station):
                grid[y_station, sx + dx] = STATION
        dock = (sx, y_buffer)
        cell = (sx, y_station)
        station_list.append(StationInfo(station_id=i, dock=dock, cell=cell))

    # Apron above buffer
    _fill_rect(grid, 1, y_buffer - 2, width - 2, 2, FREE)

    # Parking / charging zone
    park_w, park_h = 12, 8
    park_x0, park_y0 = 2, height - (park_h + 5)
    _fill_rect(grid, park_x0, park_y0, park_w, park_h, FREE)

    # Storage region boundaries
    y_top = 2
    y_bottom = y_buffer - 4
    x_left = 2
    x_right = width - 3

    storage_h = y_bottom - y_top
    storage_w = x_right - x_left + 1

    # 3x2 macro-blocks with main aisles
    aisle_w = 2
    cols, rows = 3, 2

    total_v_aisles = (cols + 1) * aisle_w
    block_w = (storage_w - total_v_aisles) // cols

    h_aisle = 2
    total_h_aisles = (rows + 1) * h_aisle
    block_h = (storage_h - total_h_aisles) // rows

    # Vertical main aisles
    x = x_left
    for _ in range(cols + 1):
        _fill_rect(grid, x, y_top, aisle_w, storage_h, FREE)
        x += aisle_w + block_w

    # Horizontal main aisles
    y = y_top
    for _ in range(rows + 1):
        _fill_rect(grid, x_left, y, storage_w, h_aisle, FREE)
        y += h_aisle + block_h

    # Fill blocks with shelf pattern
    shelves: dict[int, Cell] = {}
    shelf_id = 0

    def _fill_block(bx0: int, by0: int, bw: int, bh: int) -> None:
        nonlocal shelf_id
        ix0, iy0 = bx0 + 1, by0 + 1
        iw, ih = max(0, bw - 2), max(0, bh - 2)
        if iw <= 0 or ih <= 0:
            return

        col = 0
        xx = ix0
        while xx < ix0 + iw:
            if col % 3 in (0, 1):
                for yy in range(iy0, iy0 + ih):
                    if (yy - iy0) % 17 == 0 and (xx - ix0) % 11 == 0:
                        continue
                    if grid[yy, xx] == FREE:
                        grid[yy, xx] = SHELF
                        shelves[shelf_id] = (xx, yy)
                        shelf_id += 1
            else:
                for yy in range(iy0, iy0 + ih):
                    if grid[yy, xx] != BLOCKED:
                        grid[yy, xx] = FREE
            col += 1
            xx += 1

    y = y_top + h_aisle
    for _ in range(rows):
        x = x_left + aisle_w
        for _ in range(cols):
            _fill_block(x, y, block_w, block_h)
            x += block_w + aisle_w
        y += block_h + h_aisle

    # Cross-aisles every 10 rows
    for yy in range(y_top + 5, y_bottom, 10):
        _fill_rect(grid, x_left, yy, storage_w, 2, FREE)

    # Corridor connecting parking to apron
    corr_x = park_x0 + park_w + 2
    _fill_rect(grid, corr_x, park_y0 - 15, 2, (height - 2) - (park_y0 - 15), FREE)

    # Spawn points inside parking
    spawn_points: list[Cell] = []
    for yy in range(park_y0, park_y0 + park_h):
        for xx in range(park_x0, park_x0 + park_w):
            if grid[yy, xx] == FREE:
                spawn_points.append((xx, yy))
    rng.shuffle(spawn_points)
    spawn_points = spawn_points[:200]

    if not spawn_points:
        raise RuntimeError("No spawn points generated.")

    # Reachability validation
    from core.grid import bfs_reachable

    reachable = bfs_reachable(grid, [spawn_points[0]])
    for st in station_list:
        dx, dy = st.dock
        if not reachable[dy, dx]:
            _fill_rect(grid, dx, dy - 5, 2, 6, FREE)

    # Second pass (validate fix)
    _ = bfs_reachable(grid, [spawn_points[0]])

    return LayoutData(
        seed=seed,
        width=width,
        height=height,
        grid=grid,
        stations=station_list,
        shelves=shelves,
        spawn_points=spawn_points,
    )


def save_layout(data: LayoutData, output_dir: Path) -> None:
    """Persist layout artefacts to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "layout.npy", data.grid)

    stations_json = [
        {"station_id": s.station_id, "dock": s.dock, "cell": s.cell}
        for s in data.stations
    ]
    with open(output_dir / "stations.json", "w", encoding="utf-8") as f:
        json.dump(stations_json, f, indent=2, ensure_ascii=False)

    shelves_json = [
        {"shelf_id": sid, "home": home} for sid, home in data.shelves.items()
    ]
    with open(output_dir / "shelves.json", "w", encoding="utf-8") as f:
        json.dump(shelves_json, f, indent=2, ensure_ascii=False)

    with open(output_dir / "spawn.json", "w", encoding="utf-8") as f:
        json.dump(data.spawn_points, f, indent=2, ensure_ascii=False)
