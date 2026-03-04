"""Pack/unpack simulation data for GPU buffers.

All constants here must match the WGSL shader (sim.wgsl).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from core.grid import choose_adjacent_target, walkable_mask
from core.layout_generator import LayoutData, generate_layout
from core.order_generator import generate_orders


@lru_cache(maxsize=512)
def _cached_layout(seed: int, width: int, height: int, stations: int):
    """Cache layout + walkable + pickup_cache keyed by (seed, w, h, stations)."""
    layout = generate_layout(seed=seed, width=width, height=height, stations=stations)
    wk = walkable_mask(layout.grid)
    pickup_cache: dict[int, tuple[int, int] | None] = {}
    for sid, home in layout.shelves.items():
        pickup_cache[sid] = choose_adjacent_target(layout.grid, home, wk)
    return layout, wk, pickup_cache

# Must match sim.wgsl constants
MAX_GRID = 90_000
MAX_ROBOTS = 50
MAX_ORDERS = 600
MAX_SHELVES = 50_000
MAX_STATIONS = 30
MAX_ROUTE = 2048
MAX_HEAP = 32_768
GRID_WORDS = MAX_GRID // 4  # 22500
WALK_WORDS = (MAX_GRID + 31) // 32  # 2813

# Struct sizes (bytes)
SIM_PARAMS_SIZE = 48   # 12 × u32 (8 original + stride_grid, stride_shelves, stride_heap, _pad)
ROBOT_SIZE = 64        # 16 × u32
ORDER_SIZE = 24        # 6 × u32
METRICS_SIZE = 32      # 8 × u32


@dataclass
class IndividualData:
    """Pre-processed data for one individual ready for GPU packing."""
    robots: int
    width: int
    height: int
    stations: int
    ticks: int
    strategy_window: int
    layout: LayoutData
    orders: list[dict]
    walkable: np.ndarray
    pickup_cache: dict[int, tuple[int, int] | None]


def prepare_individual(
    robots: int,
    width: int,
    height: int,
    stations: int,
    seed: int,
    orders_count: int,
    orders_burst: bool,
    ticks: int,
    strategy_window: int,
) -> IndividualData | None:
    """Generate layout + orders for one individual. Returns None on failure."""
    if stations * 2 >= width - 2:
        return None

    try:
        layout, wk, pickup_cache = _cached_layout(seed, width, height, stations)
    except Exception:
        return None

    station_ids = [s.station_id for s in layout.stations]
    shelf_ids = list(layout.shelves.keys())
    if not shelf_ids or not station_ids:
        return None

    raw_orders = generate_orders(
        seed=seed, count=orders_count, burst=orders_burst,
        station_ids=station_ids, shelf_ids=shelf_ids,
    )
    # Sort by tick_created (GPU release_orders expects sorted order)
    raw_orders.sort(key=lambda o: o["tick_created"])

    if len(layout.spawn_points) < robots:
        return None

    return IndividualData(
        robots=robots, width=width, height=height, stations=stations,
        ticks=ticks, strategy_window=strategy_window,
        layout=layout, orders=raw_orders, walkable=wk,
        pickup_cache=pickup_cache,
    )


@dataclass
class BatchStrides:
    """Dynamic strides computed from the actual batch data."""
    stride_grid: int    # max(W*H) across batch
    stride_shelves: int # max(n_shelves) across batch
    stride_heap: int    # min(stride_grid*2, MAX_HEAP)
    grid_words: int     # ceil(stride_grid / 4)
    walk_words: int     # ceil(stride_grid / 32)


def compute_strides(individuals: list[IndividualData | None]) -> BatchStrides:
    """Compute tight strides from actual batch dimensions."""
    max_cells = 1
    max_shelves = 1
    for ind in individuals:
        if ind is None:
            continue
        cells = ind.width * ind.height
        if cells > max_cells:
            max_cells = cells
        ns = len(ind.layout.shelves)
        if ns > max_shelves:
            max_shelves = ns
    stride_grid = min(max_cells, MAX_GRID)
    stride_shelves = min(max_shelves, MAX_SHELVES)
    stride_heap = min(stride_grid * 2, MAX_HEAP)
    grid_words = (stride_grid + 3) // 4
    walk_words = (stride_grid + 31) // 32
    return BatchStrides(stride_grid, stride_shelves, stride_heap, grid_words, walk_words)


def pack_generation(individuals: list[IndividualData | None]) -> tuple[dict[str, bytearray], BatchStrides]:
    """Pack all individuals into flat byte arrays for GPU upload.

    Returns (dict of buffer_name -> bytearray, strides).
    ``None`` entries get penalty params (0 ticks -> shader writes penalty metrics).
    """
    n = len(individuals)
    strides = compute_strides(individuals)

    # Allocate byte arrays with dynamic sizes
    params_buf = bytearray(n * SIM_PARAMS_SIZE)
    grids_buf = bytearray(n * strides.grid_words * 4)
    walk_buf = bytearray(n * strides.walk_words * 4)
    robots_buf = bytearray(n * MAX_ROBOTS * ROBOT_SIZE)
    orders_buf = bytearray(n * MAX_ORDERS * ORDER_SIZE)
    shelves_buf = bytearray(n * strides.stride_shelves * 2 * 4)
    pickup_buf = bytearray(n * strides.stride_shelves * 2 * 4)
    stations_buf = bytearray(n * MAX_STATIONS * 2 * 4)

    for idx, ind in enumerate(individuals):
        if ind is None:
            # Penalty: set ticks=0 so shader writes default penalty metrics
            # 12 u32s: W=1, H=1, n_robots=0..., stride_grid, stride_shelves, stride_heap, _pad
            struct.pack_into("<12I", params_buf, idx * SIM_PARAMS_SIZE,
                             1, 1, 0, 0, 0, 0, 0, 50,
                             strides.stride_grid, strides.stride_shelves,
                             strides.stride_heap, 0)
            continue

        _pack_individual(idx, ind, strides, params_buf, grids_buf, walk_buf,
                         robots_buf, orders_buf, shelves_buf, pickup_buf,
                         stations_buf)

    return {
        "params": params_buf,
        "grids": grids_buf,
        "walkable": walk_buf,
        "robots": robots_buf,
        "orders": orders_buf,
        "shelves": shelves_buf,
        "pickup": pickup_buf,
        "stations": stations_buf,
    }, strides


def _pack_individual(
    idx: int,
    ind: IndividualData,
    strides: BatchStrides,
    params_buf: bytearray,
    grids_buf: bytearray,
    walk_buf: bytearray,
    robots_buf: bytearray,
    orders_buf: bytearray,
    shelves_buf: bytearray,
    pickup_buf: bytearray,
    stations_buf: bytearray,
) -> None:
    layout = ind.layout
    W, H = ind.width, ind.height
    n_robots = ind.robots
    n_orders = len(ind.orders)
    n_stations = len(layout.stations)
    n_shelves = len(layout.shelves)

    gw = strides.grid_words
    ww = strides.walk_words

    # SimParams (12 × u32)
    struct.pack_into("<12I", params_buf, idx * SIM_PARAMS_SIZE,
                     W, H, n_robots, n_orders, n_stations, n_shelves,
                     ind.ticks, ind.strategy_window,
                     strides.stride_grid, strides.stride_shelves,
                     strides.stride_heap, 0)

    # Grid (pack 4 cells per u32, each cell 8 bits)
    grid_flat = layout.grid.flatten().astype(np.uint8)
    grid_off = idx * gw * 4
    n_cells = W * H
    padded = np.zeros(gw * 4, dtype=np.uint8)
    padded[:n_cells] = grid_flat[:n_cells]
    packed = (padded[0::4].astype(np.uint32)
              | (padded[1::4].astype(np.uint32) << 8)
              | (padded[2::4].astype(np.uint32) << 16)
              | (padded[3::4].astype(np.uint32) << 24))
    grids_buf[grid_off:grid_off + gw * 4] = packed.tobytes()

    # Walkable (pack 32 cells per u32 as bits)
    walk_flat = ind.walkable.flatten().astype(np.uint8)
    walk_off = idx * ww * 4
    padded_w = np.zeros(ww * 32, dtype=np.uint8)
    padded_w[:n_cells] = walk_flat[:n_cells]
    walk_words_arr = np.zeros(ww, dtype=np.uint32)
    for bit in range(32):
        walk_words_arr |= padded_w[bit::32].astype(np.uint32)[:ww] << bit
    walk_buf[walk_off:walk_off + ww * 4] = walk_words_arr.tobytes()

    # Robots (initialized at spawn points, IDLE state)
    rob_off = idx * MAX_ROBOTS * ROBOT_SIZE
    for ri in range(n_robots):
        sx, sy = layout.spawn_points[ri]
        struct.pack_into("<3Ii4i5IiII", robots_buf, rob_off + ri * ROBOT_SIZE,
                         sx, sy, 0, -1,
                         -1, -1, -1, -1,
                         0, 0, 0, 0, 0,
                         -1, 0, 0)

    # Orders (sorted by tick_created already)
    ord_off = idx * MAX_ORDERS * ORDER_SIZE
    for oi, o in enumerate(ind.orders[:MAX_ORDERS]):
        struct.pack_into("<4Iii", orders_buf, ord_off + oi * ORDER_SIZE,
                         o["order_id"], o["shelf_id"], o["station_id"],
                         o["tick_created"], -1, -1)

    # Shelves (shelf_id -> x, y) — using stride_shelves
    ss = strides.stride_shelves
    sh_off = idx * ss * 2 * 4
    for sid, (sx, sy) in layout.shelves.items():
        if sid < ss:
            struct.pack_into("<2I", shelves_buf, sh_off + sid * 8, sx, sy)

    # Pickup cache (shelf_id -> pickup_x, pickup_y)
    pu_off = idx * ss * 2 * 4
    for sid, cell in ind.pickup_cache.items():
        if sid < ss:
            if cell is not None:
                struct.pack_into("<2I", pickup_buf, pu_off + sid * 8, cell[0], cell[1])
            else:
                struct.pack_into("<2I", pickup_buf, pu_off + sid * 8, 0xFFFFFFFF, 0xFFFFFFFF)

    # Stations (station_id -> dock_x, dock_y)
    st_off = idx * MAX_STATIONS * 2 * 4
    for st in layout.stations[:MAX_STATIONS]:
        dx, dy = st.dock
        struct.pack_into("<2I", stations_buf, st_off + st.station_id * 8, dx, dy)


def scratch_buffer_sizes(pop_size: int, strides: BatchStrides) -> dict[str, int]:
    """Return sizes in bytes for GPU scratch buffers."""
    return {
        "routes": pop_size * MAX_ROBOTS * MAX_ROUTE * 4,
        "cell_rsv": pop_size * strides.stride_grid * 4,
        "astar_g": pop_size * strides.stride_grid * 4,
        "astar_from": pop_size * strides.stride_grid * 4,
        "astar_heap": pop_size * strides.stride_heap * 16,   # vec4<u32> = 16 bytes
        "pending": pop_size * MAX_ORDERS * 4,
        "out_metrics": pop_size * METRICS_SIZE,
    }


_METRICS_DTYPE = np.dtype([
    ("deadlocks", "<u4"), ("completed", "<u4"),
    ("throughput_x1000", "<u4"), ("avg_time_x100", "<u4"),
    ("high_contention", "<u4"), ("total_dist", "<u4"),
    ("_pad0", "<u4"), ("_pad1", "<u4"),
])


def unpack_metrics(raw: bytes | memoryview, n_sims: int) -> list[dict[str, Any]]:
    """Unpack GPU metrics output into Python dicts matching CPU metric keys."""
    arr = np.frombuffer(raw, dtype=_METRICS_DTYPE, count=n_sims)
    results = []
    for row in arr:
        completed = int(row["completed"])
        results.append({
            "deadlocks": int(row["deadlocks"]),
            "completed_orders": completed,
            "throughput_per_1000_ticks": int(row["throughput_x1000"]) / 1000.0,
            "avg_order_time_ticks": int(row["avg_time_x100"]) / 100.0 if completed > 0 else None,
            "high_contention_events": int(row["high_contention"]),
            "total_distance_cells": int(row["total_dist"]),
        })
    return results
