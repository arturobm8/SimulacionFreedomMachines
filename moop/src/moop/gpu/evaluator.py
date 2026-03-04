"""GPU batch evaluator: runs an entire MOOP population in one dispatch."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import wgpu

from moop.gpu.buffers import (
    METRICS_SIZE,
    IndividualData,
    pack_generation,
    prepare_individual,
    scratch_buffer_sizes,
    unpack_metrics,
)

_SHADER_PATH = Path(__file__).parent / "shaders" / "sim.wgsl"


def _prepare_one(args: tuple) -> IndividualData | None:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    return prepare_individual(*args)


# Penalty metrics returned when an individual is infeasible
_PENALTY: dict[str, Any] = {
    "deadlocks": 9999,
    "throughput_per_1000_ticks": 0.0,
    "completed_orders": 0,
    "avg_order_time_ticks": 99999.0,
    "high_contention_events": 0,
    "total_distance_cells": 0,
}


class BatchGPUEvaluator:
    """Evaluates an entire population on GPU in one dispatch."""

    def __init__(self, n_workers: int = 0) -> None:
        self.n_workers = n_workers or os.cpu_count() or 4
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No WebGPU adapter found")

        # Request device with maximum buffer size
        self.device = adapter.request_device_sync(
            required_limits={
                "max_storage_buffer_binding_size": 1 << 30,  # 1 GB
                "max_buffer_size": 1 << 30,
            }
        )

        shader_source = _SHADER_PATH.read_text(encoding="utf-8")
        self.shader = self.device.create_shader_module(code=shader_source)
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.shader, "entry_point": "main"},
        )

        # Reusable scratch buffers (created on first use, resized if needed)
        self._scratch: dict[str, wgpu.GPUBuffer] = {}
        self._scratch_sizes: dict[str, int] = {}

    def _ensure_scratch(self, needed: dict[str, int]) -> dict[str, wgpu.GPUBuffer]:
        """Create or reuse scratch buffers. Only re-allocate if size grew."""
        STORAGE_RW = wgpu.BufferUsage.STORAGE
        COPY_DST = wgpu.BufferUsage.COPY_DST
        COPY_SRC = wgpu.BufferUsage.COPY_SRC

        for name, size in needed.items():
            size = max(size, 4)
            if name in self._scratch and self._scratch_sizes[name] >= size:
                continue
            # Destroy old buffer if exists
            if name in self._scratch:
                self._scratch[name].destroy()
            usage = STORAGE_RW | COPY_DST
            if name == "out_metrics":
                usage |= COPY_SRC
            self._scratch[name] = self.device.create_buffer(size=size, usage=usage)
            self._scratch_sizes[name] = size
        return self._scratch

    def evaluate_batch(
        self,
        param_sets: list[dict[str, Any]],
        seed: int,
        orders_count: int,
        ticks: int,
        strategy_window: int,
    ) -> list[dict[str, Any]]:
        """Evaluate all individuals on GPU.

        Parameters
        ----------
        param_sets : list of dicts with keys robots, width, height, stations, orders_burst
        seed, orders_count, ticks, strategy_window : fixed params

        Returns
        -------
        List of metrics dicts (same keys as CPU evaluator).
        """
        pop_size = len(param_sets)

        # 1. CPU: prepare all individuals (parallel)
        args_list = [
            (ps["robots"], ps["width"], ps["height"], ps["stations"],
             seed, orders_count, ps["orders_burst"], ticks, strategy_window)
            for ps in param_sets
        ]

        if self.n_workers > 1 and pop_size > 1:
            with ProcessPoolExecutor(max_workers=min(self.n_workers, pop_size)) as pool:
                individuals: list[IndividualData | None] = list(pool.map(_prepare_one, args_list))
        else:
            individuals = [prepare_individual(*a) for a in args_list]

        penalty_indices: set[int] = {i for i, ind in enumerate(individuals) if ind is None}

        # If ALL are penalties, skip GPU entirely
        if len(penalty_indices) == pop_size:
            return [_PENALTY.copy() for _ in range(pop_size)]

        # 2. Pack into GPU buffers
        packed, strides = pack_generation(individuals)
        scratch = scratch_buffer_sizes(pop_size, strides)

        # 3. Create GPU data buffers and upload; reuse scratch buffers
        STORAGE = wgpu.BufferUsage.STORAGE
        COPY_DST = wgpu.BufferUsage.COPY_DST

        def make_buf(data: bytes | bytearray, rw: bool = False) -> wgpu.GPUBuffer:
            usage = STORAGE | COPY_DST
            buf = self.device.create_buffer(size=len(data), usage=usage)
            self.device.queue.write_buffer(buf, 0, data)
            return buf

        # Group 0 buffers (freshly created per batch — data changes each gen)
        buf_params = make_buf(packed["params"])
        buf_grids = make_buf(packed["grids"])
        buf_walk = make_buf(packed["walkable"])
        buf_robots = make_buf(packed["robots"], rw=True)
        buf_orders = make_buf(packed["orders"], rw=True)
        buf_shelves = make_buf(packed["shelves"])
        buf_pickup = make_buf(packed["pickup"])
        buf_stations = make_buf(packed["stations"])

        # Group 1 buffers (scratch — reused across generations)
        sb = self._ensure_scratch(scratch)
        buf_routes = sb["routes"]
        buf_rsv = sb["cell_rsv"]
        buf_ag = sb["astar_g"]
        buf_af = sb["astar_from"]
        buf_ah = sb["astar_heap"]
        buf_pending = sb["pending"]
        buf_metrics = sb["out_metrics"]

        # 4. Create bind groups
        bg0 = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_params}},
                {"binding": 1, "resource": {"buffer": buf_grids}},
                {"binding": 2, "resource": {"buffer": buf_walk}},
                {"binding": 3, "resource": {"buffer": buf_robots}},
                {"binding": 4, "resource": {"buffer": buf_orders}},
                {"binding": 5, "resource": {"buffer": buf_shelves}},
                {"binding": 6, "resource": {"buffer": buf_pickup}},
                {"binding": 7, "resource": {"buffer": buf_stations}},
            ],
        )
        bg1 = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(1),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_routes}},
                {"binding": 1, "resource": {"buffer": buf_rsv}},
                {"binding": 2, "resource": {"buffer": buf_ag}},
                {"binding": 3, "resource": {"buffer": buf_af}},
                {"binding": 4, "resource": {"buffer": buf_ah}},
                {"binding": 5, "resource": {"buffer": buf_pending}},
                {"binding": 6, "resource": {"buffer": buf_metrics}},
            ],
        )

        # 5. Dispatch
        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.set_bind_group(0, bg0)
        pass_enc.set_bind_group(1, bg1)
        pass_enc.dispatch_workgroups(pop_size, 1, 1)
        pass_enc.end()
        self.device.queue.submit([encoder.finish()])

        # 6. Read back metrics
        raw_metrics = self.device.queue.read_buffer(buf_metrics)
        all_metrics = unpack_metrics(raw_metrics, pop_size)

        # 7. Patch penalty individuals
        for i in penalty_indices:
            all_metrics[i] = _PENALTY.copy()

        # Also patch individuals where ticks=0 (they ran zero-tick sims)
        for i, ind in enumerate(individuals):
            if ind is None:
                all_metrics[i] = _PENALTY.copy()

        # Destroy data buffers (scratch buffers are reused)
        for buf in (buf_params, buf_grids, buf_walk, buf_robots, buf_orders,
                     buf_shelves, buf_pickup, buf_stations):
            buf.destroy()

        return all_metrics
