from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from config import SimConfig
from core import (
    BLOCKED,
    FREE,
    SHELF,
    STATION,
    WarehouseSimulator,
)
from core.types import RobotState

# State colour mapping
_STATE_VAL = {
    RobotState.IDLE: 0.0,
    RobotState.TO_PICKUP: 1.0,
    RobotState.TO_STATION: 2.0,
    RobotState.RETURNING: 3.0,
}
_COLORS = ["#777777", "#ff7f0e", "#1f77b4", "#2ca02c"]
_CMAP = mpl.colors.ListedColormap(_COLORS, name="robot_states")
_NORM = mpl.colors.Normalize(vmin=0.0, vmax=3.0)


def _grid_image(grid: np.ndarray) -> np.ndarray:
    img = np.zeros_like(grid, dtype=float)
    img[grid == FREE] = 1.0
    img[grid == STATION] = 0.7
    img[grid == SHELF] = 0.35
    img[grid == BLOCKED] = 0.0
    return img


def render_layout(grid: np.ndarray, output_path: str | Path) -> None:
    """Save a static grayscale layout PNG."""
    img = _grid_image(grid)
    plt.figure(figsize=(10, 7))
    plt.title("CEDIS Layout")
    plt.imshow(img, origin="upper", interpolation="nearest")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def save_heatmaps(
    grid: np.ndarray,
    visits: np.ndarray,
    waits: np.ndarray,
    prefix: str | Path,
) -> None:
    """Write traffic and congestion heatmaps."""
    prefix = str(prefix)
    walkable = (grid == FREE) | (grid == STATION)

    # Visits heatmap
    v = np.where(walkable, visits, 0)
    plt.figure(figsize=(10, 7))
    plt.title("Heatmap: Cell visits (traffic intensity)")
    plt.imshow(v, origin="upper", interpolation="nearest")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{prefix}_visits.png", dpi=200)
    plt.close()

    # Waits heatmap
    e = np.where(walkable, waits, 0)
    plt.figure(figsize=(10, 7))
    plt.title("Heatmap: Wait events (congestion)")
    plt.imshow(e, origin="upper", interpolation="nearest")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{prefix}_waits.png", dpi=200)
    plt.close()

    # Ratio heatmap
    ratio = np.zeros_like(e, dtype=float)
    mask = v > 0
    ratio[mask] = e[mask] / v[mask]
    plt.figure(figsize=(10, 7))
    plt.title("Heatmap: Wait/Visit ratio (relative congestion)")
    plt.imshow(np.where(walkable, ratio, 0.0), origin="upper", interpolation="nearest")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{prefix}_ratio.png", dpi=200)
    plt.close()


def animate(
    grid: np.ndarray,
    sim: WarehouseSimulator,
    ticks: int,
    steps_per_frame: int,
    output_path: str | Path,
    fps: int,
    heatmap_prefix: str | Path,
) -> None:
    """Run the simulation while recording an animation + heatmaps."""
    h, w = grid.shape
    visits = np.zeros((h, w), dtype=np.int32)
    waits = np.zeros((h, w), dtype=np.int32)

    img = _grid_image(grid)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.set_title("Robot fleet simulation (CEDIS)")
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    positions = sim.robot_positions()
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    states = sim.robot_states()
    cvals = [_STATE_VAL.get(s, 0.0) for s in states]

    scat = ax.scatter(xs, ys, s=30, c=cvals, cmap=_CMAP, norm=_NORM)

    cbar = plt.colorbar(scat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["IDLE", "TO_PICKUP", "TO_STATION", "RETURNING"])

    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    total_frames = max(1, ticks // steps_per_frame)
    completed_count = 0
    completed_ids: set[int] = set()

    def update(frame_idx: int):
        nonlocal visits, waits, completed_count, completed_ids

        for _ in range(steps_per_frame):
            prev_pos = sim.robot_positions()
            sim.step()
            cur_pos = sim.robot_positions()

            for x, y in cur_pos:
                if 0 <= x < w and 0 <= y < h:
                    visits[y, x] += 1

            for p0, p1 in zip(prev_pos, cur_pos):
                if p0 == p1:
                    x, y = p1
                    if 0 <= x < w and 0 <= y < h:
                        waits[y, x] += 1

            for o in sim.orders:
                if o.tick_completed is not None and o.order_id not in completed_ids:
                    completed_ids.add(o.order_id)
                    completed_count += 1

        positions = sim.robot_positions()
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        scat.set_offsets(np.c_[xs, ys])

        states = sim.robot_states()
        cvals = [_STATE_VAL.get(s, 0.0) for s in states]
        scat.set_array(np.array(cvals))

        tick = sim.tick
        total = len(sim.orders)
        throughput = completed_count / (tick / 1000.0) if tick > 0 else 0.0

        text.set_text(
            f"tick={tick} | "
            f"completed={completed_count}/{total} | "
            f"throughput={throughput:.1f}/1000t | "
            f"contention={sim.high_contention_events} | "
            f"deadlock={sim.deadlock_count}"
        )
        return scat, text

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 / fps, blit=False, repeat=False,
    )

    output_str = str(output_path)
    Path(output_str).parent.mkdir(parents=True, exist_ok=True)

    if output_str.lower().endswith(".gif"):
        anim.save(output_str, writer=animation.PillowWriter(fps=fps))
    else:
        writer_cls = animation.writers["ffmpeg"]
        writer = writer_cls(fps=fps, metadata={"artist": "freedom_machines"}, bitrate=1800)
        anim.save(output_str, writer=writer)

    plt.close(fig)
    save_heatmaps(grid, visits, waits, prefix=str(heatmap_prefix))


def run_visualize(cfg: SimConfig, sim: WarehouseSimulator, grid: np.ndarray) -> None:
    """Full visualization pipeline: layout + animation + heatmaps.

    Strategy-agnostic — the caller passes a pre-built simulator.
    """
    if cfg.paths.ffmpeg and cfg.paths.ffmpeg != "ffmpeg":
        mpl.rcParams["animation.ffmpeg_path"] = cfg.paths.ffmpeg

    render_layout(grid, cfg.scenario_path("layout.png"))

    animate(
        grid=grid,
        sim=sim,
        ticks=cfg.ticks,
        steps_per_frame=cfg.visualization.steps_per_frame,
        output_path=cfg.scenario_path("simulation.mp4"),
        fps=cfg.visualization.fps,
        heatmap_prefix=cfg.scenario_path("heatmap"),
    )
