from __future__ import annotations

from textual.app import App

from moop.config import OptConfig
from moop.tui.screens import OptimizationScreen


class MoopApp(App):
    """Textual application for live optimization monitoring."""

    TITLE = "MOOP Warehouse Optimizer"

    def __init__(
        self,
        cfg: OptConfig,
        n_workers: int | None = None,
        seed: int = 42,
        orders_count: int = 600,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.n_workers = n_workers
        self.seed = seed
        self.orders_count = orders_count

    def on_mount(self) -> None:
        self.push_screen(
            OptimizationScreen(
                self.cfg,
                n_workers=self.n_workers,
                seed=self.seed,
                orders_count=self.orders_count,
            )
        )
