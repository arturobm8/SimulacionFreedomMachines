from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from moop.config import OptConfig
from moop.optimizer import OptResult, run_optimization
from moop.tui.widgets import BestConfigPanel, ConvergencePanel, ParetoTable, StatsBar


class OptimizationScreen(Screen):
    """Live monitoring screen during the optimization run."""

    BINDINGS = [Binding("q", "quit", "Quit")]

    CSS = """
    #stats-bar {
        height: 3;
        padding: 0 1;
        background: $surface;
        border-bottom: solid $primary;
    }
    #pareto-table {
        height: 1fr;
        min-height: 8;
    }
    #bottom-panels {
        height: 12;
    }
    #convergence-panel {
        width: 1fr;
        padding: 0 1;
        border: solid $primary;
    }
    #best-config-panel {
        width: 1fr;
        padding: 0 1;
        border: solid $primary;
    }
    """

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
        self._start_time: float = 0.0
        self._result: OptResult | None = None
        # Track Pareto front across generations
        self._best_params: list[dict] = []
        self._best_metrics: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatsBar(id="stats-bar")
        yield ParetoTable(id="pareto-table")
        with Horizontal(id="bottom-panels"):
            yield ConvergencePanel(self.cfg, id="convergence-panel")
            yield BestConfigPanel(self.cfg, id="best-config-panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#pareto-table", ParetoTable)
        table.setup_columns(self.cfg)

        stats = self.query_one("#stats-bar", StatsBar)
        stats.total_generations = self.cfg.generations

        self._start_time = time.monotonic()
        self._run_optimization()

    @work(thread=True)
    def _run_optimization(self) -> None:
        """Run pymoo in a background thread, updating TUI via call_from_thread."""

        def on_gen(
            gen: int,
            X: np.ndarray,
            F: np.ndarray,
            params_list: list[dict],
            metrics_list: list[dict],
        ) -> None:
            self.app.call_from_thread(
                self._update_ui, gen, X, F, params_list, metrics_list
            )

        result = run_optimization(
            self.cfg,
            on_generation=on_gen,
            n_workers=self.n_workers,
            seed=self.seed,
            orders_count=self.orders_count,
        )
        self.app.call_from_thread(self._on_complete, result)

    def _update_ui(
        self,
        gen: int,
        X: np.ndarray,
        F: np.ndarray,
        params_list: list[dict],
        metrics_list: list[dict],
    ) -> None:
        elapsed = time.monotonic() - self._start_time
        evals = gen * self.cfg.population_size

        # Update stats bar
        stats = self.query_one("#stats-bar", StatsBar)
        stats.generation = gen
        stats.total_evals = evals
        stats.elapsed_seconds = elapsed

        # Find non-dominated solutions (Pareto front)
        pareto_params, pareto_metrics = self._extract_pareto(
            params_list, metrics_list
        )
        stats.pareto_size = len(pareto_params)

        # Track best Pareto front
        self._best_params = pareto_params
        self._best_metrics = pareto_metrics

        # Update table
        table = self.query_one("#pareto-table", ParetoTable)
        table.update_front(self.cfg, pareto_params, pareto_metrics)

        # Update convergence
        conv = self.query_one("#convergence-panel", ConvergencePanel)
        conv.push_generation(metrics_list)

        # Update best config (first Pareto solution)
        if pareto_params:
            best_panel = self.query_one("#best-config-panel", BestConfigPanel)
            best_panel.update_best(pareto_params[0], pareto_metrics[0])

    def _extract_pareto(
        self, params_list: list[dict], metrics_list: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Extract non-dominated solutions based on objective values."""
        if not metrics_list:
            return [], []

        # Build objective vectors
        obj_vals = []
        for m in metrics_list:
            row = []
            for obj in self.cfg.objectives:
                val = m.get(obj.metric_key)
                if val is None:
                    val = 99999.0 if obj.direction == "minimize" else 0.0
                v = float(val)
                if obj.direction == "maximize":
                    v = -v
                row.append(v)
            obj_vals.append(row)

        F = np.array(obj_vals)
        # Simple non-dominated sort
        n = len(F)
        is_dominated = [False] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if all(F[j] <= F[i]) and any(F[j] < F[i]):
                    is_dominated[i] = True
                    break

        pareto_p = [params_list[i] for i in range(n) if not is_dominated[i]]
        pareto_m = [metrics_list[i] for i in range(n) if not is_dominated[i]]
        return pareto_p, pareto_m

    def _on_complete(self, result: OptResult) -> None:
        self._result = result

        # Save CSV if configured
        if self.cfg.save_pareto:
            self._save_pareto_csv(result)

        self.app.push_screen(
            ResultsScreen(self.cfg, result, self._best_params, self._best_metrics)
        )

    def _save_pareto_csv(self, result: OptResult) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "pareto_front.csv"

        if not result.params:
            return

        derived_keys = ["height"]
        param_keys = list(self.cfg.var_names) + derived_keys
        extra_metric_keys = ["total_distance_cells", "high_contention_events"]
        fieldnames = (
            ["seed"] + param_keys + ["orders_count"]
            + [obj.metric_key for obj in self.cfg.objectives]
            + ["completion_pct"] + extra_metric_keys
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for params, metrics in zip(result.params, result.metrics):
                row = {}
                row["seed"] = params.get("seed", "")
                for k in param_keys:
                    row[k] = params.get(k, "")
                row["orders_count"] = params.get("orders_count", "")
                for obj in self.cfg.objectives:
                    row[obj.metric_key] = metrics.get(obj.metric_key, "")
                oc = params.get("orders_count", 0)
                completed = metrics.get("completed_orders", 0)
                row["completion_pct"] = round(completed / oc * 100, 1) if oc else ""
                for k in extra_metric_keys:
                    row[k] = metrics.get(k, "")
                writer.writerow(row)

    def action_quit(self) -> None:
        self.app.exit()


class ResultsScreen(Screen):
    """Final screen showing the Pareto front and allowing export/inspection."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "save_csv", "Export CSV"),
    ]

    CSS = """
    #results-header {
        height: 3;
        padding: 0 1;
        background: $surface;
        border-bottom: solid $primary;
    }
    #results-table {
        height: 1fr;
    }
    #detail-panel {
        height: 8;
        padding: 0 1;
        border: solid $primary;
    }
    """

    def __init__(
        self,
        cfg: OptConfig,
        result: OptResult,
        pareto_params: list[dict],
        pareto_metrics: list[dict],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.result = result
        self.pareto_params = pareto_params or result.params
        self.pareto_metrics = pareto_metrics or result.metrics
        self._selected_idx = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"Optimization complete — {len(self.pareto_params)} Pareto solutions  "
            f"({self.result.n_gen} generations, {self.result.n_evals} evaluations)",
            id="results-header",
        )
        yield ParetoTable(id="results-table")
        yield Static("Select a row to see details", id="detail-panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results-table", ParetoTable)
        table.setup_columns(self.cfg)
        table.update_front(self.cfg, self.pareto_params, self.pareto_metrics)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            idx = int(str(event.row_key.value))
        except (ValueError, TypeError):
            return

        if idx < len(self.pareto_params):
            params = self.pareto_params[idx]
            metrics = self.pareto_metrics[idx]

            lines = []
            for k, v in params.items():
                if k == "orders_burst":
                    lines.append(f"{k}: {'yes' if v else 'no'}")
                else:
                    lines.append(f"{k}: {v}")
            lines.append("---")
            for obj in self.cfg.objectives:
                val = metrics.get(obj.metric_key)
                if val is not None and isinstance(val, float):
                    lines.append(f"{obj.name}: {val:.2f}")
                else:
                    lines.append(f"{obj.name}: {val}")

            detail = self.query_one("#detail-panel", Static)
            detail.update("  ".join(lines))

    def action_save_csv(self) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "pareto_front.csv"

        derived_keys = ["height"]
        param_keys = list(self.cfg.var_names) + derived_keys
        extra_metric_keys = ["total_distance_cells", "high_contention_events"]
        fieldnames = (
            ["seed"] + param_keys + ["orders_count"]
            + [obj.metric_key for obj in self.cfg.objectives]
            + ["completion_pct"] + extra_metric_keys
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for params, metrics in zip(self.pareto_params, self.pareto_metrics):
                row = {}
                row["seed"] = params.get("seed", "")
                for k in param_keys:
                    row[k] = params.get(k, "")
                row["orders_count"] = params.get("orders_count", "")
                for obj in self.cfg.objectives:
                    row[obj.metric_key] = metrics.get(obj.metric_key, "")
                oc = params.get("orders_count", 0)
                completed = metrics.get("completed_orders", 0)
                row["completion_pct"] = round(completed / oc * 100, 1) if oc else ""
                for k in extra_metric_keys:
                    row[k] = metrics.get(k, "")
                writer.writerow(row)

        self.notify(f"Saved to {csv_path}")

    def action_quit(self) -> None:
        self.app.exit()
