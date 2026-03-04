from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import DataTable, Static

from moop.config import OptConfig


SPARKLINE_CHARS = " ▁▂▃▄▅▆▇"


def _sparkline(values: list[float], width: int = 8) -> str:
    """Render a sparkline string from a list of values."""
    if not values:
        return ""
    # Take last `width` values
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi != lo else 1.0
    return "".join(
        SPARKLINE_CHARS[min(int((v - lo) / span * (len(SPARKLINE_CHARS) - 1)), len(SPARKLINE_CHARS) - 1)]
        for v in vals
    )


class ParetoTable(DataTable):
    """DataTable displaying the current Pareto front."""

    def setup_columns(self, cfg: OptConfig) -> None:
        self.add_column("#", key="rank")
        for obj in cfg.objectives:
            self.add_column(obj.name.replace("_", " ").title(), key=obj.name)
        for var_name in cfg.var_names:
            self.add_column(var_name, key=var_name)

    def update_front(
        self,
        cfg: OptConfig,
        params_list: list[dict],
        metrics_list: list[dict],
    ) -> None:
        self.clear()
        for i, (params, metrics) in enumerate(zip(params_list, metrics_list)):
            row = [str(i + 1)]
            for obj in cfg.objectives:
                val = metrics.get(obj.metric_key)
                if val is None:
                    row.append("—")
                elif isinstance(val, float):
                    row.append(f"{val:.1f}")
                else:
                    row.append(str(val))
            for var_name in cfg.var_names:
                val = params.get(var_name)
                if var_name == "orders_burst":
                    row.append("yes" if val else "no")
                else:
                    row.append(str(val) if val is not None else "—")
            self.add_row(*row, key=str(i))


class StatsBar(Static):
    """Displays generation progress, eval count, elapsed time, and rate."""

    generation = reactive(0)
    total_generations = reactive(0)
    total_evals = reactive(0)
    elapsed_seconds = reactive(0.0)
    pareto_size = reactive(0)

    def render(self) -> str:
        gen = self.generation
        total = self.total_generations
        evals = self.total_evals
        elapsed = self.elapsed_seconds

        # Progress bar
        pct = gen / max(1, total)
        filled = int(pct * 20)
        bar = "\u2588" * filled + "\u2591" * (20 - filled)

        # Elapsed time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs:02d}s"

        # Rate
        rate = evals / elapsed if elapsed > 0 else 0.0

        return (
            f"Gen {gen}/{total}  [{bar}] {pct:.0%}    "
            f"Evals: {evals}  Elapsed: {time_str}  Rate: {rate:.1f} eval/s  "
            f"Pareto: {self.pareto_size} solutions"
        )


class ConvergencePanel(Static):
    """Shows sparkline convergence per objective."""

    def __init__(self, cfg: OptConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.history: dict[str, list[float]] = {
            obj.name: [] for obj in cfg.objectives
        }

    def push_generation(self, metrics_list: list[dict]) -> None:
        """Record the best value for each objective from the current gen."""
        for obj in self.cfg.objectives:
            vals = [
                m.get(obj.metric_key, None)
                for m in metrics_list
                if m.get(obj.metric_key) is not None
            ]
            if vals:
                if obj.direction == "minimize":
                    best = min(vals)
                else:
                    best = max(vals)
                self.history[obj.name].append(float(best))

        self.refresh()

    def render(self) -> str:
        lines = ["Convergence"]
        for obj in self.cfg.objectives:
            spark = _sparkline(self.history.get(obj.name, []))
            label = obj.name.replace("_", " ")
            lines.append(f"  {label:<18} {spark}")
        return "\n".join(lines)


class BestConfigPanel(Static):
    """Shows the best configuration for the selected objective (first by default)."""

    def __init__(self, cfg: OptConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self._params: dict = {}
        self._metrics: dict = {}

    def update_best(self, params: dict, metrics: dict) -> None:
        self._params = params
        self._metrics = metrics
        self.refresh()

    def render(self) -> str:
        if not self._params:
            return "Best config\n  (waiting...)"

        lines = ["Best config"]
        for name in self.cfg.var_names:
            val = self._params.get(name)
            if name == "orders_burst":
                lines.append(f"  {name}: {'yes' if val else 'no'}")
            else:
                lines.append(f"  {name}: {val}")
        return "\n".join(lines)
