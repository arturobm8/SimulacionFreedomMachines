from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem, Problem

from moop.config import OptConfig
from moop.evaluator import evaluate_individual


class WarehouseOptProblem(ElementwiseProblem):
    """pymoo problem wrapping the warehouse simulation.

    Decision variables are floats (pymoo convention) — we round to ints
    inside ``_evaluate``.  Fixed parameters (seed, orders_count,
    orders_burst, strategy_window) come from SimConfig.
    """

    def __init__(
        self,
        cfg: OptConfig,
        seed: int,
        orders_count: int,
        strategy_window: int,
        **kwargs,
    ) -> None:
        self.cfg = cfg
        self.var_names = cfg.var_names
        self.obj_specs = cfg.objectives
        self.fixed_seed = seed
        self.fixed_orders_count = orders_count
        self.fixed_strategy_window = strategy_window
        self.fixed_area = cfg.fixed_area

        super().__init__(
            n_var=cfg.n_var,
            n_obj=cfg.n_obj,
            xl=np.array(cfg.xl),
            xu=np.array(cfg.xu),
            **kwargs,
        )

    def _decode(self, x: np.ndarray) -> dict[str, int | bool]:
        """Convert pymoo float vector to simulation parameters."""
        params: dict[str, int | bool] = {}
        for i, name in enumerate(self.var_names):
            params[name] = int(round(x[i]))
        # Derive height from fixed area and width
        params["height"] = self.fixed_area // params["width"]
        # Convert orders_burst from int (0/1) to bool
        params["orders_burst"] = bool(params.get("orders_burst", 0))
        # Fixed values (not decision variables)
        params["seed"] = self.fixed_seed
        params["orders_count"] = self.fixed_orders_count
        params["strategy_window"] = self.fixed_strategy_window
        return params

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        params = self._decode(x)

        metrics = evaluate_individual(
            robots=params["robots"],
            width=params["width"],
            height=params["height"],
            stations=params["stations"],
            orders_count=params["orders_count"],
            orders_burst=params["orders_burst"],
            strategy_window=params["strategy_window"],
            seed=params["seed"],
            ticks=self.cfg.eval_ticks,
        )

        objectives = []
        for obj in self.obj_specs:
            raw = metrics.get(obj.metric_key, 0.0)
            if raw is None:
                raw = 99999.0 if obj.direction == "minimize" else 0.0
            val = float(raw)
            # pymoo always minimizes; negate "maximize" objectives
            if obj.direction == "maximize":
                val = -val
            objectives.append(val)

        out["F"] = np.array(objectives)
        # Stash decoded params + raw metrics for TUI reporting
        out["params"] = params
        out["metrics"] = metrics


class GPUWarehouseOptProblem(Problem):
    """pymoo batch Problem using GPU evaluation.

    Evaluates the entire population in one GPU dispatch instead of
    element-wise.  The ``_evaluate`` method receives the full ``X``
    matrix (pop_size × n_var).
    """

    def __init__(
        self,
        cfg: OptConfig,
        seed: int,
        orders_count: int,
        strategy_window: int,
        n_workers: int = 0,
        **kwargs,
    ) -> None:
        self.cfg = cfg
        self.var_names = cfg.var_names
        self.obj_specs = cfg.objectives
        self.fixed_seed = seed
        self.fixed_orders_count = orders_count
        self.fixed_strategy_window = strategy_window
        self.fixed_area = cfg.fixed_area

        from moop.gpu.evaluator import BatchGPUEvaluator
        self.gpu_eval = BatchGPUEvaluator(n_workers=n_workers)

        super().__init__(
            n_var=cfg.n_var,
            n_obj=cfg.n_obj,
            xl=np.array(cfg.xl),
            xu=np.array(cfg.xu),
            **kwargs,
        )

    def _decode(self, x: np.ndarray) -> dict[str, int | bool]:
        params: dict[str, int | bool] = {}
        for i, name in enumerate(self.var_names):
            params[name] = int(round(x[i]))
        # Derive height from fixed area and width
        params["height"] = self.fixed_area // params["width"]
        # Convert orders_burst from int (0/1) to bool
        params["orders_burst"] = bool(params.get("orders_burst", 0))
        return params

    def _evaluate(self, X, out, *args, **kwargs) -> None:
        pop_size = X.shape[0]

        param_sets = [self._decode(X[i]) for i in range(pop_size)]
        all_metrics = self.gpu_eval.evaluate_batch(
            param_sets=param_sets,
            seed=self.fixed_seed,
            orders_count=self.fixed_orders_count,
            ticks=self.cfg.eval_ticks,
            strategy_window=self.fixed_strategy_window,
        )

        # Build F matrix (pop_size × n_obj)
        F = np.zeros((pop_size, len(self.obj_specs)))
        all_params = []
        for i in range(pop_size):
            metrics = all_metrics[i]
            for j, obj in enumerate(self.obj_specs):
                raw = metrics.get(obj.metric_key, 0.0)
                if raw is None:
                    raw = 99999.0 if obj.direction == "minimize" else 0.0
                val = float(raw)
                if obj.direction == "maximize":
                    val = -val
                F[i, j] = val
            all_params.append({**param_sets[i],
                               "seed": self.fixed_seed,
                               "orders_count": self.fixed_orders_count,
                               "strategy_window": self.fixed_strategy_window})

        out["F"] = F
        out["params"] = all_params
        out["metrics"] = all_metrics
