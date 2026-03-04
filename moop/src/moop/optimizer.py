from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.indicators.hv import HV

from moop.config import OptConfig
from moop.problem import GPUWarehouseOptProblem


OnGeneration = Callable[[int, np.ndarray, np.ndarray, list[dict], list[dict]], None]


class ProgressCallback(Callback):
    """Fires ``on_generation`` after each pymoo generation."""

    def __init__(self, on_generation: OnGeneration | None = None) -> None:
        super().__init__()
        self.on_generation = on_generation

    def notify(self, algorithm) -> None:
        if self.on_generation is None:
            return

        gen = algorithm.n_gen
        pop = algorithm.pop

        X = pop.get("X")
        F = pop.get("F")

        # Collect per-individual params/metrics from problem output
        params_list: list[dict] = []
        metrics_list: list[dict] = []
        for ind in pop:
            params_list.append(ind.get("params") or {})
            metrics_list.append(ind.get("metrics") or {})

        self.on_generation(gen, X, F, params_list, metrics_list)


@dataclass
class OptResult:
    """Holds the final optimization output."""

    X: np.ndarray
    F: np.ndarray
    params: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    n_gen: int = 0
    n_evals: int = 0


def run_optimization(
    cfg: OptConfig,
    on_generation: OnGeneration | None = None,
    n_workers: int | None = None,
    seed: int = 42,
    orders_count: int = 600,
) -> OptResult:
    """Run the multi-objective optimization and return results."""
    workers = n_workers if n_workers is not None else cfg.n_workers
    if workers <= 0:
        workers = os.cpu_count() or 4

    problem = GPUWarehouseOptProblem(
        cfg,
        seed=seed,
        orders_count=orders_count,
        strategy_window=cfg.strategy_window,
        n_workers=workers,
    )

    algorithm = NSGA2(
        pop_size=cfg.population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    callback = ProgressCallback(on_generation=on_generation)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", cfg.generations),
        callback=callback,
        seed=seed,
        verbose=False,
    )

    # Extract Pareto-front individuals
    pareto_X = res.X if res.X is not None else np.empty((0, cfg.n_var))
    pareto_F = res.F if res.F is not None else np.empty((0, cfg.n_obj))

    # Decode params/metrics for Pareto solutions
    pareto_params: list[dict] = []
    pareto_metrics: list[dict] = []
    if res.pop is not None:
        opt_pop = res.opt
        for ind in opt_pop:
            pareto_params.append(ind.get("params") or {})
            pareto_metrics.append(ind.get("metrics") or {})

    # Sort by hypervolume contribution (highest first)
    order = _hv_contribution_order(pareto_F)
    pareto_X = pareto_X[order]
    pareto_F = pareto_F[order]
    pareto_params = [pareto_params[i] for i in order]
    pareto_metrics = [pareto_metrics[i] for i in order]

    return OptResult(
        X=pareto_X,
        F=pareto_F,
        params=pareto_params,
        metrics=pareto_metrics,
        n_gen=res.algorithm.n_gen if res.algorithm else 0,
        n_evals=res.algorithm.evaluator.n_eval if res.algorithm else 0,
    )


def _hv_contribution_order(F: np.ndarray) -> list[int]:
    """Return indices sorted by descending hypervolume contribution."""
    n = len(F)
    if n == 0:
        return []

    # Reference point: worst value per objective + 10% margin
    ref_point = F.max(axis=0) * 1.1
    # Handle zeros/negatives in ref_point
    ref_point = np.where(ref_point > 0, ref_point, F.max(axis=0) + 1.0)

    hv = HV(ref_point=ref_point)
    total_hv = hv(F)

    contributions = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        hv_without_i = hv(F[mask]) if mask.sum() > 0 else 0.0
        contributions[i] = total_hv - hv_without_i

    return list(np.argsort(-contributions))
