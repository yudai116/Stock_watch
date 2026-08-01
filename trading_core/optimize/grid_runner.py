"""Coarse grid + plateau selection — the PRIMARY optimizer (SPEC_ADDENDUM_v2 R3).

Protocol:
  1. Enumerate the full grid from config/params.yaml ``grid_search.<strategy>``
     (3-5 levels per parameter, HARD LIMIT 6 parameters per strategy).
  2. Evaluate every combination on TRAIN data only; log every trial to
     trial_logger (DSR denominator).
  3. Plateau selection: take the top decile by objective, then within it
     pick the point whose GRID NEIGHBOURS (one step in one dimension) have
     the LOWEST objective variance — a sharp isolated peak is rejected in
     favour of a stable plateau.

GA (optimize/ga_runner.py) remains available for experiments but is no
longer the primary path.
"""

from __future__ import annotations

import itertools
import math
import uuid
from typing import Any, Callable

import numpy as np

from data.config_loader import load_config
from optimize.ga_runner import objective_value
from optimize.trial_logger import TrialLogger

MAX_PARAMS = 6      # R3 hard limit


def grid_for(strategy: str) -> dict[str, list]:
    grid = load_config("params")["grid_search"][strategy]
    if len(grid) > MAX_PARAMS:
        raise ValueError(
            f"{strategy}: {len(grid)} parameters exceed the R3 limit of {MAX_PARAMS}")
    return grid


def enumerate_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    names = list(grid)
    return [dict(zip(names, combo))
            for combo in itertools.product(*(grid[n] for n in names))]


def _neighbours(point: dict, grid: dict[str, list]) -> list[dict]:
    """Points one level-step away in exactly one dimension."""
    out = []
    for name, levels in grid.items():
        i = levels.index(point[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(levels):
                nb = dict(point)
                nb[name] = levels[j]
                out.append(nb)
    return out


def plateau_select(results: list[tuple[dict, float]],
                   grid: dict[str, list],
                   top_fraction: float = 0.10) -> tuple[dict, float, dict]:
    """``results``: [(params, objective)]. Returns (params, objective, info).

    Among the top ``top_fraction`` by objective, pick the candidate whose
    neighbourhood (itself + grid neighbours) has minimal objective variance;
    ties break toward the higher objective.
    """
    if not results:
        raise ValueError("no results to select from")
    by_key = {tuple(sorted(p.items())): obj for p, obj in results}
    ranked = sorted(results, key=lambda r: r[1], reverse=True)
    n_top = max(1, math.ceil(len(ranked) * top_fraction))
    candidates = ranked[:n_top]

    best = None
    for params, obj in candidates:
        neigh = _neighbours(params, grid)
        objs = [obj] + [by_key[tuple(sorted(nb.items()))]
                        for nb in neigh if tuple(sorted(nb.items())) in by_key]
        var = float(np.var(objs)) if len(objs) > 1 else float("inf")
        key = (var, -obj)
        if best is None or key < best[0]:
            best = (key, params, obj, {"neighbourhood_var": var,
                                       "neighbourhood_size": len(objs)})
    _, params, obj, info = best
    return params, obj, info


def run_grid(
    evaluate: Callable[[dict], dict],
    strategy: str,
    run_id: str | None = None,
    logger: TrialLogger | None = None,
) -> dict:
    """``evaluate(params)`` -> metrics dict computed on TRAIN data only.

    Returns {best_params, best_objective, best_metrics, n_trials, run_id,
    selection_info}. IS numbers are never reported as results.
    """
    grid = grid_for(strategy)
    run_id = run_id or f"grid-{strategy}-{uuid.uuid4().hex[:8]}"
    logger = logger or TrialLogger()

    results: list[tuple[dict, float]] = []
    metrics_by_key: dict[tuple, dict] = {}
    for params in enumerate_grid(grid):
        metrics = evaluate(dict(params))
        obj = objective_value(metrics)
        logger.log(run_id=run_id, phase=f"grid_{strategy}", params=params,
                   metrics={**metrics, "objective": obj})
        results.append((params, obj))
        metrics_by_key[tuple(sorted(params.items()))] = metrics

    best_params, best_obj, info = plateau_select(results, grid)
    return {
        "best_params": best_params,
        "best_objective": best_obj,
        "best_metrics": metrics_by_key[tuple(sorted(best_params.items()))],
        "n_trials": logger.count_trials(run_id),
        "run_id": run_id,
        "selection_info": info,
    }
