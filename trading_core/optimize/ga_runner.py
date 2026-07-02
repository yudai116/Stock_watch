"""GA optimization over param_space using DEAP, with full trial logging.

Objective (SPEC §8.2): Calmar (CAGR/MaxDD) with a penalty when MaxDD exceeds
the threshold; individuals with DSR<=0 are rejected at the selection level
by the caller after validation. Fitness here is IN-SAMPLE (train folds only)
— IS numbers are never reported as results.
"""

from __future__ import annotations

import random
import uuid
from typing import Callable

from data.config_loader import load_config
from optimize import param_space
from optimize.trial_logger import TrialLogger


def objective_value(metrics: dict) -> float:
    cfg = load_config("params")["objective"]
    calmar = float(metrics.get("calmar", 0.0))
    dd = abs(float(metrics.get("max_dd", 0.0))) * 100.0
    threshold = float(cfg["dd_penalty_threshold_pct"])
    penalty = max(0.0, dd - threshold) * 0.1
    return calmar - penalty


def run_ga(
    evaluate: Callable[[dict], dict],
    n_generations: int = 10,
    pop_size: int = 24,
    seed: int = 42,
    run_id: str | None = None,
    logger: TrialLogger | None = None,
) -> dict:
    """``evaluate(params)`` -> metrics dict computed on TRAIN data only.

    Returns {best_params, best_metrics, run_id, n_trials}.
    """
    from deap import base, creator, tools

    rng = random.Random(seed)
    run_id = run_id or f"ga-{uuid.uuid4().hex[:8]}"
    logger = logger or TrialLogger()

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", dict, fitness=creator.FitnessMax)

    def new_ind():
        return creator.Individual(param_space.sample(rng))

    toolbox = base.Toolbox()
    toolbox.register("individual", new_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_ind(ind, generation):
        metrics = evaluate(dict(ind))
        fit = objective_value(metrics)
        logger.log(run_id=run_id, phase="ga", generation=generation,
                   params=dict(ind), metrics={**metrics, "fitness": fit})
        return metrics, fit

    pop = toolbox.population(n=pop_size)
    best_ind, best_fit, best_metrics = None, float("-inf"), None

    for gen in range(n_generations):
        for ind in pop:
            metrics, fit = eval_ind(ind, gen)
            ind.fitness.values = (fit,)
            if fit > best_fit:
                best_ind, best_fit, best_metrics = dict(ind), fit, metrics
        # elitism + tournament selection, crossover + mutation
        pop.sort(key=lambda i: i.fitness.values[0], reverse=True)
        elite = pop[: max(2, pop_size // 8)]
        children = []
        while len(children) < pop_size - len(elite):
            a, b = rng.sample(pop[: pop_size // 2], 2)
            child_params = param_space.mutate(
                param_space.crossover(dict(a), dict(b), rng), rng)
            child = creator.Individual(child_params)
            children.append(child)
        pop = [creator.Individual(dict(e)) for e in elite] + children

    return {"best_params": best_ind, "best_metrics": best_metrics,
            "run_id": run_id, "n_trials": logger.count_trials(run_id)}
