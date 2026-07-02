"""GA parameter space, parsed from config/params.yaml `ga_search`.

Only the parameters listed there are searchable (SPEC: gates and structural
constants are fixed). Individuals are plain dicts param -> value.
"""

from __future__ import annotations

import random
from typing import Any

from data.config_loader import load_config


def space() -> dict[str, dict]:
    return load_config("params")["ga_search"]


def sample(rng: random.Random) -> dict[str, Any]:
    ind = {}
    for name, spec in space().items():
        ind[name] = _sample_one(spec, rng)
    return ind


def _sample_one(spec: dict, rng: random.Random) -> Any:
    t = spec["type"]
    if t == "int":
        return rng.randint(int(spec["low"]), int(spec["high"]))
    if t == "float":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    if t == "choice":
        return rng.choice(spec["options"])
    raise ValueError(f"unknown param type {t}")


def mutate(ind: dict[str, Any], rng: random.Random, p: float = 0.25) -> dict[str, Any]:
    out = dict(ind)
    for name, spec in space().items():
        if rng.random() >= p:
            continue
        t = spec["type"]
        if t == "choice":
            out[name] = rng.choice(spec["options"])
        else:
            lo, hi = float(spec["low"]), float(spec["high"])
            span = (hi - lo) * 0.2
            val = float(out[name]) + rng.gauss(0, span)
            val = min(hi, max(lo, val))
            out[name] = int(round(val)) if t == "int" else val
    return out


def crossover(a: dict[str, Any], b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def merged_params(individual: dict[str, Any]) -> dict[str, Any]:
    """Individual + fixed params, ready for SwingStrategy."""
    fixed = load_config("params")["fixed"]
    return {**fixed, **individual}
