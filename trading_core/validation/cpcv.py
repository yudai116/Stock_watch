"""Combinatorial Purged Cross-Validation (López de Prado).

Splits the timeline into G contiguous groups; every combination of k groups
forms a test set. Train observations within ``purge`` days of any test
boundary are dropped; ``embargo`` days after each test block are also
excluded from training (leakage via serial correlation / open positions).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from data.config_loader import load_config


def cpcv_splits(index: pd.DatetimeIndex,
                n_groups: int | None = None,
                n_test_groups: int | None = None,
                purge_days: int | None = None,
                embargo_days: int | None = None):
    """Yield (train_mask, test_mask) boolean arrays aligned to ``index``."""
    cfg = load_config("params")["validation"]
    G = n_groups or int(cfg["cpcv_groups"])
    k = n_test_groups or int(cfg["cpcv_test_groups"])
    purge = pd.Timedelta(days=purge_days if purge_days is not None else cfg["purge_days"])
    embargo = pd.Timedelta(days=embargo_days if embargo_days is not None else cfg["embargo_days"])

    idx = pd.DatetimeIndex(index).sort_values()
    n = len(idx)
    bounds = np.linspace(0, n, G + 1, dtype=int)
    groups = [(idx[bounds[g]], idx[bounds[g + 1] - 1]) for g in range(G)]

    for combo in combinations(range(G), k):
        test_mask = np.zeros(n, dtype=bool)
        train_mask = np.ones(n, dtype=bool)
        for g in combo:
            s, e = groups[g]
            in_g = (idx >= s) & (idx <= e)
            test_mask |= in_g
            # purge train data around the test block, embargo after it
            kill = (idx >= s - purge) & (idx <= e + purge + embargo)
            train_mask &= ~kill
        train_mask &= ~test_mask
        yield train_mask, test_mask


def n_paths(n_groups: int, n_test_groups: int) -> int:
    from math import comb
    return comb(n_groups, n_test_groups) * n_test_groups // n_groups
