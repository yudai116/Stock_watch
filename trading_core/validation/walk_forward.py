"""Anchored walk-forward fold generation + WF efficiency (SPEC §8).

Folds: anchored train windows, disjoint consecutive OOS test windows.
Only OOS metrics are ever reported (CLAUDE.md absolute rule).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.config_loader import load_config


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp       # exclusive


def make_folds(index: pd.DatetimeIndex, n_folds: int | None = None,
               train_fraction: float | None = None) -> list[Fold]:
    cfg = load_config("params")["validation"]
    n_folds = n_folds or int(cfg["wfa_folds"])
    train_fraction = train_fraction or float(cfg["train_fraction"])
    anchored = bool(cfg["wfa_anchored"])

    idx = pd.DatetimeIndex(index).sort_values()
    n = len(idx)
    # first train block ends at train_fraction of the first fold segment;
    # OOS windows tile the remainder equally
    first_train = int(n * train_fraction * (1 / (1 + (n_folds - 1) * (1 - train_fraction))))
    oos_len = max(1, (n - first_train) // n_folds)

    folds = []
    for k in range(n_folds):
        te_s = first_train + k * oos_len
        te_e = min(te_s + oos_len, n)
        if te_s >= n:
            break
        tr_s = 0 if anchored else max(0, te_s - first_train)
        folds.append(Fold(
            fold=k,
            train_start=idx[tr_s], train_end=idx[te_s],
            test_start=idx[te_s], test_end=idx[te_e - 1] + pd.Timedelta(seconds=1),
        ))
    return folds


def wf_efficiency(is_sharpe: float, oos_sharpe: float) -> float:
    """OOS/IS Sharpe ratio; >= 0.5 required (SPEC §8.1)."""
    if is_sharpe <= 0:
        return 0.0
    return oos_sharpe / is_sharpe


def aggregate_oos(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold OOS metrics (mean; DD = worst)."""
    if not fold_metrics:
        return {}
    out = {}
    keys = fold_metrics[0].keys()
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m and m[k] is not None]
        if not vals:
            continue
        out[k] = float(min(vals)) if k == "max_dd" else float(np.mean(vals))
    out["n_folds"] = len(fold_metrics)
    return out
