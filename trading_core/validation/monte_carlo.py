"""Stationary block bootstrap of OOS returns: confidence bands for Sharpe,
CAGR and max drawdown; probability of breaching the hard DD limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.config_loader import load_config


def block_bootstrap(returns: pd.Series, n_sims: int = 2000,
                    avg_block: int = 24, seed: int = 11) -> pd.DataFrame:
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < avg_block * 2:
        raise ValueError("too few observations for block bootstrap")
    rng = np.random.default_rng(seed)
    p = 1.0 / avg_block
    out = []
    for _ in range(n_sims):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
        sim = r[idx]
        eq = np.cumprod(1 + sim)
        peak = np.maximum.accumulate(eq)
        max_dd = float(((eq - peak) / peak).min())
        sd = sim.std()
        out.append({
            "sharpe_per_bar": sim.mean() / sd if sd > 0 else 0.0,
            "total_return": eq[-1] - 1,
            "max_dd": max_dd,
        })
    return pd.DataFrame(out)


def summarize(sims: pd.DataFrame, bars_per_year: float) -> dict:
    hard_dd = float(load_config("params")["targets"]["max_dd_pct"]) / 100.0
    ann = np.sqrt(bars_per_year)
    return {
        "sharpe_p05": float(sims["sharpe_per_bar"].quantile(0.05) * ann),
        "sharpe_p50": float(sims["sharpe_per_bar"].quantile(0.50) * ann),
        "sharpe_p95": float(sims["sharpe_per_bar"].quantile(0.95) * ann),
        "max_dd_p95_worst": float(sims["max_dd"].quantile(0.05)),
        "p_dd_breach_hard": float((sims["max_dd"] <= -hard_dd).mean()),
    }
