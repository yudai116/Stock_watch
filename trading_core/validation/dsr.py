"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

DSR = PSR evaluated against SR0, the expected maximum Sharpe among N
independent trials. The trial count N MUST come from optimize/trial_logger
(every GA individual, ablation and manual tweak counts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def probabilistic_sharpe(sr_hat: float, sr_benchmark: float, n_obs: int,
                         skew: float, kurt: float) -> float:
    """P[SR* > sr_benchmark]. ``kurt`` is NON-excess kurtosis (normal=3)."""
    if n_obs < 2:
        return 0.0
    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2))
    z = (sr_hat - sr_benchmark) * np.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max SR] of N zero-true-SR trials with cross-trial variance var_sharpe."""
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    z1 = stats.norm.ppf(1 - 1.0 / n)
    z2 = stats.norm.ppf(1 - 1.0 / (n * np.e))
    return float(np.sqrt(var_sharpe) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int,
                    var_sharpe_across_trials: float | None = None) -> dict:
    """Returns per-period SR stats and the DSR probability.

    ``returns``: per-bar/day OOS returns of the FINAL strategy.
    ``var_sharpe_across_trials``: variance of per-period SR across all logged
    trials; default conservatively uses the strategy's own SR estimation
    variance 1/T if not supplied.
    """
    r = returns.dropna()
    T = len(r)
    if T < 10:
        return {"dsr": 0.0, "sr": 0.0, "sr0": 0.0, "n_trials": n_trials, "n_obs": T}
    sr = float(r.mean() / (r.std() + 1e-18))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    var_tr = var_sharpe_across_trials if var_sharpe_across_trials is not None else 1.0 / T
    sr0 = expected_max_sharpe(n_trials, var_tr)
    dsr_p = probabilistic_sharpe(sr, sr0, T, skew, kurt)
    return {"dsr": dsr_p, "sr": sr, "sr0": sr0, "n_trials": n_trials, "n_obs": T,
            "skew": skew, "kurt": kurt}


def passes(dsr_result: dict, confidence: float = 0.95) -> bool:
    """SPEC §8.1: DSR > 0 at the 95% level -> require PSR >= confidence."""
    return dsr_result["dsr"] >= confidence
