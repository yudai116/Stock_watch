"""Drawdown-linked risk reduction (SPEC §6.2).

Below dd_soft: full risk. Between soft and hard: linear scale-down to the
floor. At/beyond dd_hard: no new entries (multiplier 0).
"""

from __future__ import annotations

from data.config_loader import load_config


def risk_multiplier(equity: float, peak_equity: float) -> float:
    cfg = load_config("params")["fixed"]
    soft = float(cfg["dd_soft_pct"]) / 100.0
    hard = float(cfg["dd_hard_pct"]) / 100.0
    floor = float(cfg["dd_scale_floor"])
    if peak_equity <= 0:
        return 1.0
    dd = max(0.0, 1.0 - equity / peak_equity)
    if dd <= soft:
        return 1.0
    if dd >= hard:
        return 0.0
    frac = (dd - soft) / (hard - soft)
    return 1.0 - frac * (1.0 - floor)
