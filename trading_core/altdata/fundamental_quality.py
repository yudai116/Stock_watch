"""Fundamental quality score in [0, 1] (SPEC §6.3).

Components (equal-weighted by default; weights are the ONLY ablation knob):
  * cash ratio        = cash / total_assets (level, cross-time z within symbol)
  * debt trend        = decreasing total_debt over last 4 quarters is good
  * inventory turns Δ = for semiconductors only: improving cogs/inventory

All inputs come from the bitemporal store via an AsOfView; quarterly values
are forward-filled FROM available_ts (never from period end), so a decision
at t only sees filings accepted before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.bitemporal_store import AsOfView


EMPTY = pd.DataFrame(columns=["event_ts", "available_ts", "value"])


def _metric_history(view: AsOfView, symbol: str, metric: str) -> pd.DataFrame:
    df = view.records("fundamentals", keys=[f"{symbol}|{metric}"])
    if df.empty:
        return EMPTY.copy()
    rows = [
        {"event_ts": r["event_ts"], "available_ts": r["available_ts"],
         "value": r["payload"].get("value")}
        for _, r in df.iterrows()
        if r["payload"].get("value") is not None
    ]
    if not rows:
        return EMPTY.copy()
    return pd.DataFrame(rows).sort_values("event_ts").reset_index(drop=True)


def fundamental_quality(view: AsOfView, symbol: str, is_semiconductor: bool,
                        weights: dict[str, float] | None = None) -> float:
    """Return quality score in [0,1]; 0.5 = neutral / no data."""
    w = weights or {"cash_ratio": 1.0, "debt_trend": 1.0, "inventory_turns": 1.0}
    parts: list[tuple[float, float]] = []  # (score, weight)

    cash = _metric_history(view, symbol, "cash")
    assets = _metric_history(view, symbol, "total_assets")
    if len(cash) >= 4 and len(assets) >= 4:
        merged = pd.merge_asof(cash, assets, on="event_ts", suffixes=("_c", "_a"),
                               tolerance=pd.Timedelta(days=10))
        merged = merged.dropna()
        if len(merged) >= 4:
            ratio = merged["value_c"] / merged["value_a"].replace(0, np.nan)
            z = (ratio.iloc[-1] - ratio.mean()) / (ratio.std() + 1e-9)
            parts.append((_sigmoid(z), w["cash_ratio"]))

    debt = _metric_history(view, symbol, "total_debt")
    if len(debt) >= 4:
        recent = debt["value"].tail(4).to_numpy(dtype=float)
        slope = np.polyfit(np.arange(4), recent, 1)[0]
        norm = slope / (abs(recent).mean() + 1e-9)
        parts.append((_sigmoid(-norm * 10), w["debt_trend"]))  # falling debt -> >0.5

    if is_semiconductor:
        inv = _metric_history(view, symbol, "inventory")
        cogs = _metric_history(view, symbol, "cogs")
        if len(inv) >= 4 and len(cogs) >= 4:
            m = pd.merge_asof(cogs, inv, on="event_ts", suffixes=("_cogs", "_inv"),
                              tolerance=pd.Timedelta(days=10)).dropna()
            if len(m) >= 4:
                turns = m["value_cogs"] / m["value_inv"].replace(0, np.nan)
                delta = turns.iloc[-1] - turns.iloc[-4]
                parts.append((_sigmoid(delta), w["inventory_turns"]))

    if not parts:
        return 0.5
    total_w = sum(p[1] for p in parts)
    return float(sum(s * ww for s, ww in parts) / total_w)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
