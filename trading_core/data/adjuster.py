"""Point-in-time split/dividend adjustment.

Bars are stored RAW. Adjustment is computed *as of* a timestamp: only
corporate actions whose ``available_ts`` has passed are applied, so a
backtest decision at time t sees exactly the adjusted series it would have
seen live at t (no future splits leak into the past).

Convention: backward adjustment. For a split with ratio r (new/old shares,
e.g. 4-for-1 -> r=4) effective at ex time e, all bars with ts < e get
price /= r and volume *= r. For a cash dividend d with ex time e, prices
before e are multiplied by (1 - d / close_before_ex) (total-return style).
"""

from __future__ import annotations

import pandas as pd

from data.bitemporal_store import AsOfView


def extract_actions(view: AsOfView, symbol: str) -> pd.DataFrame:
    """Corporate actions for ``symbol`` known at view.as_of."""
    df = view.records("corporate_actions")
    if df.empty:
        return pd.DataFrame(columns=["event_ts", "type", "ratio", "cash"])
    rows = []
    for _, r in df.iterrows():
        p = r["payload"]
        if p.get("symbol", str(r["key"]).split("|")[0]) != symbol:
            continue
        if p.get("type") == "split":
            old, new = float(p.get("old_rate", 1) or 1), float(p.get("new_rate", 1) or 1)
            ratio = new / old if old else 1.0
            rows.append({"event_ts": r["event_ts"], "type": "split", "ratio": ratio, "cash": 0.0})
        elif p.get("type") == "dividend":
            rows.append({"event_ts": r["event_ts"], "type": "dividend", "ratio": 1.0,
                         "cash": float(p.get("cash", 0) or 0)})
    if not rows:
        # actions exist in the store, but none for THIS symbol (e.g. a stock
        # that never split nor paid a dividend)
        return pd.DataFrame(columns=["event_ts", "type", "ratio", "cash"])
    return pd.DataFrame(rows).sort_values("event_ts").reset_index(drop=True)


def adjust_bars(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Apply backward adjustment for all actions in ``actions``.

    ``bars`` must be raw bars sorted by ts. Returns a copy with adjusted
    open/high/low/close/volume.
    """
    df = bars.copy()
    if df.empty or actions.empty:
        return df
    price_cols = ["open", "high", "low", "close"]
    for _, a in actions.iterrows():
        before = df["ts"] < a["event_ts"]
        if not before.any():
            continue
        if a["type"] == "split" and a["ratio"] not in (0, 1.0):
            df.loc[before, price_cols] = df.loc[before, price_cols] / a["ratio"]
            df.loc[before, "volume"] = df.loc[before, "volume"] * a["ratio"]
        elif a["type"] == "dividend" and a["cash"] > 0:
            last_close = df.loc[before, "close"].iloc[-1]
            if last_close > 0:
                factor = 1.0 - a["cash"] / last_close
                df.loc[before, price_cols] = df.loc[before, price_cols] * factor
    return df


def adjusted_bars_asof(view: AsOfView, symbol: str, timeframe: str) -> pd.DataFrame:
    """The one-stop leak-proof accessor: raw bars + PIT adjustment."""
    bars = view.bars(symbol, timeframe)
    actions = extract_actions(view, symbol)
    return adjust_bars(bars, actions)
