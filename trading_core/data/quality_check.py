"""Data quality checks: gaps, zero volume, extreme moves, NaNs, stale prices.

Run after each ingest. Produces a per-symbol report DataFrame; blocking
issues should stop the pipeline before feature computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Hourly return beyond this is flagged as a suspect bad print (not auto-fixed).
EXTREME_RETURN = 0.35
STALE_RUN = 24  # consecutive identical closes (hourly) -> stale feed suspicion


def check_bars(symbol: str, bars: pd.DataFrame, timeframe: str) -> dict:
    issues: list[str] = []
    if bars.empty:
        return {"symbol": symbol, "timeframe": timeframe, "n_bars": 0,
                "issues": ["EMPTY"], "blocking": True}

    n = len(bars)
    if bars[["open", "high", "low", "close"]].isna().any().any():
        issues.append("NAN_PRICES")
    if (bars["volume"] <= 0).mean() > 0.02:
        issues.append(f"ZERO_VOLUME_{(bars['volume'] <= 0).mean():.1%}")
    bad_ohlc = (bars["high"] < bars["low"]).sum()
    if bad_ohlc:
        issues.append(f"HIGH_LT_LOW_{bad_ohlc}")
    if (bars["close"] <= 0).any():
        issues.append("NONPOSITIVE_CLOSE")

    ret = bars["close"].pct_change().abs()
    n_extreme = int((ret > EXTREME_RETURN).sum())
    if n_extreme:
        issues.append(f"EXTREME_RETURNS_{n_extreme}")

    runs = (bars["close"].diff() == 0).astype(int)
    grp = (runs.diff() != 0).cumsum()
    max_run = runs.groupby(grp).sum().max() if n > 1 else 0
    if timeframe == "1h" and max_run >= STALE_RUN:
        issues.append(f"STALE_PRICE_RUN_{int(max_run)}")

    if timeframe == "1h":
        deltas = bars["ts"].diff().dropna()
        big_gaps = int((deltas > pd.Timedelta(days=5)).sum())
        if big_gaps:
            issues.append(f"GAPS_GT_5D_{big_gaps}")

    blocking = any(i.startswith(("NAN", "NONPOSITIVE", "HIGH_LT_LOW", "EMPTY")) for i in issues)
    return {"symbol": symbol, "timeframe": timeframe, "n_bars": n,
            "issues": issues, "blocking": blocking}


def check_all(store, timeframe: str = "1h") -> pd.DataFrame:
    from data.bitemporal_store import BitemporalStore  # noqa: F401
    from datetime import datetime, timezone

    as_of = datetime.now(timezone.utc)
    rows = []
    for sym in store.list_bar_symbols(timeframe):
        bars = store.get_bars_asof(sym, timeframe, as_of)
        rows.append(check_bars(sym, bars, timeframe))
    rep = pd.DataFrame(rows)
    if not rep.empty and rep["blocking"].any():
        bad = rep[rep["blocking"]]["symbol"].tolist()
        print(f"BLOCKING quality issues: {bad}")
    return rep


def main() -> None:
    from data.bitemporal_store import BitemporalStore
    from data.config_loader import data_root

    store = BitemporalStore(data_root())
    for tf in ("1h", "1d"):
        rep = check_all(store, tf)
        print(rep.to_string() if not rep.empty else f"no {tf} bars stored")


if __name__ == "__main__":
    main()
