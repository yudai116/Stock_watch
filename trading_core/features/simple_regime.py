"""Simple regime filter (SPEC_ADDENDUM_v2 R4) — the benchmark HMM must beat.

Rule (evaluated on DAILY closes, row t uses data <= t only):
    bull  : QQQ close > 200-day MA  AND  VIX < threshold
    bear  : QQQ close < 200-day MA
    range : QQQ above MA but VIX elevated -> no new entries, hold existing

If no VIX series is available, annualized 21-day realized vol of QQQ (x100)
is used as a fallback proxy against the same threshold — VIX ~ 25 roughly
corresponds to RV ~ 25%.

The output series is indexed by day-close timestamps; consumers look up the
LAST label with ts <= decision time (same convention as the HMM regimes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.config_loader import load_config


def realized_vol_proxy(qqq_close: pd.Series, days: int = 21) -> pd.Series:
    ret = np.log(qqq_close / qqq_close.shift(1))
    return ret.rolling(days).std() * np.sqrt(252) * 100.0


def simple_regime(qqq_close: pd.Series, vix: pd.Series | None = None) -> pd.Series:
    """Regime label per day. Both inputs indexed by day-close ts (UTC)."""
    cfg = load_config("params")["simple_regime"]
    ma_days = int(cfg["ma_days"])
    threshold = float(cfg["vix_threshold"])

    ma = qqq_close.rolling(ma_days).mean()
    # time-based as-of alignment: the VIX series is stamped at ITS OWN close
    # times (e.g. 21:00 UTC), the QQQ close index at next-day midnight —
    # method="ffill" takes the latest already-available VIX value (PIT-safe).
    # A plain reindex().ffill() would yield all-NaN on non-matching stamps.
    #
    # Adversarial-review fix M2: decisions use the PREVIOUS session's VIX.
    # FRED publishes VIXCLS with a lag, so a live decision after day D's
    # close can only see day D-1's VIX. Applying shift(1) AFTER alignment
    # makes backtest and live identical by construction, independent of the
    # runner's wall-clock time. The realized-vol fallback is price-derived
    # (usable same-day) and is intentionally not lagged.
    if vix is not None:
        vol = vix.sort_index().reindex(qqq_close.index, method="ffill").shift(1)
    else:
        vol = realized_vol_proxy(qqq_close)

    labels = pd.Series(index=qqq_close.index, dtype=object)
    above = qqq_close > ma
    calm = vol < threshold
    labels[above & calm] = "bull"
    labels[~above & ma.notna()] = "bear"
    labels[above & ~calm] = "range"
    return labels.dropna()
