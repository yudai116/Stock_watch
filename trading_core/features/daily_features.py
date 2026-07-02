"""Daily features (SPEC §4): realized vol, relative strength vs QQQ/SMH, 200d RS.

No-lookahead contract: every row t uses data with ts <= t only (rolling
windows ending at t). Values become usable at the day's close; hourly
consumers must look up the LAST row with ts <= decision time.
Verified by tests/test_no_lookahead_features.py (future perturbation test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.config_loader import load_config


def realized_vol(close: pd.Series, days: int | None = None) -> pd.Series:
    days = days or load_config("params")["features"]["realized_vol_days"]
    ret = np.log(close / close.shift(1))
    return ret.rolling(days).std() * np.sqrt(252)


def relative_strength(close: pd.Series, benchmark_close: pd.Series, days: int) -> pd.Series:
    """Ratio momentum: (sym / bench) change over ``days``, aligned on index."""
    ratio = close / benchmark_close.reindex(close.index).ffill()
    return ratio / ratio.shift(days) - 1.0


def rs_200(close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    days = load_config("params")["features"]["rs_long_days"]
    return relative_strength(close, benchmark_close, days)


def rs_rank(closes: dict[str, pd.Series], benchmark_close: pd.Series, days: int) -> pd.DataFrame:
    """Cross-sectional RS percentile rank in [0,1] per day (1 = strongest).

    Each row t ranks symbols by RS computed from data <= t.
    """
    rs = pd.DataFrame({s: relative_strength(c, benchmark_close, days) for s, c in closes.items()})
    return rs.rank(axis=1, pct=True)


def breadth(closes: dict[str, pd.Series], ma_days: int = 50) -> pd.Series:
    """Fraction of symbols above their own ``ma_days`` MA (market-level HMM input)."""
    panel = pd.DataFrame(closes)
    above = panel > panel.rolling(ma_days).mean()
    valid = panel.rolling(ma_days).mean().notna()
    return above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)


def build_daily_frame(close: pd.Series, bench_tech: pd.Series, bench_semi: pd.Series) -> pd.DataFrame:
    """Per-symbol daily feature frame."""
    cfg = load_config("params")["features"]
    return pd.DataFrame({
        "close": close,
        "realized_vol": realized_vol(close, cfg["realized_vol_days"]),
        "rs_qqq": relative_strength(close, bench_tech, 63),
        "rs_smh": relative_strength(close, bench_semi, 63),
        "rs_200": relative_strength(close, bench_tech, cfg["rs_long_days"]),
    })
