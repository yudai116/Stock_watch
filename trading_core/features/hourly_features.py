"""Hourly features (SPEC §4): ATR, Donchian, RSI/ROC, volume z, MA slope, price z.

No-lookahead contract: row t uses bars with ts <= t only. Donchian channels
EXCLUDE the current bar (shift(1)) so "close > donchian_high" is a true
breakout of prior structure. Verified by tests/test_no_lookahead_features.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.config_loader import load_config


def atr(bars: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def donchian(bars: pd.DataFrame, period: int) -> pd.DataFrame:
    """Prior-bar channel: highest high / lowest low of the LAST ``period``
    bars excluding the current one."""
    hi = bars["high"].shift(1).rolling(period).max()
    lo = bars["low"].shift(1).rolling(period).min()
    return pd.DataFrame({"donchian_high": hi, "donchian_low": lo,
                         "donchian_width": (hi - lo)})


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def roc(close: pd.Series, period: int) -> pd.Series:
    return close / close.shift(period) - 1.0


def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    mu = volume.rolling(window).mean()
    sd = volume.rolling(window).std()
    return (volume - mu) / sd.replace(0, np.nan)


def ma_slope(close: pd.Series, period: int) -> pd.Series:
    ma = close.rolling(period).mean()
    return (ma - ma.shift(period // 5)) / (ma.shift(period // 5).abs() + 1e-12)


def price_zscore(close: pd.Series, window: int) -> pd.Series:
    mu = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - mu) / sd.replace(0, np.nan)


def build_hourly_frame(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """All hourly features for one symbol, indexed like ``bars``.

    ``params`` carries the GA individual's values (atr_period,
    donchian_entry_period, short_donchian_period, mr_rsi_period).
    """
    cfg = load_config("params")["features"]
    out = pd.DataFrame(index=bars.index)
    out["ts"] = bars["ts"]
    out["close"] = bars["close"]
    out["atr"] = atr(bars, int(params["atr_period"]))
    dc = donchian(bars, int(params["donchian_entry_period"]))
    out["donchian_high"] = dc["donchian_high"]
    out["donchian_width"] = dc["donchian_width"]
    dc_s = donchian(bars, int(params["short_donchian_period"]))
    out["donchian_low_s"] = dc_s["donchian_low"]
    out["rsi"] = rsi(bars["close"], int(params.get("mr_rsi_period", 14)))
    out["roc"] = roc(bars["close"], 20)
    out["volume_z"] = volume_zscore(bars["volume"], cfg["zscore_window"])
    out["ma_slope"] = ma_slope(bars["close"], cfg["hourly_ma_slope_period"])
    out["price_z"] = price_zscore(bars["close"], cfg["zscore_window"])
    return out
