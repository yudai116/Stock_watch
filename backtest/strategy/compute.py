"""
strategy/compute.py — OHLCV → テクニカル指標値 → スコア行列 変換

indicators.py のスコア関数が期待する「前処理済み指標値」を
OHLCV から計算し、(n_ind, T) スコア行列を返す。

【スイング指標 (8個)】
  RSI(14), MACD_hist(12,26,9), BB%B(20,2), EMA200_dev%,
  ROC63%, Stoch%K(14,3), CCI(20), 52WK%

【デイトレ指標 (8個)】
  RSI(14), MACD_hist(12,26,9), BB%B(20,2), MA20_dev%,
  RVOL(20), VWAP_dev%, ORB_score, MOM3B%
"""
from __future__ import annotations

import numpy as np

from backtest.strategy.indicators import (
    SWING_INDICATORS, IND_NAMES_SWING,
    DAY_INDICATORS,   IND_NAMES_DAY,
    score_rsi_swing,  score_macd_swing,  score_bb_swing,    score_ema200_swing,
    score_mom3m_swing, score_stoch_swing, score_cci_swing,   score_52wk_swing,
    score_rsi_day,    score_macd_day,    score_bb_day,       score_ma_day,
    score_rvol_day,   score_vwap_bull_day, score_orb_day,    score_mom3b_day,
)


# ── 内部テクニカル計算ユーティリティ ──────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """指数移動平均 (EMA)"""
    out = np.full_like(arr, np.nan)
    k = 2.0 / (period + 1)
    # 最初の有効値を SMA で初期化
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) < period:
        return out
    start = valid[period - 1]
    out[start] = np.mean(arr[valid[0]:start + 1])
    for i in range(start + 1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """単純移動平均"""
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(period - 1, len(arr)):
        seg = arr[i - period + 1: i + 1]
        if not np.any(np.isnan(seg)):
            out[i] = np.mean(seg)
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (Wilder法)"""
    delta = np.diff(closes.astype(float))
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    avg_g = np.full(len(closes), np.nan)
    avg_l = np.full(len(closes), np.nan)

    # 最初の平均は SMA
    if len(gain) < period:
        return np.full(len(closes), np.nan)

    avg_g[period] = np.mean(gain[:period])
    avg_l[period] = np.mean(loss[:period])
    for i in range(period + 1, len(closes)):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i - 1]) / period

    rs  = np.where(avg_l > 1e-10, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi[:period] = np.nan
    return rsi


def _macd_hist(closes: np.ndarray,
               fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD ヒストグラム"""
    ema_f = _ema(closes, fast)
    ema_s = _ema(closes, slow)
    macd  = ema_f - ema_s
    sig   = _ema(macd, signal)
    return macd - sig


def _bb_pct(closes: np.ndarray, period: int = 20, k: float = 2.0) -> np.ndarray:
    """Bollinger Band %B"""
    out = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        seg = closes[i - period + 1: i + 1]
        if np.any(np.isnan(seg)):
            continue
        mid = np.mean(seg)
        std = np.std(seg)
        if std < 1e-10:
            out[i] = 0.5
        else:
            out[i] = (closes[i] - (mid - k * std)) / (2 * k * std)
    return out


def _stoch_k(closes: np.ndarray, highs: np.ndarray,
             lows: np.ndarray, period: int = 14) -> np.ndarray:
    """%K"""
    out = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        lo = np.nanmin(lows[i - period + 1: i + 1])
        hi = np.nanmax(highs[i - period + 1: i + 1])
        if hi - lo < 1e-10:
            out[i] = 50.0
        else:
            out[i] = (closes[i] - lo) / (hi - lo) * 100
    return out


def _cci(closes: np.ndarray, highs: np.ndarray,
         lows: np.ndarray, period: int = 20) -> np.ndarray:
    """CCI"""
    tp  = (highs + lows + closes) / 3.0
    out = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        seg = tp[i - period + 1: i + 1]
        if np.any(np.isnan(seg)):
            continue
        mean_tp = np.mean(seg)
        mad = np.mean(np.abs(seg - mean_tp))
        if mad < 1e-10:
            out[i] = 0.0
        else:
            out[i] = (tp[i] - mean_tp) / (0.015 * mad)
    return out


def _roc(closes: np.ndarray, period: int = 63) -> np.ndarray:
    """Rate of Change (%)"""
    out = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        prev = closes[i - period]
        if prev > 1e-10 and not np.isnan(prev):
            out[i] = (closes[i] - prev) / prev * 100
    return out


def _52wk_pct(closes: np.ndarray, highs: np.ndarray,
              lows: np.ndarray, period: int = 252) -> np.ndarray:
    """52週レンジ内位置 [0, 1]"""
    out = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        lo = np.nanmin(lows[i - period + 1: i + 1])
        hi = np.nanmax(highs[i - period + 1: i + 1])
        if hi - lo < 1e-10:
            out[i] = 0.5
        else:
            out[i] = (closes[i] - lo) / (hi - lo)
    return out


def _vwap_rolling(closes: np.ndarray, volumes: np.ndarray,
                  period: int = 20) -> np.ndarray:
    """ローリング VWAP (session-reset の代替)"""
    out = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        c = closes[i - period + 1: i + 1]
        v = volumes[i - period + 1: i + 1]
        total_vol = np.nansum(v)
        if total_vol < 1e-3:
            out[i] = closes[i]
        else:
            out[i] = np.nansum(c * v) / total_vol
    return out


def _orb_score(closes: np.ndarray, highs: np.ndarray,
               lows: np.ndarray, orb_bars: int = 3) -> np.ndarray:
    """
    ORB スコア [0, 1]: 過去 orb_bars 本の高値・安値レンジを基準に、
    現在の終値がブレイクアウトしている程度を示す。
    """
    out = np.full(len(closes), np.nan)
    for i in range(orb_bars, len(closes)):
        orb_hi = np.nanmax(highs[i - orb_bars: i])
        orb_lo = np.nanmin(lows[i - orb_bars: i])
        orb_range = orb_hi - orb_lo
        if orb_range < 1e-10:
            out[i] = 0.0
        else:
            # +1 = 完全ブレイクアウト、0 = レンジ内中央、-1 = 下ブレイク
            raw = (closes[i] - orb_hi) / orb_range
            # [0, 1] に正規化: ブレイクアウト方向のみ
            out[i] = float(np.clip(raw + 0.5, 0.0, 1.5))
    return out


def _mom3b(closes: np.ndarray, period: int = 3) -> np.ndarray:
    """3バーモメンタム (%)"""
    out = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        prev = closes[i - period]
        if prev > 1e-10 and not np.isnan(prev):
            out[i] = (closes[i] - prev) / prev * 100
    return out


def _ema200_dev(closes: np.ndarray) -> np.ndarray:
    """EMA200 乖離率 (%)"""
    ema200 = _ema(closes, 200)
    out    = np.full(len(closes), np.nan)
    valid  = (ema200 > 1e-10) & ~np.isnan(ema200)
    out    = np.where(valid, (closes - ema200) / ema200 * 100, np.nan)
    return out


def _rvol(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """相対出来高"""
    ma  = _sma(volumes.astype(float), period)
    out = np.full(len(volumes), np.nan)
    ok  = (ma > 1e-3) & ~np.isnan(ma)
    out = np.where(ok, volumes / ma, np.nan)
    return out


# ── スコア行列構築 ──────────────────────────────────────────────────────────────

def compute_swing_scores(
    closes:  np.ndarray,
    highs:   np.ndarray,
    lows:    np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """
    スイング指標のスコア行列を構築する。

    Returns
    -------
    np.ndarray (n_ind, T) — T = len(closes)
    """
    T    = len(closes)
    n    = len(IND_NAMES_SWING)
    mat  = np.zeros((n, T), dtype=np.float32)

    raw_rsi   = _rsi(closes)
    raw_macd  = _macd_hist(closes)
    raw_bb    = _bb_pct(closes)
    raw_ema   = _ema200_dev(closes)
    raw_mom   = _roc(closes, 63)
    raw_stoch = _stoch_k(closes, highs, lows)
    raw_cci   = _cci(closes, highs, lows)
    raw_52wk  = _52wk_pct(closes, highs, lows)

    fn_map = {
        "RSI":    (score_rsi_swing,    raw_rsi),
        "MACD":   (score_macd_swing,   raw_macd),
        "BB":     (score_bb_swing,     raw_bb),
        "EMA200": (score_ema200_swing, raw_ema),
        "MOM3M":  (score_mom3m_swing,  raw_mom),
        "Stoch":  (score_stoch_swing,  raw_stoch),
        "CCI":    (score_cci_swing,    raw_cci),
        "52WK":   (score_52wk_swing,   raw_52wk),
    }
    for i, name in enumerate(IND_NAMES_SWING):
        fn, raw = fn_map[name]
        mat[i]  = np.nan_to_num(fn(raw), nan=0.0).astype(np.float32)

    return mat


def compute_day_scores(
    closes:  np.ndarray,
    highs:   np.ndarray,
    lows:    np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """
    デイトレ指標のスコア行列を構築する。

    Returns
    -------
    np.ndarray (n_ind, T)
    """
    T   = len(closes)
    n   = len(IND_NAMES_DAY)
    mat = np.zeros((n, T), dtype=np.float32)

    raw_rsi  = _rsi(closes)
    raw_macd = _macd_hist(closes)
    raw_bb   = _bb_pct(closes)
    vwap     = _vwap_rolling(closes, volumes)
    raw_vwap = np.where(
        (vwap > 1e-10) & ~np.isnan(vwap),
        (closes - vwap) / vwap * 100,
        np.nan,
    )
    raw_ma = np.full(T, np.nan)
    ma20   = _sma(closes, 20)
    ok     = (ma20 > 1e-10) & ~np.isnan(ma20)
    raw_ma = np.where(ok, (closes - ma20) / ma20 * 100, np.nan)

    raw_rvol = _rvol(volumes)
    raw_orb  = _orb_score(closes, highs, lows)
    raw_mom3 = _mom3b(closes)

    fn_map = {
        "RSI":    (score_rsi_day,       raw_rsi),
        "MACD":   (score_macd_day,      raw_macd),
        "BB":     (score_bb_day,        raw_bb),
        "MA":     (score_ma_day,        raw_ma),
        "RVOL":   (score_rvol_day,      raw_rvol),
        "VWAP_B": (score_vwap_bull_day, raw_vwap),
        "ORB":    (score_orb_day,       raw_orb),
        "MOM3B":  (score_mom3b_day,     raw_mom3),
    }
    for i, name in enumerate(IND_NAMES_DAY):
        fn, raw = fn_map[name]
        mat[i]  = np.nan_to_num(fn(raw), nan=0.0).astype(np.float32)

    return mat
