#!/usr/bin/env python3
"""
Sell Target Backtest — Comprehensive Exit Strategy Analysis
===========================================================

Tests 43+ exit strategies across 50 semiconductor/tech stocks using
regime-switching GBM simulation. Evaluates which exit strategy maximizes
risk-adjusted returns (Sharpe, profit factor, win rate, R:R).

Output: /home/user/Stock_watch/backtest/sell_target_results.json
"""

import json
import time
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ─── Seeded RNG (reproducible) ──────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ─── Stock universe (same as run_backtest.py) ────────────────────────────────
STOCKS = {
    "NVDA":   (0.55, 0.60),
    "AMD":    (0.35, 0.62),
    "ASML":   (0.30, 0.28),
    "LRCX":   (0.28, 0.30),
    "KLAC":   (0.25, 0.30),
    "AMAT":   (0.22, 0.32),
    "TER":    (0.18, 0.35),
    "8035.T": (0.28, 0.42),
    "6857.T": (0.25, 0.38),
    "6920.T": (0.30, 0.55),
    "TSM":    (0.18, 0.30),
    "AVGO":   (0.25, 0.35),
    "QCOM":   (0.12, 0.30),
    "INTC":   (0.02, 0.32),
    "MRVL":   (0.30, 0.45),
    "ARM":    (0.25, 0.55),
    "CDNS":   (0.22, 0.28),
    "SNPS":   (0.25, 0.28),
    "AMBA":   (0.12, 0.55),
    "4063.T": (0.15, 0.25),
    "6762.T": (0.12, 0.35),
    "6503.T": (0.10, 0.25),
    "WOLF":   (0.08, 0.65),
    "ONTO":   (0.20, 0.45),
    "MU":     (0.22, 0.48),
    "WDC":    (0.10, 0.45),
    "MSFT":   (0.32, 0.22),
    "GOOGL":  (0.22, 0.25),
    "META":   (0.22, 0.40),
    "AMZN":   (0.20, 0.28),
    "ORCL":   (0.18, 0.25),
    "CRM":    (0.22, 0.30),
    "CRWD":   (0.30, 0.45),
    "SMCI":   (0.35, 0.80),
    "SNOW":   (0.25, 0.60),
    "PLTR":   (0.18, 0.70),
    "PATH":   (0.15, 0.55),
    "SOUN":   (0.20, 1.00),
    "BBAI":   (0.10, 1.00),
    "IREN":   (0.20, 1.00),
    "AI":     (0.05, 0.85),
    "6501.T": (0.20, 0.25),
    "6702.T": (0.18, 0.28),
    "IONQ":   (0.15, 0.90),
    "RGTI":   (0.10, 1.20),
    "QBTS":   (0.10, 1.00),
    "IBM":    (0.05, 0.22),
    "OKLO":   (0.15, 0.80),
    "SMR":    (0.15, 0.75),
    "BWXT":   (0.15, 0.25),
}

SIZE_GROUPS = {
    "large": ["NVDA","AMD","ASML","TSM","AVGO","QCOM","INTC","MSFT","GOOGL","META",
              "AMZN","ORCL","CRM","IBM","MU","LRCX","KLAC","AMAT","CDNS","SNPS",
              "8035.T","4063.T","6501.T"],
    "mid":   ["MRVL","ARM","SMCI","TER","CRWD","SNOW","PLTR","WDC","PATH","6857.T",
              "6762.T","6702.T","6920.T","6503.T","BWXT","ONTO","WOLF","AMBA"],
    "small": ["SOUN","BBAI","IREN","AI","IONQ","RGTI","QBTS","OKLO","SMR"],
}

# ─── Regime-switching parameters (same as run_backtest.py) ──────────────────
TRANS = np.array([[0.97, 0.015, 0.015],
                  [0.05, 0.93,  0.02 ],
                  [0.03, 0.01,  0.96 ]])
D_MUL = np.array([1.8, -2.5,  0.1])
V_MUL = np.array([0.9,  1.8,  0.7])

# ─── Simulation config ───────────────────────────────────────────────────────
N_PATHS  = 10
N_DAYS   = 2520
WARMUP   = 60
MAX_HOLD = 20   # max holding period (backstop)

# ─── Exit strategy definitions ───────────────────────────────────────────────
# Each strategy is a dict with 'type' and parameters
EXIT_STRATEGIES = {}

# Group 1: Fixed % targets
for pct in [3, 5, 8, 10, 12, 15, 20, 25]:
    EXIT_STRATEGIES[f"PCT_{pct}"] = {"type": "pct_target", "pct": pct / 100.0}

# Group 2: ATR-based targets
for mul in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    key = f"ATR_{mul}x".replace(".0x", "x")
    EXIT_STRATEGIES[key] = {"type": "atr_target", "mul": mul}

# Group 3: Moving average exits
for k in [5, 10, 20, 50]:
    EXIT_STRATEGIES[f"EMA_{k}"] = {"type": "ema_trail", "k": k}
EXIT_STRATEGIES["EMA20_touch"] = {"type": "ema_touch", "k": 20}
EXIT_STRATEGIES["EMA50_touch"] = {"type": "ema_touch", "k": 50}

# Group 4: Bollinger Band targets (20-day, varying sigma)
for sig in [1.0, 1.5, 2.0, 2.5, 3.0]:
    key = f"BB_{sig}s".replace(".0s", "s").replace(".", "_")  # e.g. BB_1s, BB_1_5s
    # Use cleaner naming
    key = f"BB_{sig}sigma"
    EXIT_STRATEGIES[key] = {"type": "bb_upper", "sigma": sig}

# Group 5: RSI overbought exits
for thr in [60, 65, 70, 75]:
    EXIT_STRATEGIES[f"RSI_{thr}"] = {"type": "rsi_ob", "thr": thr}

# Group 6: Prior high breakout exits
for nd in [10, 20, 30, 60, 90]:
    EXIT_STRATEGIES[f"HIGH_{nd}d"] = {"type": "prior_high", "nd": nd}

# Group 7: Time stops
for td in [5, 10, 15, 20]:
    EXIT_STRATEGIES[f"TIME_{td}d"] = {"type": "time_stop", "td": td}

# Group 8: Combined (target OR time stop)
EXIT_STRATEGIES["ATR_2x_OR_TIME10"]    = {"type": "combo_or", "t1": {"type": "atr_target", "mul": 2.0}, "t2": {"type": "time_stop", "td": 10}}
EXIT_STRATEGIES["ATR_2x_OR_TIME20"]    = {"type": "combo_or", "t1": {"type": "atr_target", "mul": 2.0}, "t2": {"type": "time_stop", "td": 20}}
EXIT_STRATEGIES["ATR_1_5x_OR_TIME10"]  = {"type": "combo_or", "t1": {"type": "atr_target", "mul": 1.5}, "t2": {"type": "time_stop", "td": 10}}
EXIT_STRATEGIES["BB_2s_OR_TIME10"]     = {"type": "combo_or", "t1": {"type": "bb_upper", "sigma": 2.0}, "t2": {"type": "time_stop", "td": 10}}
EXIT_STRATEGIES["BB_2s_OR_TIME20"]     = {"type": "combo_or", "t1": {"type": "bb_upper", "sigma": 2.0}, "t2": {"type": "time_stop", "td": 20}}
EXIT_STRATEGIES["PCT_10_OR_TIME10"]    = {"type": "combo_or", "t1": {"type": "pct_target", "pct": 0.10}, "t2": {"type": "time_stop", "td": 10}}
EXIT_STRATEGIES["PCT_10_OR_TIME20"]    = {"type": "combo_or", "t1": {"type": "pct_target", "pct": 0.10}, "t2": {"type": "time_stop", "td": 20}}
EXIT_STRATEGIES["PCT_15_OR_TIME20"]    = {"type": "combo_or", "t1": {"type": "pct_target", "pct": 0.15}, "t2": {"type": "time_stop", "td": 20}}

# Group 9: Stop-loss + target combinations
EXIT_STRATEGIES["ATR_2x_tgt__ATR_1x_stp"]    = {"type": "target_stop", "tgt": {"type": "atr_target", "mul": 2.0}, "stp": {"type": "atr_stop", "mul": 1.0}}
EXIT_STRATEGIES["ATR_2x_tgt__ATR_0_5x_stp"]  = {"type": "target_stop", "tgt": {"type": "atr_target", "mul": 2.0}, "stp": {"type": "atr_stop", "mul": 0.5}}
EXIT_STRATEGIES["ATR_1_5x_tgt__ATR_1x_stp"]  = {"type": "target_stop", "tgt": {"type": "atr_target", "mul": 1.5}, "stp": {"type": "atr_stop", "mul": 1.0}}
EXIT_STRATEGIES["ATR_3x_tgt__ATR_1x_stp"]    = {"type": "target_stop", "tgt": {"type": "atr_target", "mul": 3.0}, "stp": {"type": "atr_stop", "mul": 1.0}}
EXIT_STRATEGIES["PCT_10_tgt__PCT_5_stp"]      = {"type": "target_stop", "tgt": {"type": "pct_target", "pct": 0.10}, "stp": {"type": "pct_stop", "pct": 0.05}}
EXIT_STRATEGIES["PCT_15_tgt__PCT_5_stp"]      = {"type": "target_stop", "tgt": {"type": "pct_target", "pct": 0.15}, "stp": {"type": "pct_stop", "pct": 0.05}}
EXIT_STRATEGIES["PCT_10_tgt__PCT_3_stp"]      = {"type": "target_stop", "tgt": {"type": "pct_target", "pct": 0.10}, "stp": {"type": "pct_stop", "pct": 0.03}}
EXIT_STRATEGIES["BB_2s_tgt__ATR_1x_stp"]     = {"type": "target_stop", "tgt": {"type": "bb_upper", "sigma": 2.0}, "stp": {"type": "atr_stop", "mul": 1.0}}
EXIT_STRATEGIES["BB_2s_tgt__PCT_5_stp"]      = {"type": "target_stop", "tgt": {"type": "bb_upper", "sigma": 2.0}, "stp": {"type": "pct_stop", "pct": 0.05}}

print(f"Total exit strategies: {len(EXIT_STRATEGIES)}")

# ─── Price simulation (regime-switching GBM with OHLCV) ─────────────────────

def sim_ohlcv(cagr: float, vol: float) -> dict:
    """
    Returns dict with keys: close, open, high, low, volume
    Each is shape (N_PATHS, N_DAYS).
    """
    dd = np.log(1 + cagr) / 252
    dv = vol / np.sqrt(252)

    # Generate close prices (regime-switching GBM with t-dist innovations)
    z = RNG.standard_t(df=5, size=(N_PATHS, N_DAYS)).astype(float) / np.sqrt(5 / 3)
    close = np.empty((N_PATHS, N_DAYS))

    for p in range(N_PATHS):
        reg = 0
        lr  = np.empty(N_DAYS)
        for t in range(N_DAYS):
            reg   = int(RNG.choice(3, p=TRANS[reg]))
            lr[t] = dd * D_MUL[reg] + dv * V_MUL[reg] * z[p, t]
        close[p, 0]  = 100.0
        close[p, 1:] = 100.0 * np.exp(np.cumsum(lr)[:-1])

    daily_vol = dv  # per-day vol scalar

    # Open: previous close * intraday gap
    open_px = np.empty_like(close)
    open_px[:, 0] = close[:, 0]
    gap = RNG.normal(0, daily_vol * 0.3, size=(N_PATHS, N_DAYS - 1))
    open_px[:, 1:] = close[:, :-1] * np.exp(gap)

    # High/Low: intraday range around open/close
    range_h = np.abs(RNG.normal(0, daily_vol * 0.5, size=(N_PATHS, N_DAYS)))
    range_l = np.abs(RNG.normal(0, daily_vol * 0.5, size=(N_PATHS, N_DAYS)))
    high = np.maximum(open_px, close) * (1.0 + range_h)
    low  = np.minimum(open_px, close) * (1.0 - range_l)

    # Volume: log-normal (mean ~1M, std ~0.5M)
    # log-normal params: mean=1e6, std=0.5e6 → mu=13.72, sigma≈0.47
    mu_v = np.log(1e6) - 0.5 * np.log(1 + (0.5/1.0)**2)
    sig_v = np.sqrt(np.log(1 + (0.5/1.0)**2))
    volume = RNG.lognormal(mu_v, sig_v, size=(N_PATHS, N_DAYS))

    return {"close": close, "open": open_px, "high": high, "low": low, "volume": volume}

# ─── Indicator computation ───────────────────────────────────────────────────

def compute_ema(p: np.ndarray, k: int) -> np.ndarray:
    """EMA over last axis (time), shape (P, T). Handles NaN-prefix inputs."""
    if p.ndim == 1:
        p = p[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False
    P, T = p.shape
    out = np.full((P, T), np.nan)
    if T < k:
        return out.squeeze(0) if squeeze else out
    a = 2.0 / (k + 1)

    if not np.any(np.isnan(p)):
        # Fast path: no NaNs
        out[:, k - 1] = p[:, :k].mean(1)
        for t in range(k, T):
            out[:, t] = p[:, t] * a + out[:, t - 1] * (1 - a)
    else:
        # NaN-prefix path: find first valid per row
        for pi in range(P):
            row   = p[pi]
            valid = ~np.isnan(row)
            if not valid.any():
                continue
            fv    = int(np.argmax(valid))
            start = fv + k - 1
            if start >= T:
                continue
            out[pi, start] = row[fv:start + 1].mean()
            for t in range(start + 1, T):
                if np.isnan(row[t]):
                    continue
                if np.isnan(out[pi, t - 1]):
                    # restart from here
                    out[pi, t] = row[t]
                else:
                    out[pi, t] = row[t] * a + out[pi, t - 1] * (1 - a)
    return out.squeeze(0) if squeeze else out

def compute_rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI(n) for shape (P, T)."""
    P, T = close.shape
    out  = np.full((P, T), np.nan)
    if T <= n:
        return out
    d  = np.diff(close, axis=1)
    g  = np.where(d > 0, d, 0.0)
    l  = np.where(d < 0, -d, 0.0)
    ag = g[:, :n].mean(1)
    al = l[:, :n].mean(1)
    for i in range(n, T):
        rs = np.where(al > 1e-12, ag / al, 1e9)
        out[:, i] = 100.0 - 100.0 / (1.0 + rs)
        if i < T - 1:
            ag = (ag * (n - 1) + g[:, i]) / n
            al = (al * (n - 1) + l[:, i]) / n
    return out

def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """ATR(n) for shape (P, T). Returns (P, T)."""
    P, T = close.shape
    out  = np.full((P, T), np.nan)
    # True range
    tr = np.zeros((P, T))
    tr[:, 0] = high[:, 0] - low[:, 0]
    tr[:, 1:] = np.maximum(
        high[:, 1:] - low[:, 1:],
        np.maximum(
            np.abs(high[:, 1:] - close[:, :-1]),
            np.abs(low[:, 1:]  - close[:, :-1])
        )
    )
    # Wilder smoothing
    out[:, n - 1] = tr[:, :n].mean(1)
    for t in range(n, T):
        out[:, t] = (out[:, t - 1] * (n - 1) + tr[:, t]) / n
    return out

def compute_macd_hist(close: np.ndarray) -> np.ndarray:
    """MACD histogram (MACD - signal)."""
    ema12 = compute_ema(close, 12)
    ema26 = compute_ema(close, 26)
    macd_line = ema12 - ema26
    signal = compute_ema(macd_line, 9)
    return macd_line - signal

def compute_bb_upper(close: np.ndarray, n: int = 20, sigma: float = 2.0) -> np.ndarray:
    """Bollinger upper band for shape (P, T)."""
    P, T = close.shape
    out  = np.full((P, T), np.nan)
    for t in range(n - 1, T):
        w = close[:, t - n + 1:t + 1]
        m = w.mean(1)
        s = w.std(1, ddof=1)
        out[:, t] = m + sigma * s
    return out

def compute_vwap(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Rolling 20-day VWAP."""
    P, T = close.shape
    n = 20
    out = np.full((P, T), np.nan)
    for t in range(n - 1, T):
        v_sl  = volume[:, t - n + 1:t + 1]
        c_sl  = close[:,  t - n + 1:t + 1]
        denom = v_sl.sum(1)
        ok = denom > 0
        out[ok, t] = (c_sl[ok] * v_sl[ok]).sum(1) / denom[ok]
    return out

def compute_stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray, k: int = 14) -> np.ndarray:
    """Stochastic %K(k)."""
    P, T = close.shape
    out  = np.full((P, T), np.nan)
    for t in range(k - 1, T):
        h_max = high[:, t - k + 1:t + 1].max(1)
        l_min = low[:,  t - k + 1:t + 1].min(1)
        rng   = h_max - l_min
        ok    = rng > 1e-12
        out[ok, t] = (close[ok, t] - l_min[ok]) / rng[ok] * 100.0
    return out

def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """ADX(n)."""
    P, T = close.shape
    out  = np.full((P, T), np.nan)
    # +DM, -DM
    up   = np.diff(high,  axis=1)
    down = -np.diff(low,  axis=1)  # positive when low drops
    dm_p = np.where((up > down) & (up > 0), up,   0.0)
    dm_m = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.zeros((P, T))
    tr[:, 0] = high[:, 0] - low[:, 0]
    tr[:, 1:] = np.maximum(
        high[:, 1:] - low[:, 1:],
        np.maximum(np.abs(high[:, 1:] - close[:, :-1]),
                   np.abs(low[:,  1:] - close[:, :-1]))
    )
    # Wilder smoothing
    atr14 = np.full((P, T), np.nan)
    dmp14 = np.full((P, T), np.nan)
    dmm14 = np.full((P, T), np.nan)
    atr14[:, n - 1] = tr[:, :n].mean(1)
    dmp14[:, n - 1] = dm_p[:, :n].mean(1)
    dmm14[:, n - 1] = dm_m[:, :n].mean(1)
    for t in range(n, T):
        atr14[:, t] = (atr14[:, t-1] * (n-1) + tr[:,   t])  / n
        dmp14[:, t] = (dmp14[:, t-1] * (n-1) + dm_p[:, t-1]) / n
        dmm14[:, t] = (dmm14[:, t-1] * (n-1) + dm_m[:, t-1]) / n
    ok = atr14 > 1e-12
    di_p = np.where(ok, 100.0 * dmp14 / atr14, np.nan)
    di_m = np.where(ok, 100.0 * dmm14 / atr14, np.nan)
    di_sum = np.abs(di_p) + np.abs(di_m)
    di_diff = np.abs(di_p - di_m)
    dx = np.where(di_sum > 1e-12, 100.0 * di_diff / di_sum, np.nan)
    # ADX = Wilder smooth of DX
    adx14 = np.full((P, T), np.nan)
    start = 2 * n - 1
    if start < T:
        valid_rows = ~np.isnan(dx[:, start - n + 1:start + 1]).any(1)
        if valid_rows.any():
            adx14[valid_rows, start] = np.nanmean(dx[valid_rows, start - n + 1:start + 1], axis=1)
        for t in range(start + 1, T):
            prev_ok = ~np.isnan(adx14[:, t-1]) & ~np.isnan(dx[:, t])
            adx14[prev_ok, t] = (adx14[prev_ok, t-1] * (n-1) + dx[prev_ok, t]) / n
    return adx14

def compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    P, T = close.shape
    obv  = np.zeros((P, T))
    d    = np.diff(close, axis=1)
    sign = np.sign(d)
    for t in range(1, T):
        obv[:, t] = obv[:, t-1] + sign[:, t-1] * volume[:, t]
    return obv

def compute_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                volume: np.ndarray, n: int = 14) -> np.ndarray:
    """Money Flow Index."""
    P, T = close.shape
    out  = np.full((P, T), np.nan)
    tp   = (high + low + close) / 3.0
    mf   = tp * volume
    for t in range(n, T):
        pos = np.where(np.diff(tp[:, t-n:t+1], axis=1) > 0,
                       mf[:, t-n+1:t+1], 0.0).sum(1)
        neg = np.where(np.diff(tp[:, t-n:t+1], axis=1) < 0,
                       mf[:, t-n+1:t+1], 0.0).sum(1)
        ok  = neg > 1e-12
        mfr = np.where(ok, pos / neg, 1e9)
        out[:, t] = 100.0 - 100.0 / (1.0 + mfr)
    return out

def compute_williams_r(high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, n: int = 14) -> np.ndarray:
    """Williams %R."""
    P, T = close.shape
    out  = np.full((P, T), np.nan)
    for t in range(n - 1, T):
        h_max = high[:, t - n + 1:t + 1].max(1)
        l_min = low[:,  t - n + 1:t + 1].min(1)
        rng   = h_max - l_min
        ok    = rng > 1e-12
        out[ok, t] = -100.0 * (h_max[ok] - close[ok, t]) / rng[ok]
    return out

# ─── Buy signal detection ────────────────────────────────────────────────────

def detect_buy_signals(ohlcv: dict) -> np.ndarray:
    """
    Returns boolean array (P, T) where True = buy signal fired.
    Conditions:
    1. RSI(14) was below 40 in last 5 days AND is now >= 38 (recovering)
    2. Price is above EMA20 OR within 2% below EMA20
    3. MACD histogram turned positive in last 3 days
    """
    close  = ohlcv["close"]
    P, T   = close.shape
    rsi    = compute_rsi(close, 14)
    ema20  = compute_ema(close, 20)
    macd_h = compute_macd_hist(close)

    signals = np.zeros((P, T), dtype=bool)

    for t in range(WARMUP, T - 1):  # -1 because we enter next day
        # Condition 1: RSI recovering from oversold
        rsi_now = rsi[:, t]
        rsi_ok  = rsi_now >= 38.0
        # Was below 40 in last 5 days
        t_start = max(0, t - 4)
        rsi_was_low = (rsi[:, t_start:t] < 40.0).any(axis=1)
        cond1 = rsi_ok & rsi_was_low

        # Condition 2: Price above EMA20 or within 2% below
        ema_now   = ema20[:, t]
        price_now = close[:, t]
        cond2 = ~np.isnan(ema_now) & (price_now >= ema_now * 0.98)

        # Condition 3: MACD histogram turned positive in last 3 days
        t_s = max(0, t - 2)
        mh_window = macd_h[:, t_s:t + 1]
        # At least one bar in window has positive MACD histogram, AND current > some prior negative
        mh_now    = macd_h[:, t]
        mh_turned = (mh_window > 0).any(axis=1) & ~np.isnan(mh_now)
        cond3 = mh_turned

        sig = cond1 & cond2 & cond3 & ~np.isnan(rsi_now)
        signals[:, t] = sig

    return signals

# ─── Pre-compute all indicators needed for exits ─────────────────────────────

def precompute_indicators(ohlcv: dict) -> dict:
    """Pre-compute all indicators needed for exit strategies."""
    close  = ohlcv["close"]
    high   = ohlcv["high"]
    low    = ohlcv["low"]
    volume = ohlcv["volume"]

    ind = {}
    ind["rsi"]    = compute_rsi(close, 14)
    ind["ema5"]   = compute_ema(close, 5)
    ind["ema10"]  = compute_ema(close, 10)
    ind["ema20"]  = compute_ema(close, 20)
    ind["ema50"]  = compute_ema(close, 50)
    ind["atr14"]  = compute_atr(high, low, close, 14)
    ind["bb_upper"] = {
        1.0: compute_bb_upper(close, 20, 1.0),
        1.5: compute_bb_upper(close, 20, 1.5),
        2.0: compute_bb_upper(close, 20, 2.0),
        2.5: compute_bb_upper(close, 20, 2.5),
        3.0: compute_bb_upper(close, 20, 3.0),
    }
    # Prior rolling highs (for exit)
    # We pre-compute for each look-back period
    ind["roll_high"] = {}
    for nd in [10, 20, 30, 60, 90]:
        rh = np.full_like(close, np.nan)
        for t in range(nd, close.shape[1]):
            rh[:, t] = high[:, t - nd:t].max(1)
        ind["roll_high"][nd] = rh

    # Additional for Layer 5 indicator candidates
    ind["vwap"]    = compute_vwap(close, volume)
    ind["stoch"]   = compute_stoch(high, low, close, 14)
    ind["adx"]     = compute_adx(high, low, close, 14)
    ind["obv"]     = compute_obv(close, volume)
    ind["mfi"]     = compute_mfi(high, low, close, volume, 14)
    ind["williams_r"] = compute_williams_r(high, low, close, 14)

    # Rolling high for 52-week (252 days)
    rh52 = np.full_like(close, np.nan)
    for t in range(252, close.shape[1]):
        rh52[:, t] = high[:, t - 252:t].max(1)
    ind["high_52w"] = rh52

    # Rolling avg volume (20d) for surge detection
    rv20 = np.full_like(close, np.nan)
    for t in range(20, close.shape[1]):
        rv20[:, t] = volume[:, t - 20:t].mean(1)
    ind["avg_vol_20"] = rv20

    # ATR as fraction of price (ATR normalization)
    with np.errstate(invalid='ignore', divide='ignore'):
        ind["atr_ratio"] = np.where(close > 0, ind["atr14"] / close, np.nan)

    return ind

# ─── Single trade exit simulation ────────────────────────────────────────────

def get_exit_day(strategy: dict, path_idx: int, entry_day: int, entry_price: float,
                 ohlcv: dict, ind: dict) -> tuple:
    """
    Returns (exit_day, exit_price, exit_type) for a single trade.
    exit_type: 'target', 'stop', 'time', 'ema_trail', etc.
    Max hold = MAX_HOLD days.
    """
    T     = ohlcv["close"].shape[1]
    p     = path_idx
    stype = strategy["type"]
    max_day = min(entry_day + MAX_HOLD, T - 1)

    close  = ohlcv["close"]
    high   = ohlcv["high"]
    low    = ohlcv["low"]

    def _pct_target_hit(t, pct_override=None):
        tgt = pct_override if pct_override is not None else strategy.get("pct", 0)
        return high[p, t] >= entry_price * (1 + tgt)

    def _atr_target_hit(t, mul):
        atr_entry = ind["atr14"][p, entry_day]
        if np.isnan(atr_entry):
            return False
        return high[p, t] >= entry_price + mul * atr_entry

    def _atr_stop_hit(t, mul):
        atr_entry = ind["atr14"][p, entry_day]
        if np.isnan(atr_entry):
            return False
        return low[p, t] <= entry_price - mul * atr_entry

    def _pct_stop_hit(t, pct):
        return low[p, t] <= entry_price * (1 - pct)

    def _bb_upper_hit(t, sigma):
        bb_val = ind["bb_upper"][sigma][p, t]
        if np.isnan(bb_val):
            return False
        return high[p, t] >= bb_val

    def _rsi_ob_hit(t, thr):
        return ind["rsi"][p, t] >= thr

    def _prior_high_hit(t, nd):
        ph = ind["roll_high"][nd][p, entry_day]
        if np.isnan(ph):
            return False
        return high[p, t] >= ph

    def _time_stop_hit(t, td, entry_d):
        return t >= entry_d + td

    # EMA trail: price crossed above then back below EMA
    def _ema_trail(t, ema_arr):
        ec = ema_arr[p, t]
        if np.isnan(ec):
            return False, False
        crossed_above = close[p, t] > ec
        return crossed_above, close[p, t] < ec

    if stype == "pct_target":
        for t in range(entry_day + 1, max_day + 1):
            if _pct_target_hit(t):
                tgt = strategy["pct"]
                return t, entry_price * (1 + tgt), "target"
        return max_day, close[p, max_day], "time"

    elif stype == "atr_target":
        mul = strategy["mul"]
        for t in range(entry_day + 1, max_day + 1):
            if _atr_target_hit(t, mul):
                atr_entry = ind["atr14"][p, entry_day]
                return t, entry_price + mul * atr_entry, "target"
        return max_day, close[p, max_day], "time"

    elif stype == "bb_upper":
        sigma = strategy["sigma"]
        for t in range(entry_day + 1, max_day + 1):
            if _bb_upper_hit(t, sigma):
                bb_val = ind["bb_upper"][sigma][p, t]
                return t, bb_val, "target"
        return max_day, close[p, max_day], "time"

    elif stype == "rsi_ob":
        thr = strategy["thr"]
        for t in range(entry_day + 1, max_day + 1):
            if _rsi_ob_hit(t, thr):
                return t, close[p, t], "target"
        return max_day, close[p, max_day], "time"

    elif stype == "prior_high":
        nd = strategy["nd"]
        for t in range(entry_day + 1, max_day + 1):
            if _prior_high_hit(t, nd):
                ph = ind["roll_high"][nd][p, entry_day]
                return t, ph, "target"
        return max_day, close[p, max_day], "time"

    elif stype == "time_stop":
        td = strategy["td"]
        exit_day = min(entry_day + td, max_day)
        return exit_day, close[p, exit_day], "time"

    elif stype == "ema_trail":
        k    = strategy["k"]
        ema_arr = ind[f"ema{k}"]
        above = False
        for t in range(entry_day + 1, max_day + 1):
            ec = ema_arr[p, t]
            if np.isnan(ec):
                continue
            if not above and close[p, t] > ec:
                above = True
            elif above and close[p, t] < ec:
                return t, close[p, t], "trail"
        return max_day, close[p, max_day], "time"

    elif stype == "ema_touch":
        k    = strategy["k"]
        ema_arr = ind[f"ema{k}"]
        # Entry is when price is below EMA; exit when it reaches EMA from below
        for t in range(entry_day + 1, max_day + 1):
            ec = ema_arr[p, t]
            if np.isnan(ec):
                continue
            if close[p, t] >= ec:
                return t, ec, "target"
        return max_day, close[p, max_day], "time"

    elif stype == "combo_or":
        # Whichever fires first: t1 (profit target) or t2 (time stop)
        t1_cfg = strategy["t1"]
        t2_cfg = strategy["t2"]
        # Time stop day
        time_td  = t2_cfg.get("td", MAX_HOLD)
        time_day = min(entry_day + time_td, max_day)

        for t in range(entry_day + 1, time_day + 1):
            hit = False
            ep  = None
            et  = "target"
            if t1_cfg["type"] == "pct_target":
                hit = _pct_target_hit(t, pct_override=t1_cfg["pct"])
                if hit:
                    ep = entry_price * (1 + t1_cfg["pct"])
            elif t1_cfg["type"] == "atr_target":
                hit = _atr_target_hit(t, t1_cfg["mul"])
                if hit:
                    atr_e = ind["atr14"][p, entry_day]
                    ep = entry_price + t1_cfg["mul"] * atr_e if not np.isnan(atr_e) else close[p, t]
            elif t1_cfg["type"] == "bb_upper":
                hit = _bb_upper_hit(t, t1_cfg["sigma"])
                if hit:
                    ep = ind["bb_upper"][t1_cfg["sigma"]][p, t]
            if hit:
                return t, ep if ep is not None else close[p, t], "target"

        return time_day, close[p, time_day], "time"

    elif stype == "target_stop":
        tgt_cfg = strategy["tgt"]
        stp_cfg = strategy["stp"]

        for t in range(entry_day + 1, max_day + 1):
            # Check stop first (conservative)
            stp_hit = False
            if stp_cfg["type"] == "atr_stop":
                stp_hit = _atr_stop_hit(t, stp_cfg["mul"])
                if stp_hit:
                    atr_e = ind["atr14"][p, entry_day]
                    ep = entry_price - stp_cfg["mul"] * atr_e if not np.isnan(atr_e) else low[p, t]
                    return t, ep, "stop"
            elif stp_cfg["type"] == "pct_stop":
                stp_hit = _pct_stop_hit(t, stp_cfg["pct"])
                if stp_hit:
                    return t, entry_price * (1 - stp_cfg["pct"]), "stop"

            # Check target
            tgt_hit = False
            ep_tgt  = None
            if tgt_cfg["type"] == "pct_target":
                tgt_hit = _pct_target_hit(t, pct_override=tgt_cfg["pct"])
                if tgt_hit:
                    ep_tgt = entry_price * (1 + tgt_cfg["pct"])
            elif tgt_cfg["type"] == "atr_target":
                tgt_hit = _atr_target_hit(t, tgt_cfg["mul"])
                if tgt_hit:
                    atr_e = ind["atr14"][p, entry_day]
                    ep_tgt = entry_price + tgt_cfg["mul"] * atr_e if not np.isnan(atr_e) else high[p, t]
            elif tgt_cfg["type"] == "bb_upper":
                tgt_hit = _bb_upper_hit(t, tgt_cfg["sigma"])
                if tgt_hit:
                    ep_tgt = ind["bb_upper"][tgt_cfg["sigma"]][p, t]
            if tgt_hit:
                return t, ep_tgt if ep_tgt is not None else close[p, t], "target"

        return max_day, close[p, max_day], "time"

    else:
        return max_day, close[p, max_day], "time"


# ─── Backtest engine ─────────────────────────────────────────────────────────

def run_strategy_backtest(ohlcv: dict, ind: dict, signals: np.ndarray,
                          strategy: dict) -> dict:
    """
    Run a single exit strategy over all paths and return trade statistics.
    Returns dict with aggregated metrics.
    """
    close  = ohlcv["close"]
    high   = ohlcv["high"]
    low    = ohlcv["low"]
    P, T   = close.shape

    returns_list   = []
    hold_days_list = []
    mae_list       = []
    mfe_list       = []
    exit_types     = []
    entry_days_list = []

    for p in range(P):
        in_trade    = False
        entry_day   = -1
        entry_price = 0.0

        for t in range(WARMUP, T):
            if in_trade:
                # Check if we've hit max hold (already handled by get_exit_day)
                continue

            if signals[p, t]:
                # Enter next day's open
                entry_t = t + 1
                if entry_t >= T:
                    continue
                entry_price = ohlcv["open"][p, entry_t]
                if entry_price <= 0:
                    continue

                # Get exit
                exit_day, exit_price, exit_type = get_exit_day(
                    strategy, p, entry_t, entry_price, ohlcv, ind
                )

                # Compute return
                ret = (exit_price - entry_price) / entry_price
                hold = exit_day - entry_t

                # MAE / MFE (using high/low during trade)
                if hold > 0:
                    trade_highs = high[p, entry_t:exit_day + 1]
                    trade_lows  = low[p,  entry_t:exit_day + 1]
                    mfe_val = (trade_highs.max() - entry_price) / entry_price
                    mae_val = (entry_price - trade_lows.min()) / entry_price
                else:
                    mfe_val = max(0.0, ret)
                    mae_val = max(0.0, -ret)

                returns_list.append(ret)
                hold_days_list.append(max(hold, 0))
                mae_list.append(mae_val)
                mfe_list.append(mfe_val)
                exit_types.append(exit_type)
                entry_days_list.append(entry_t)

                # Skip to exit day (no re-entry until exit)
                t = exit_day  # This won't work in Python for loop — we'll handle differently

    # Since we can't modify loop variable in Python for loops, we need a different approach
    # Let's redo with while loop
    returns_list   = []
    hold_days_list = []
    mae_list       = []
    mfe_list       = []
    exit_types     = []
    entry_days_list = []

    for p in range(P):
        t = WARMUP
        while t < T:
            if signals[p, t]:
                entry_t = t + 1
                if entry_t >= T:
                    t += 1
                    continue
                entry_price = ohlcv["open"][p, entry_t]
                if entry_price <= 0:
                    t += 1
                    continue

                exit_day, exit_price, exit_type = get_exit_day(
                    strategy, p, entry_t, entry_price, ohlcv, ind
                )

                ret  = (exit_price - entry_price) / entry_price
                hold = max(exit_day - entry_t, 0)

                if hold > 0:
                    trade_highs = high[p, entry_t:exit_day + 1]
                    trade_lows  = low[p,  entry_t:exit_day + 1]
                    mfe_val = (trade_highs.max() - entry_price) / entry_price
                    mae_val = (entry_price - trade_lows.min()) / entry_price
                else:
                    mfe_val = max(0.0, ret)
                    mae_val = max(0.0, -ret)

                returns_list.append(ret)
                hold_days_list.append(hold)
                mae_list.append(mae_val)
                mfe_list.append(mfe_val)
                exit_types.append(exit_type)

                # Jump to exit day + 1 (no re-entry until cleared)
                t = exit_day + 1
            else:
                t += 1

    if len(returns_list) == 0:
        return {"n_trades": 0, "mean_return": np.nan, "sharpe_ratio": np.nan,
                "win_rate": np.nan, "profit_factor": np.nan}

    rets       = np.array(returns_list)
    holds      = np.array(hold_days_list)
    maes       = np.array(mae_list)
    mfes       = np.array(mfe_list)
    n          = len(rets)
    avg_hold   = holds.mean() if n > 0 else 1.0

    mean_ret   = float(rets.mean())
    std_ret    = float(rets.std(ddof=1)) if n > 1 else 1e-6
    sharpe     = float(mean_ret / std_ret * np.sqrt(252.0 / max(avg_hold, 1))) if std_ret > 1e-12 else 0.0
    median_ret = float(np.median(rets))
    wins       = rets[rets > 0]
    losses     = rets[rets <= 0]
    win_rate   = float((rets > 0).mean())
    avg_win    = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss   = float(losses.mean()) if len(losses) > 0 else 0.0
    pf_num     = float(wins.sum()) if len(wins) > 0 else 0.0
    pf_den     = float(abs(losses.sum())) if len(losses) > 0 else 1e-9
    profit_fac = pf_num / pf_den if pf_den > 1e-12 else 999.0
    rr         = avg_win / abs(avg_loss) if abs(avg_loss) > 1e-12 else 999.0

    # Hit rate by day (% trades that achieved positive return by day N)
    hit_by_day = {}
    for nd in [1, 3, 5, 10, 15, 20]:
        mask = holds <= nd
        if mask.sum() > 0:
            hit_by_day[nd] = float((rets[mask] > 0).mean())
        else:
            hit_by_day[nd] = 0.0

    return {
        "n_trades":      n,
        "mean_return":   round(mean_ret,   6),
        "std_return":    round(std_ret,    6),
        "sharpe_ratio":  round(sharpe,     4),
        "median_return": round(median_ret, 6),
        "win_rate":      round(win_rate,   4),
        "avg_win":       round(avg_win,    6),
        "avg_loss":      round(avg_loss,   6),
        "profit_factor": round(profit_fac, 4),
        "rr":            round(rr,         4),
        "avg_hold_days": round(float(avg_hold), 2),
        "mae":           round(float(maes.mean()), 6),
        "mfe":           round(float(mfes.mean()), 6),
        "hit_by_day":    hit_by_day,
        "exit_type_pct": {
            "target": float((np.array(exit_types) == "target").mean()),
            "stop":   float((np.array(exit_types) == "stop").mean()),
            "time":   float((np.array(exit_types) == "time").mean()),
            "trail":  float((np.array(exit_types) == "trail").mean()),
        }
    }


# ─── Layer 5: Indicator candidate analysis ───────────────────────────────────

def analyze_indicator_candidates(ohlcv: dict, ind: dict, signals: np.ndarray) -> dict:
    """
    For each candidate indicator, split buy signal days into favorable/unfavorable
    and compare 10-day forward return distribution.
    """
    close  = ohlcv["close"]
    high   = ohlcv["high"]
    P, T   = close.shape
    fwd_10 = np.full((P, T), np.nan)
    for t in range(T - 10):
        ep = close[:, t]
        xp = close[:, t + 10]
        with np.errstate(invalid='ignore', divide='ignore'):
            fwd_10[:, t] = np.where(ep > 0, (xp - ep) / ep, np.nan)

    results = {}

    def compare_groups(fav_mask: np.ndarray, name: str):
        """fav_mask: (P, T) bool — favorable condition at signal day."""
        sig_and_fav  = signals & fav_mask
        sig_and_unfav = signals & ~fav_mask

        ret_fav   = fwd_10[sig_and_fav  & ~np.isnan(fwd_10)]
        ret_unfav = fwd_10[sig_and_unfav & ~np.isnan(fwd_10)]

        def stats(r):
            if len(r) < 5:
                return {"n": len(r), "mean": np.nan, "std": np.nan, "sharpe": np.nan}
            std = r.std(ddof=1)
            sh  = float(r.mean() / std * np.sqrt(252 / 10)) if std > 1e-12 else 0.0
            return {"n": int(len(r)), "mean": round(float(r.mean()), 6),
                    "std": round(float(std), 6), "sharpe": round(sh, 4)}

        sf = stats(ret_fav)
        su = stats(ret_unfav)
        delta_sh = sf["sharpe"] - su["sharpe"] if not np.isnan(sf["sharpe"]) and not np.isnan(su["sharpe"]) else np.nan
        return {"favorable": sf, "unfavorable": su, "delta_sharpe_10d": round(delta_sh, 4) if not np.isnan(delta_sh) else None}

    # 1. ATR normalization: low ATR ratio = below 50th percentile of ATR/price
    atr_ratio = ind["atr_ratio"]
    atr_median = np.nanmedian(atr_ratio)
    results["ATR_norm_low"] = compare_groups(
        signals & (atr_ratio < atr_median) & ~np.isnan(atr_ratio), "ATR_norm_low"
    )

    # 2. Volume surge: volume > 1.5x 20d avg
    vol_surge = ohlcv["volume"] > (ind["avg_vol_20"] * 1.5)
    results["Volume_surge"] = compare_groups(
        signals & vol_surge & ~np.isnan(ind["avg_vol_20"]), "Volume_surge"
    )

    # 3. Days from 52w high: price < 70% of 52w high (far from high)
    pct_from_52w = close / ind["high_52w"]
    far_from_high = (pct_from_52w < 0.85) & ~np.isnan(ind["high_52w"])
    results["Far_from_52w_high"] = compare_groups(
        signals & far_from_high, "Far_from_52w_high"
    )

    # 4. Stochastic oversold: Stoch %K < 30
    stoch_oversold = ind["stoch"] < 30.0
    results["Stoch_oversold"] = compare_groups(
        signals & stoch_oversold & ~np.isnan(ind["stoch"]), "Stoch_oversold"
    )

    # 5. Williams %R oversold: W%R < -80
    wr_oversold = ind["williams_r"] < -80.0
    results["Williams_R_oversold"] = compare_groups(
        signals & wr_oversold & ~np.isnan(ind["williams_r"]), "Williams_R_oversold"
    )

    # 6. MFI oversold: MFI < 40
    mfi_low = ind["mfi"] < 40.0
    results["MFI_oversold"] = compare_groups(
        signals & mfi_low & ~np.isnan(ind["mfi"]), "MFI_oversold"
    )

    # 7. OBV rising: OBV today > OBV 5 days ago
    obv = ind["obv"]
    obv_rising = np.zeros((P, T), dtype=bool)
    obv_rising[:, 5:] = obv[:, 5:] > obv[:, :-5]
    results["OBV_rising"] = compare_groups(
        signals & obv_rising, "OBV_rising"
    )

    # 8. VWAP below (price < VWAP = underpriced vs VWAP)
    vwap_below = (close < ind["vwap"]) & ~np.isnan(ind["vwap"])
    results["VWAP_below"] = compare_groups(
        signals & vwap_below, "VWAP_below"
    )

    # 9. ADX strong trend: ADX > 25
    adx_strong = ind["adx"] > 25.0
    results["ADX_strong"] = compare_groups(
        signals & adx_strong & ~np.isnan(ind["adx"]), "ADX_strong"
    )

    return results


# ─── Aggregate results across size groups ────────────────────────────────────

def aggregate_strategies(strategy_results: dict) -> dict:
    """Compute overall stats across all trades."""
    agg = {}
    for strat, sr in strategy_results.items():
        # sr is list of trade-level stats dicts
        all_rets   = []
        all_holds  = []
        all_wins   = []
        all_losses = []
        n_total    = 0

        # sr is already aggregated metrics dict per (strat, group)
        # We store as-is
        agg[strat] = sr
    return agg


# ─── Main backtest loop ───────────────────────────────────────────────────────

def main():
    t_global = time.time()
    print("=" * 72)
    print("  SELL TARGET BACKTEST — Comprehensive Exit Strategy Analysis")
    print("=" * 72)
    print(f"  {len(EXIT_STRATEGIES)} exit strategies  |  "
          f"{len(STOCKS)} stocks  |  {N_PATHS} paths/stock  |  {N_DAYS} days")
    print(f"  Size groups: large({len(SIZE_GROUPS['large'])})  "
          f"mid({len(SIZE_GROUPS['mid'])})  small({len(SIZE_GROUPS['small'])})\n")

    # ── Run per-size-group analysis ──────────────────────────────────────────
    all_results = {}         # [group][strategy] = metrics dict
    all_ind_results = {}     # [group][indicator] = delta_sharpe

    for gname, tickers in SIZE_GROUPS.items():
        print(f"\n{'='*72}")
        print(f"  GROUP: {gname.upper()} ({len(tickers)} stocks)")
        print(f"{'='*72}")

        group_strategy_stats = {s: [] for s in EXIT_STRATEGIES}
        group_ind_stats = {k: {"fav_rets": [], "unfav_rets": []} for k in [
            "ATR_norm_low","Volume_surge","Far_from_52w_high","Stoch_oversold",
            "Williams_R_oversold","MFI_oversold","OBV_rising","VWAP_below","ADX_strong"
        ]}

        stocks_done = 0
        for ticker in tickers:
            if ticker not in STOCKS:
                continue
            cagr, vol = STOCKS[ticker]
            try:
                ohlcv   = sim_ohlcv(cagr, vol)
                ind     = precompute_indicators(ohlcv)
                signals = detect_buy_signals(ohlcv)

                n_signals = int(signals.sum())
                if n_signals == 0:
                    print(f"    {ticker}: 0 signals, skipping")
                    continue

                # Run all exit strategies
                strat_results_this_stock = {}
                for sname, scfg in EXIT_STRATEGIES.items():
                    try:
                        stats = run_strategy_backtest(ohlcv, ind, signals, scfg)
                        strat_results_this_stock[sname] = stats
                        group_strategy_stats[sname].append(stats)
                    except Exception as e:
                        print(f"    ERROR {ticker}/{sname}: {e}")

                # Layer 5: indicator candidates
                try:
                    ind_cands = analyze_indicator_candidates(ohlcv, ind, signals)
                    for ind_name, ic in ind_cands.items():
                        if ind_name in group_ind_stats:
                            fav  = ic.get("favorable",   {})
                            unfav = ic.get("unfavorable", {})
                            nf    = fav.get("n", 0)
                            nu    = unfav.get("n", 0)
                            if nf >= 5:
                                group_ind_stats[ind_name]["fav_rets"].extend(
                                    [fav["mean"]] * nf)
                            if nu >= 5:
                                group_ind_stats[ind_name]["unfav_rets"].extend(
                                    [unfav["mean"]] * nu)
                except Exception as e:
                    print(f"    ERROR layer5 {ticker}: {e}")

            except Exception as e:
                print(f"    ERROR simulating {ticker}: {e}")
                continue

            stocks_done += 1
            if stocks_done % 5 == 0:
                elapsed = time.time() - t_global
                print(f"  [{gname}] {stocks_done}/{len(tickers)} stocks done  ({elapsed:.0f}s elapsed)")

        print(f"  [{gname}] All {stocks_done} stocks done  ({time.time()-t_global:.0f}s)")

        # Aggregate strategy stats across stocks
        group_agg = {}
        for sname, stats_list in group_strategy_stats.items():
            if not stats_list:
                group_agg[sname] = {"n_trades": 0}
                continue
            # Aggregate weighted by n_trades
            total_n = sum(s.get("n_trades", 0) for s in stats_list)
            if total_n == 0:
                group_agg[sname] = {"n_trades": 0}
                continue

            def wt_avg(key):
                vals = [s.get(key, np.nan) for s in stats_list]
                ns   = [s.get("n_trades", 0) for s in stats_list]
                pairs = [(v, n) for v, n in zip(vals, ns) if not np.isnan(v) and n > 0]
                if not pairs:
                    return np.nan
                return np.average([p[0] for p in pairs], weights=[p[1] for p in pairs])

            # Sharpe: re-compute from aggregated returns
            # Proxy: weighted average of per-stock Sharpe
            agg_m = {
                "n_trades":      total_n,
                "mean_return":   round(wt_avg("mean_return"),   6),
                "std_return":    round(wt_avg("std_return"),    6),
                "sharpe_ratio":  round(wt_avg("sharpe_ratio"),  4),
                "median_return": round(wt_avg("median_return"), 6),
                "win_rate":      round(wt_avg("win_rate"),      4),
                "avg_win":       round(wt_avg("avg_win"),       6),
                "avg_loss":      round(wt_avg("avg_loss"),      6),
                "profit_factor": round(wt_avg("profit_factor"), 4),
                "rr":            round(wt_avg("rr"),            4),
                "avg_hold_days": round(wt_avg("avg_hold_days"), 2),
                "mae":           round(wt_avg("mae"),           6),
                "mfe":           round(wt_avg("mfe"),           6),
            }
            group_agg[sname] = agg_m

        all_results[gname] = group_agg

        # Aggregate indicator candidate results
        group_ind_agg = {}
        for ind_name, data in group_ind_stats.items():
            fr = np.array(data["fav_rets"])
            ur = np.array(data["unfav_rets"])
            def sh_from_rets(r):
                if len(r) < 5: return np.nan
                std = r.std(ddof=1)
                if std < 1e-12: return 0.0
                return float(r.mean() / std * np.sqrt(252/10))

            fav_sh  = sh_from_rets(fr)
            unfav_sh = sh_from_rets(ur)
            delta   = fav_sh - unfav_sh if not np.isnan(fav_sh) and not np.isnan(unfav_sh) else np.nan
            group_ind_agg[ind_name] = {
                "n_fav":       int(len(fr)),
                "n_unfav":     int(len(ur)),
                "sharpe_fav":  round(fav_sh, 4) if not np.isnan(fav_sh) else None,
                "sharpe_unfav": round(unfav_sh, 4) if not np.isnan(unfav_sh) else None,
                "delta_sharpe": round(delta, 4) if not np.isnan(delta) else None,
            }

        all_ind_results[gname] = group_ind_agg

    # ── Compute overall (weighted average across groups) ─────────────────────
    overall = {}
    for sname in EXIT_STRATEGIES:
        parts = []
        for gname in SIZE_GROUPS:
            if sname in all_results.get(gname, {}) and all_results[gname][sname].get("n_trades", 0) > 0:
                parts.append((all_results[gname][sname], all_results[gname][sname]["n_trades"]))
        if not parts:
            overall[sname] = {"n_trades": 0}
            continue
        total_n = sum(p[1] for p in parts)
        def wt_ov(key):
            vals = [p[0].get(key, np.nan) for p in parts]
            wts  = [p[1] for p in parts]
            pairs = [(v, w) for v, w in zip(vals, wts) if not (v is None or np.isnan(v)) and w > 0]
            if not pairs: return np.nan
            return float(np.average([p[0] for p in pairs], weights=[p[1] for p in pairs]))
        overall[sname] = {
            "n_trades":      total_n,
            "sharpe_ratio":  round(wt_ov("sharpe_ratio"),  4),
            "mean_return":   round(wt_ov("mean_return"),   6),
            "win_rate":      round(wt_ov("win_rate"),      4),
            "profit_factor": round(wt_ov("profit_factor"), 4),
            "rr":            round(wt_ov("rr"),            4),
            "avg_hold_days": round(wt_ov("avg_hold_days"), 2),
            "avg_win":       round(wt_ov("avg_win"),       6),
            "avg_loss":      round(wt_ov("avg_loss"),      6),
            "mae":           round(wt_ov("mae"),           6),
            "mfe":           round(wt_ov("mfe"),           6),
        }

    # ── PRINT RESULTS ─────────────────────────────────────────────────────────

    print("\n\n" + "=" * 72)
    print("  SELL TARGET BACKTEST RESULTS")
    print("=" * 72)

    # Layer 1: Rank all strategies by Sharpe
    def rank_by(metric, reverse=True):
        ranked = sorted(
            [(s, overall[s].get(metric, np.nan)) for s in EXIT_STRATEGIES
             if overall[s].get("n_trades", 0) > 0],
            key=lambda x: x[1] if not np.isnan(x[1]) else -999,
            reverse=reverse
        )
        return ranked

    print("\n--- TOP EXIT STRATEGIES BY SHARPE (OVERALL) ---")
    print(f"{'Rank':<5} {'Strategy':<35} {'Large':>7} {'Mid':>7} {'Small':>7} {'Overall':>8}")
    print("-" * 72)
    ranked_sharpe = rank_by("sharpe_ratio")
    for rank, (sname, ov_sh) in enumerate(ranked_sharpe[:15], 1):
        lg = all_results["large"].get(sname, {}).get("sharpe_ratio", np.nan)
        md = all_results["mid"].get(sname, {}).get("sharpe_ratio", np.nan)
        sm = all_results["small"].get(sname, {}).get("sharpe_ratio", np.nan)
        def fmt(v): return f"{v:7.3f}" if not np.isnan(v) else "    ---"
        print(f"{rank:<5} {sname:<35} {fmt(lg)} {fmt(md)} {fmt(sm)} {fmt(ov_sh)}")

    print("\n--- TOP EXIT STRATEGIES BY PROFIT FACTOR ---")
    print(f"{'Rank':<5} {'Strategy':<35} {'Large':>7} {'Mid':>7} {'Small':>7} {'Overall':>8}")
    print("-" * 72)
    ranked_pf = rank_by("profit_factor")
    for rank, (sname, ov_pf) in enumerate(ranked_pf[:10], 1):
        lg = all_results["large"].get(sname, {}).get("profit_factor", np.nan)
        md = all_results["mid"].get(sname, {}).get("profit_factor", np.nan)
        sm = all_results["small"].get(sname, {}).get("profit_factor", np.nan)
        def fmt(v): return f"{v:7.3f}" if not np.isnan(v) else "    ---"
        print(f"{rank:<5} {sname:<35} {fmt(lg)} {fmt(md)} {fmt(sm)} {fmt(ov_pf)}")

    print("\n--- TOP EXIT STRATEGIES BY WIN RATE ---")
    print(f"{'Rank':<5} {'Strategy':<35} {'Large':>7} {'Mid':>7} {'Small':>7} {'Overall':>8}")
    print("-" * 72)
    ranked_wr = rank_by("win_rate")
    for rank, (sname, ov_wr) in enumerate(ranked_wr[:10], 1):
        lg = all_results["large"].get(sname, {}).get("win_rate", np.nan)
        md = all_results["mid"].get(sname, {}).get("win_rate", np.nan)
        sm = all_results["small"].get(sname, {}).get("win_rate", np.nan)
        def fmt(v): return f"{v:7.1%}" if not np.isnan(v) else "    ---"
        print(f"{rank:<5} {sname:<35} {fmt(lg)} {fmt(md)} {fmt(sm)} {fmt(ov_wr)}")

    print("\n--- TOP EXIT STRATEGIES BY R:R (avg_win / abs(avg_loss)) ---")
    print(f"{'Rank':<5} {'Strategy':<35} {'Large':>7} {'Mid':>7} {'Small':>7} {'Overall':>8}")
    print("-" * 72)
    ranked_rr = rank_by("rr")
    for rank, (sname, ov_rr) in enumerate(ranked_rr[:10], 1):
        lg = all_results["large"].get(sname, {}).get("rr", np.nan)
        md = all_results["mid"].get(sname, {}).get("rr", np.nan)
        sm = all_results["small"].get(sname, {}).get("rr", np.nan)
        def fmt(v): return f"{v:7.2f}" if not np.isnan(v) and v < 99 else "  99.00" if not np.isnan(v) else "    ---"
        print(f"{rank:<5} {sname:<35} {fmt(lg)} {fmt(md)} {fmt(sm)} {fmt(ov_rr)}")

    # Layer 2: ATR multiple analysis
    print("\n--- OPTIMAL ATR MULTIPLE BY SIZE (Sharpe) ---")
    print(f"{'Multiple':<12} {'Large':>8} {'Mid':>8} {'Small':>8} {'Overall':>9}")
    print("-" * 50)
    for mul in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        key = f"ATR_{mul}x".replace(".0x", "x")
        lg = all_results["large"].get(key, {}).get("sharpe_ratio", np.nan)
        md = all_results["mid"].get(key, {}).get("sharpe_ratio", np.nan)
        sm = all_results["small"].get(key, {}).get("sharpe_ratio", np.nan)
        ov = overall.get(key, {}).get("sharpe_ratio", np.nan)
        def fmt(v): return f"{v:8.3f}" if not np.isnan(v) else "     ---"
        print(f"{mul:<12.1f} {fmt(lg)} {fmt(md)} {fmt(sm)} {fmt(ov)}")

    # Layer 3: BB sigma analysis
    print("\n--- OPTIMAL BB SIGMA BY SIZE (Sharpe) ---")
    print(f"{'Sigma':<12} {'Large':>8} {'Mid':>8} {'Small':>8} {'Overall':>9}")
    print("-" * 50)
    for sig in [1.0, 1.5, 2.0, 2.5, 3.0]:
        key = f"BB_{sig}sigma"
        lg = all_results["large"].get(key, {}).get("sharpe_ratio", np.nan)
        md = all_results["mid"].get(key, {}).get("sharpe_ratio", np.nan)
        sm = all_results["small"].get(key, {}).get("sharpe_ratio", np.nan)
        ov = overall.get(key, {}).get("sharpe_ratio", np.nan)
        def fmt(v): return f"{v:8.3f}" if not np.isnan(v) else "     ---"
        print(f"{sig:<12.1f} {fmt(lg)} {fmt(md)} {fmt(sm)} {fmt(ov)}")

    # Layer 4: Stop-loss impact
    print("\n--- STOP LOSS IMPACT ANALYSIS ---")
    # Compare top target strategies with/without stops
    top_targets = ["ATR_2x", "PCT_10", "BB_2.0sigma"]
    for tgt_base in top_targets:
        # Find matching strategies
        no_stop_key = None
        if "ATR" in tgt_base:
            mul_s = tgt_base.split("_")[1].replace("x","")
            no_stop_key = f"ATR_{mul_s}x"
        elif "PCT" in tgt_base:
            pct_s = tgt_base.split("_")[1]
            no_stop_key = f"PCT_{pct_s}"
        elif "BB" in tgt_base:
            no_stop_key = tgt_base

        print(f"\n  Target: {no_stop_key}")
        ns = overall.get(no_stop_key, {})
        print(f"  {'Stop Level':<30} {'Sharpe':>7} {'PF':>6} {'WR%':>6} {'R:R':>6}")
        print(f"  {'-'*55}")
        print(f"  {'No stop':<30} {ns.get('sharpe_ratio',np.nan):7.3f} "
              f"{ns.get('profit_factor',np.nan):6.3f} "
              f"{ns.get('win_rate',np.nan):6.1%} "
              f"{ns.get('rr',np.nan):6.2f}")

        # Check stop variants
        stop_variants = [
            (f"ATR_2x_tgt__ATR_1x_stp",   "ATR_1x stop"),
            (f"ATR_2x_tgt__ATR_0_5x_stp", "ATR_0.5x stop"),
            (f"PCT_10_tgt__PCT_5_stp",     "PCT_5 stop"),
            (f"PCT_10_tgt__PCT_3_stp",     "PCT_3 stop"),
            (f"BB_2s_tgt__ATR_1x_stp",    "ATR_1x stop"),
            (f"BB_2s_tgt__PCT_5_stp",     "PCT_5 stop"),
        ]
        printed = set()
        for sv_key, sv_label in stop_variants:
            if ("ATR" in tgt_base and "ATR" in sv_key and "tgt" in sv_key) or \
               ("PCT_10" in tgt_base and "PCT_10" in sv_key) or \
               ("BB" in tgt_base and "BB_2s" in sv_key):
                if sv_key not in printed and sv_key in overall:
                    sv = overall[sv_key]
                    print(f"  {sv_label:<30} {sv.get('sharpe_ratio',np.nan):7.3f} "
                          f"{sv.get('profit_factor',np.nan):6.3f} "
                          f"{sv.get('win_rate',np.nan):6.1%} "
                          f"{sv.get('rr',np.nan):6.2f}")
                    printed.add(sv_key)

    # Layer 5: Indicator candidates
    print("\n--- NEW INDICATOR CANDIDATES (Layer 5) ---")
    print(f"{'Indicator':<25} {'ΔSharpe(10d)':>12} {'n_fav':>7} {'n_unfav':>8} {'Recommendation'}")
    print("-" * 72)
    for gname in SIZE_GROUPS:
        print(f"  [{gname.upper()}]")
        for ind_name, ir in all_ind_results.get(gname, {}).items():
            delta  = ir.get("delta_sharpe")
            nf     = ir.get("n_fav", 0)
            nu     = ir.get("n_unfav", 0)
            delta_str = f"{delta:+.4f}" if delta is not None else "   N/A"
            rec = "ADD" if (delta is not None and delta > 0.05) else \
                  "WEAK" if (delta is not None and delta > 0.0) else "SKIP"
            print(f"  {ind_name:<23} {delta_str:>12} {nf:>7} {nu:>8}   {rec}")

    # Layer 6: Swing vs Day trade
    print("\n--- SWING vs DAY TRADE EXIT ANALYSIS ---")
    print("\n  SWING MODE (5-20 day holds):")
    swing_strats = [(s, overall[s]) for s in EXIT_STRATEGIES
                    if overall[s].get("avg_hold_days", 0) >= 4
                    and overall[s].get("n_trades", 0) > 0]
    swing_ranked = sorted(swing_strats, key=lambda x: x[1].get("sharpe_ratio", -999), reverse=True)
    print(f"  {'Rank':<5} {'Strategy':<35} {'Sharpe':>7} {'PF':>6} {'WR%':>6} {'Hold':>6}")
    print(f"  {'-'*68}")
    for rank, (sname, sm) in enumerate(swing_ranked[:10], 1):
        print(f"  {rank:<5} {sname:<35} {sm.get('sharpe_ratio',np.nan):7.3f} "
              f"{sm.get('profit_factor',np.nan):6.3f} "
              f"{sm.get('win_rate',np.nan):6.1%} "
              f"{sm.get('avg_hold_days',np.nan):6.1f}d")

    print("\n  DAY TRADE MODE (1-3 day holds):")
    day_strats = [(s, overall[s]) for s in EXIT_STRATEGIES
                  if overall[s].get("avg_hold_days", 999) <= 5
                  and overall[s].get("n_trades", 0) > 0]
    day_ranked = sorted(day_strats, key=lambda x: x[1].get("sharpe_ratio", -999), reverse=True)
    print(f"  {'Rank':<5} {'Strategy':<35} {'Sharpe':>7} {'PF':>6} {'WR%':>6} {'Hold':>6}")
    print(f"  {'-'*68}")
    for rank, (sname, dm) in enumerate(day_ranked[:10], 1):
        print(f"  {rank:<5} {sname:<35} {dm.get('sharpe_ratio',np.nan):7.3f} "
              f"{dm.get('profit_factor',np.nan):6.3f} "
              f"{dm.get('win_rate',np.nan):6.1%} "
              f"{dm.get('avg_hold_days',np.nan):6.1f}d")

    # ── Find optimal strategies for final recommendation ──────────────────────
    best_swing = swing_ranked[0] if swing_ranked else ("none", {})
    best_day   = day_ranked[0]   if day_ranked   else ("none", {})

    # Best overall with stop
    stop_strats = [(s, overall[s]) for s in EXIT_STRATEGIES
                   if "stp" in s and overall[s].get("n_trades", 0) > 0]
    stop_ranked = sorted(stop_strats, key=lambda x: x[1].get("sharpe_ratio", -999), reverse=True)
    best_stop = stop_ranked[0] if stop_ranked else ("none", {})

    # Indicator recommendations (average delta across groups)
    ind_avg_delta = {}
    for ind_name in ["ATR_norm_low","Volume_surge","Far_from_52w_high","Stoch_oversold",
                     "Williams_R_oversold","MFI_oversold","OBV_rising","VWAP_below","ADX_strong"]:
        deltas = [all_ind_results[g].get(ind_name, {}).get("delta_sharpe")
                  for g in SIZE_GROUPS if all_ind_results.get(g, {}).get(ind_name)]
        valid  = [d for d in deltas if d is not None]
        ind_avg_delta[ind_name] = float(np.mean(valid)) if valid else 0.0

    ind_sorted = sorted(ind_avg_delta.items(), key=lambda x: x[1], reverse=True)
    add_inds   = [n for n, d in ind_sorted if d > 0.02]

    print("\n\n" + "=" * 72)
    print("  FINAL RECOMMENDATION")
    print("=" * 72)
    print(f"\n  Swing exit (5-20d): {best_swing[0]}")
    if best_swing[1]:
        s = best_swing[1]
        print(f"    Sharpe={s.get('sharpe_ratio',0):.3f}  PF={s.get('profit_factor',0):.3f}  "
              f"WR={s.get('win_rate',0):.1%}  Avg hold={s.get('avg_hold_days',0):.1f}d")

    print(f"\n  Day trade exit (1-3d): {best_day[0]}")
    if best_day[1]:
        d = best_day[1]
        print(f"    Sharpe={d.get('sharpe_ratio',0):.3f}  PF={d.get('profit_factor',0):.3f}  "
              f"WR={d.get('win_rate',0):.1%}  Avg hold={d.get('avg_hold_days',0):.1f}d")

    print(f"\n  Best risk-managed (with stop): {best_stop[0]}")
    if best_stop[1]:
        ss = best_stop[1]
        print(f"    Sharpe={ss.get('sharpe_ratio',0):.3f}  PF={ss.get('profit_factor',0):.3f}  "
              f"WR={ss.get('win_rate',0):.1%}  R:R={ss.get('rr',0):.2f}")

    print(f"\n  New indicators to add to buy signal (avg ΔSharpe > 0.02):")
    for nm, dv in ind_sorted:
        rec = "ADD" if dv > 0.02 else "SKIP"
        print(f"    {nm:<25}  ΔSharpe={dv:+.4f}  [{rec}]")

    print(f"\n  Total runtime: {time.time() - t_global:.1f}s")

    # ── Save to JSON ─────────────────────────────────────────────────────────
    output = {
        "meta": {
            "n_strategies": len(EXIT_STRATEGIES),
            "n_stocks": len(STOCKS),
            "n_paths_per_stock": N_PATHS,
            "n_days": N_DAYS,
            "max_hold": MAX_HOLD,
            "runtime_seconds": round(time.time() - t_global, 1),
        },
        "strategies": list(EXIT_STRATEGIES.keys()),
        "per_group": all_results,
        "overall": overall,
        "indicator_candidates": all_ind_results,
        "recommendations": {
            "swing_exit": best_swing[0],
            "swing_exit_metrics": best_swing[1],
            "day_trade_exit": best_day[0],
            "day_trade_exit_metrics": best_day[1],
            "best_risk_managed": best_stop[0],
            "best_risk_managed_metrics": best_stop[1],
            "new_indicators": [{"name": n, "avg_delta_sharpe": round(d, 4)}
                               for n, d in ind_sorted],
        }
    }

    out_path = Path(__file__).parent / "sell_target_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Saved → {out_path}")
    return output


if __name__ == "__main__":
    main()
