#!/usr/bin/env python3
"""
6-Indicator Optimization — score52wHigh + scoreOBV bonus analysis
=================================================================

Extends the existing 4-indicator (RSI/MACD/BB/MA) backtest by adding:
  - score52wHigh  (0-12 pts): distance below 52-week high
  - scoreOBV      (0-8 pts):  OBV trend normalization

Formula:  total = min(100, base_4_score + bonus_52w + bonus_obv)

Also performs full 6-indicator weight optimization to see if it beats bonus approach.
"""

import json, time, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

# ─── Config (mirrors run_backtest.py) ─────────────────────────────────────────

STOCKS = {
    "NVDA":   (0.55, 0.60), "AMD":    (0.35, 0.62), "ASML":   (0.30, 0.28),
    "LRCX":   (0.28, 0.30), "KLAC":   (0.25, 0.30), "AMAT":   (0.22, 0.32),
    "TER":    (0.18, 0.35), "8035.T": (0.28, 0.42), "6857.T": (0.25, 0.38),
    "6920.T": (0.30, 0.55), "TSM":    (0.18, 0.30), "AVGO":   (0.25, 0.35),
    "QCOM":   (0.12, 0.30), "INTC":   (0.02, 0.32), "MRVL":   (0.30, 0.45),
    "ARM":    (0.25, 0.55), "CDNS":   (0.22, 0.28), "SNPS":   (0.25, 0.28),
    "AMBA":   (0.12, 0.55), "4063.T": (0.15, 0.25), "6762.T": (0.12, 0.35),
    "6503.T": (0.10, 0.25), "WOLF":   (0.08, 0.65), "ONTO":   (0.20, 0.45),
    "MU":     (0.22, 0.48), "WDC":    (0.10, 0.45), "MSFT":   (0.32, 0.22),
    "GOOGL":  (0.22, 0.25), "META":   (0.22, 0.40), "AMZN":   (0.20, 0.28),
    "ORCL":   (0.18, 0.25), "CRM":    (0.22, 0.30), "CRWD":   (0.30, 0.45),
    "SMCI":   (0.35, 0.80), "SNOW":   (0.25, 0.60), "PLTR":   (0.18, 0.70),
    "PATH":   (0.15, 0.55), "SOUN":   (0.20, 1.00), "BBAI":   (0.10, 1.00),
    "IREN":   (0.20, 1.00), "AI":     (0.05, 0.85), "6501.T": (0.20, 0.25),
    "6702.T": (0.18, 0.28), "IONQ":   (0.15, 0.90), "RGTI":   (0.10, 1.20),
    "QBTS":   (0.10, 1.00), "IBM":    (0.05, 0.22), "OKLO":   (0.15, 0.80),
    "SMR":    (0.15, 0.75), "BWXT":   (0.15, 0.25),
}

TRANS = np.array([[0.97, 0.015, 0.015],
                  [0.05, 0.93,  0.02 ],
                  [0.03, 0.01,  0.96 ]])
D_MUL = np.array([1.8, -2.5,  0.1])
V_MUL = np.array([0.9,  1.8,  0.7])

N_PATHS  = 10          # per stock (faster than original 15)
N_DAYS   = 2520        # 10 years
WARMUP   = 60

THRESHOLDS = [55, 60, 65, 70]
HOLD_DAYS  = [1, 3, 5, 7, 10]
FOLDS      = [(504, 504), (756, 504), (1008, 504), (1260, 504), (1512, 504)]

SIZE_GROUPS: dict[str, list[str]] = {
    "large": ["NVDA","AMD","ASML","TSM","AVGO","QCOM","INTC","MSFT","GOOGL","META",
              "AMZN","ORCL","CRM","IBM","MU","LRCX","KLAC","AMAT","CDNS","SNPS",
              "8035.T","4063.T","6501.T"],
    "mid":   ["MRVL","ARM","SMCI","TER","CRWD","SNOW","PLTR","WDC","PATH","6857.T",
              "6762.T","6702.T","6920.T","6503.T","BWXT","ONTO","WOLF","AMBA"],
    "small": ["SOUN","BBAI","IREN","AI","IONQ","RGTI","QBTS","OKLO","SMR"],
}

# Existing optimal MULTS from backtest_results.json
EXISTING_MULTS = {
    "large": {"rsi": 0.722, "macd": 1.274, "bb": 0.795, "ma": 1.209},
    "mid":   {"rsi": 0.842, "macd": 1.021, "bb": 0.956, "ma": 1.182},
    "small": {"rsi": 1.086, "macd": 0.913, "bb": 1.001, "ma": 1.000},
}

# ─── Simulation ───────────────────────────────────────────────────────────────

def sim_batch(cagr: float, vol: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (prices, volumes) both shape (N_PATHS, N_DAYS).
    Volume = abs(normal(1_000_000, 300_000)), up-day gets positive sign internally.
    """
    dd = np.log(1 + cagr) / 252
    dv = vol / np.sqrt(252)
    z  = RNG.standard_t(df=5, size=(N_PATHS, N_DAYS)).astype(float) / np.sqrt(5 / 3)
    prices  = np.empty((N_PATHS, N_DAYS))
    volumes = np.empty((N_PATHS, N_DAYS))

    # raw volume draws (always positive)
    raw_vol = np.abs(RNG.normal(1_000_000, 300_000, size=(N_PATHS, N_DAYS)))
    raw_vol = np.maximum(raw_vol, 1.0)

    for p in range(N_PATHS):
        reg = 0
        lr  = np.empty(N_DAYS)
        for t in range(N_DAYS):
            reg   = int(RNG.choice(3, p=TRANS[reg]))
            lr[t] = dd * D_MUL[reg] + dv * V_MUL[reg] * z[p, t]
        prices[p, 0]  = 100.0
        prices[p, 1:] = 100.0 * np.exp(np.cumsum(lr)[:-1])

        # volume signed by price direction (for OBV signal)
        volumes[p, 0] = raw_vol[p, 0]
        for t in range(1, N_DAYS):
            if prices[p, t] > prices[p, t - 1]:
                volumes[p, t] = raw_vol[p, t]
            else:
                volumes[p, t] = -raw_vol[p, t]

    return prices, volumes

# ─── Helpers from run_backtest.py ─────────────────────────────────────────────

def ema_v(p: np.ndarray, k: int) -> np.ndarray:
    T   = p.shape[-1]
    out = np.full_like(p, np.nan)
    if T < k:
        return out
    a = 2.0 / (k + 1)
    if not np.any(np.isnan(p)):
        out[..., k - 1] = p[..., :k].mean(-1)
        for t in range(k, T):
            out[..., t] = p[..., t] * a + out[..., t - 1] * (1 - a)
        return out
    P = p.shape[0]
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
            out[pi, t] = row[t] * a + out[pi, t - 1] * (1 - a)
    return out

def rsi_v(p: np.ndarray, n: int = 14) -> np.ndarray:
    P, T = p.shape
    out  = np.full((P, T), np.nan)
    if T <= n:
        return out
    d  = np.diff(p, axis=1)
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

def bb_v(p: np.ndarray, n: int = 20):
    P, T  = p.shape
    pb    = np.full((P, T), np.nan)
    bw    = np.full((P, T), np.nan)
    for t in range(n - 1, T):
        w  = p[:, t - n + 1:t + 1]
        m  = w.mean(1)
        s  = w.std(1, ddof=1)
        ok = s > 0
        pb[ok, t] = (p[ok, t] - (m[ok] - 2 * s[ok])) / (4 * s[ok])
        bw[ok, t] = 4 * s[ok] / m[ok]
    return pb, bw

def _ip(v: float, bp) -> float:
    if v <= bp[0][0]:
        return bp[0][1]
    if v >= bp[-1][0]:
        return bp[-1][1]
    for i in range(len(bp) - 1):
        x0, y0 = bp[i]
        x1, y1 = bp[i + 1]
        if x0 <= v <= x1:
            return y0 + (y1 - y0) * (v - x0) / (x1 - x0)
    return bp[-1][1]

# ─── 4-indicator score functions ──────────────────────────────────────────────

def sc_rsi(p: np.ndarray) -> np.ndarray:
    rv      = rsi_v(p)
    P, T    = p.shape
    out     = np.full((P, T), np.nan)
    for t in range(1, T):
        c    = rv[:, t]
        prev = rv[:, t - 1]
        ok   = ~(np.isnan(c) | np.isnan(prev))
        if not ok.any():
            continue
        up = c > prev
        sc = np.full(P, np.nan)
        for i in np.where(ok)[0]:
            cv, uv = c[i], up[i]
            if   cv < 25: r = 23.0
            elif cv < 35: r = min(_ip(cv, [(25, 22), (35, 12)]) + (3 if uv else 0), 25.0)
            elif cv < 45: r = _ip(cv, [(35, 14), (45, 10)])
            elif cv < 55: r = _ip(cv, [(45, 12), (55, 8)])
            elif cv < 65: r = _ip(cv, [(55, 8),  (65, 5)])
            elif cv < 75: r = _ip(cv, [(65, 5),  (75, 2)])
            else:         r = _ip(cv, [(75, 2),  (85, 0)])
            sc[i] = r
        out[:, t] = sc
    return out

def sc_macd(p: np.ndarray) -> np.ndarray:
    ml   = ema_v(p, 12) - ema_v(p, 26)
    sl   = ema_v(ml, 9)
    hl   = ml - sl
    P, T = p.shape
    out  = np.full((P, T), np.nan)
    for t in range(3, T):
        mc, mp = ml[:, t], ml[:, t - 1]
        scl, sp = sl[:, t], sl[:, t - 1]
        hc, hp  = hl[:, t], hl[:, t - 1]
        ok = ~(np.isnan(mc) | np.isnan(scl))
        if not ok.any():
            continue
        bn = (mp < sp) & (mc >= scl)
        dn = (mp > sp) & (mc <= scl)
        br = np.zeros(P, bool)
        for j in range(2, min(4, t)):
            br |= (~np.isnan(ml[:, t-j-1]) &
                   (ml[:, t-j-1] < sl[:, t-j-1]) &
                   (ml[:, t-j]   >= sl[:, t-j]) &
                   (mc > scl))
        s = np.full(P, 3.0)
        s[mc > scl]              = 10.0
        s[(mc > scl) & (hc > hp)] = 15.0
        s[br & (hc > 0)]         = 20.0
        s[bn]                    = 24.0
        s[(mc < scl) & (hc > hp)] = 6.0
        s[dn]                    = 2.0
        s[~ok]                   = np.nan
        out[:, t] = s
    return out

def sc_bb(p: np.ndarray) -> np.ndarray:
    pb, bw  = bb_v(p)
    P, T    = p.shape
    out     = np.full((P, T), np.nan)
    for t in range(4, T):
        ok = ~np.isnan(pb[:, t])
        if not ok.any():
            continue
        c     = pb[:, t]
        ws    = bw[:, max(0, t - 4):t + 1]
        cnt   = np.sum(~np.isnan(ws), axis=1)
        bwa   = np.where(np.isnan(ws), 0.0, ws)
        mean_bw = np.where(cnt > 0, bwa.sum(1) / np.maximum(cnt, 1), np.nan)
        sq    = ok & ~np.isnan(bw[:, t]) & (bw[:, t] < mean_bw * 0.85) & (c < 0.3)
        raw   = np.full(P, np.nan)
        for i in np.where(ok)[0]:
            cv = c[i]
            if   cv < 0.00: rv = _ip(cv, [(-0.2, 25), (0,    21)])
            elif cv < 0.10: rv = _ip(cv, [(0,    21), (0.1,  17)])
            elif cv < 0.25: rv = _ip(cv, [(0.1,  17), (0.25, 12)])
            elif cv < 0.50: rv = _ip(cv, [(0.25, 12), (0.5,   8)])
            elif cv < 0.75: rv = _ip(cv, [(0.5,   8), (0.75,  4)])
            elif cv < 0.90: rv = _ip(cv, [(0.75,  4), (0.9,   1)])
            else:            rv = _ip(cv, [(0.9,   1), (1.2,   0)])
            raw[i] = min(rv + (3 if sq[i] else 0), 25.0)
        out[:, t] = raw
    return out

def sc_ma(p: np.ndarray) -> np.ndarray:
    m20  = ema_v(p, 20)
    m50  = ema_v(p, 50)
    P, T = p.shape
    out  = np.full((P, T), np.nan)
    for t in range(3, T):
        ok = ~(np.isnan(m20[:, t]) | np.isnan(m50[:, t]))
        if not ok.any():
            continue
        pn, pp = p[:, t],    p[:, t - 1]
        mn, mp = m20[:, t],  m20[:, t - 1]
        fn     = m50[:, t]
        ct     = (pp < mp) & (pn >= mn)
        cr     = np.zeros(P, bool)
        for j in range(2, min(4, t)):
            cr |= (p[:, t-j-1] < m20[:, t-j-1]) & (p[:, t-j] >= m20[:, t-j]) & (pn > mn)
        gap   = mn - fn
        gprev = m20[:, t - 1] - m50[:, t - 1]
        wide  = np.abs(gap) > np.abs(gprev)
        sc    = np.full(P, np.nan)
        for i in np.where(ok)[0]:
            pv, mv, fv = pn[i], mn[i], fn[i]
            if   ct[i]: ps = 15
            elif cr[i]: ps = 12
            elif pv > mv:
                ps = round(_ip(pv / mv, [(1.0, 9), (1.05, 6), (1.1, 4)]))
            elif abs(pv - mv) / mv < 0.005:
                ps = 5
            else:
                ps = round(_ip(pv / mv, [(0.9, 4), (0.95, 3), (1.0, 0)]))
            gv = gap[i]
            ms = (9 if wide[i] else 6) if gv > 0 else \
                 (4 if abs(gv / fv) < 0.005 else (1 if wide[i] else 3))
            sc[i] = min(ps + ms, 25.0)
        out[:, t] = sc
    return out

# ─── New indicator: score52wHigh ──────────────────────────────────────────────

def sc_52w_high(p: np.ndarray) -> np.ndarray:
    """score52wHigh: 0-12 pts based on distance below 252-day rolling high.
    Shape: (P, T)
    """
    P, T = p.shape
    out  = np.full((P, T), np.nan)
    for t in range(252, T):
        high_252 = p[:, t - 252:t + 1].max(axis=1)  # includes today
        dist_pct = (high_252 - p[:, t]) / high_252 * 100.0
        sc = np.empty(P)
        for i in range(P):
            d = dist_pct[i]
            if   d <  5.0: sc[i] =  9.0
            elif d < 15.0: sc[i] = 12.0
            elif d < 25.0: sc[i] = 10.0
            elif d < 40.0: sc[i] =  7.0
            elif d < 60.0: sc[i] =  4.0
            else:          sc[i] =  1.0
        out[:, t] = sc
    return out

# ─── New indicator: scoreOBV ──────────────────────────────────────────────────

def sc_obv(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """scoreOBV: 0-8 pts based on OBV normalized by avg daily volume over 30d.
    v is signed volume (+ on up days, - on down days).
    Shape: (P, T)
    """
    P, T = p.shape
    out  = np.full((P, T), np.nan)

    # Compute cumulative OBV for each path
    obv = np.cumsum(v, axis=1)   # (P, T)

    for t in range(40, T):       # need at least 30d avg vol + 10d lookback
        # OBV change over last 10 days
        obv_change = obv[:, t] - obv[:, t - 10]

        # avg daily absolute volume over last 30 days
        avg_vol = np.abs(v[:, t - 30:t]).mean(axis=1)
        avg_vol = np.maximum(avg_vol, 1.0)

        obv_norm = obv_change / avg_vol

        sc = np.empty(P)
        for i in range(P):
            n = obv_norm[i]
            if   n >  1.5: sc[i] = 8.0
            elif n >  0.5: sc[i] = 6.0
            elif n >  0.0: sc[i] = 4.0
            elif n > -0.5: sc[i] = 3.0
            else:          sc[i] = 1.0
        out[:, t] = sc
    return out

# ─── Vectorized sc_52w and sc_obv (faster) ────────────────────────────────────

def sc_52w_high_vec(p: np.ndarray) -> np.ndarray:
    """Fully vectorized score52wHigh."""
    P, T = p.shape
    out  = np.full((P, T), np.nan)
    # rolling max using stride tricks-style cummax per column range
    for t in range(252, T):
        high_252 = p[:, t - 252:t + 1].max(axis=1)
        dist_pct = (high_252 - p[:, t]) / high_252 * 100.0
        sc = np.where(dist_pct <  5.0,  9.0,
             np.where(dist_pct < 15.0, 12.0,
             np.where(dist_pct < 25.0, 10.0,
             np.where(dist_pct < 40.0,  7.0,
             np.where(dist_pct < 60.0,  4.0,
                                         1.0)))))
        out[:, t] = sc
    return out

def sc_obv_vec(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fully vectorized scoreOBV."""
    P, T = p.shape
    out  = np.full((P, T), np.nan)
    obv  = np.cumsum(v, axis=1)

    for t in range(40, T):
        obv_change = obv[:, t] - obv[:, t - 10]
        avg_vol    = np.abs(v[:, t - 30:t]).mean(axis=1)
        avg_vol    = np.maximum(avg_vol, 1.0)
        n          = obv_change / avg_vol
        sc = np.where(n >  1.5, 8.0,
             np.where(n >  0.5, 6.0,
             np.where(n >  0.0, 4.0,
             np.where(n > -0.5, 3.0,
                                 1.0))))
        out[:, t] = sc
    return out

# ─── Composite score builders ─────────────────────────────────────────────────

def base4_composite(p: np.ndarray, mults: dict) -> np.ndarray:
    """Compute the 4-indicator composite using provided multipliers.
    Returns (P, T) array (can have NaN).
    """
    s_rsi  = sc_rsi(p)
    s_macd = sc_macd(p)
    s_bb   = sc_bb(p)
    s_ma   = sc_ma(p)

    m = np.array([mults["rsi"], mults["macd"], mults["bb"], mults["ma"]])
    # stack → (P, 4, T), then einsum
    sc_stack = np.stack([s_rsi, s_macd, s_bb, s_ma], axis=1)  # (P, 4, T)
    comp     = np.einsum('k,pkt->pt', m, sc_stack)             # (P, T)
    nan_mask = np.any(np.isnan(sc_stack), axis=1)
    comp[nan_mask] = np.nan
    return comp, sc_stack

def combined_composite(base_comp: np.ndarray, bonus_52w: np.ndarray,
                       bonus_obv: np.ndarray,
                       use_52w: bool = True, use_obv: bool = True) -> np.ndarray:
    """Add bonus indicators to base composite, capped at 100."""
    result = base_comp.copy()
    if use_52w:
        # add where both available
        mask = ~np.isnan(bonus_52w)
        result = np.where(mask, result + bonus_52w, result)
    if use_obv:
        mask = ~np.isnan(bonus_obv)
        result = np.where(mask, result + bonus_obv, result)
    return np.minimum(result, 100.0)

# ─── Backtest evaluation ──────────────────────────────────────────────────────

def eval_composite(comp: np.ndarray, price: np.ndarray,
                   thr: float, hold: int,
                   s_idx: int = 0, e_idx: int = None) -> dict:
    """Evaluate a composite signal array. Returns Sharpe, n, winrate."""
    P, T = price.shape
    if e_idx is None:
        e_idx = T
    t0 = max(s_idx, WARMUP)
    t1 = e_idx - hold
    if t0 >= t1:
        return {"sh": -99.0, "n": 0, "wr": 0.0}

    ep  = price[:, t0:t1]
    xp  = price[:, t0 + hold:t1 + hold]
    with np.errstate(invalid='ignore', divide='ignore'):
        fwd = np.where(ep > 0, (xp - ep) / ep, np.nan)

    c_sl = comp[:, t0:t1]
    sig  = (c_sl >= thr) & ~np.isnan(c_sl) & ~np.isnan(fwd)
    returns = fwd[sig]
    n = len(returns)
    if n < 10:
        return {"sh": -99.0, "n": n, "wr": 0.0}
    std = returns.std(ddof=1)
    if std < 1e-12:
        return {"sh": 0.0, "n": n, "wr": float((returns > 0).mean())}
    sh = float(returns.mean() / std * np.sqrt(252.0 / hold))
    return {"sh": sh, "n": n, "wr": float((returns > 0).mean())}

# ─── Grid search helpers ──────────────────────────────────────────────────────

def best_thr_hold(comp: np.ndarray, price: np.ndarray,
                  train_end: int) -> tuple[float, int, float]:
    """Return (best_thr, best_hold, best_sh) for training window."""
    best_sh, best_thr, best_hold = -999.0, 65, 5
    for hold in HOLD_DAYS:
        for thr in THRESHOLDS:
            r = eval_composite(comp, price, thr, hold, 0, train_end)
            if r["sh"] > best_sh:
                best_sh, best_thr, best_hold = r["sh"], thr, hold
    return best_thr, best_hold, best_sh

# ─── 6-indicator full weight grid (coarse for speed) ─────────────────────────

def weight_grid_6(step: int = 20):
    """6-dim weight grid summing to 100, step=20 → manageable size."""
    out = []
    vals = list(range(0, 101, step))
    for a in vals:
        for b in vals:
            if a + b > 100: break
            for c in vals:
                if a + b + c > 100: break
                for d in vals:
                    if a + b + c + d > 100: break
                    for e in vals:
                        if a + b + c + d + e > 100: break
                        f = 100 - a - b - c - d - e
                        if f >= 0:
                            out.append((a, b, c, d, e, f))
    return out

# ─── Normalisation for 6-indicator combo scores ──────────────────────────────

def compute_6ind_composite(sc6: np.ndarray, w: tuple) -> np.ndarray:
    """sc6: (P, 6, T). w: raw weights summing to 100.
    Scale each weight by indicator max (RSI/MACD/BB/MA → /25, H52 → /12, OBV → /8).
    Then sum. No hard cap here (cap applied later if needed).
    """
    # To make 6-ind comparable to 4-ind scale (0-100), we normalize so
    # that full weight on any one indicator gives 0-100 range:
    # base4 indicators: raw score 0-25, multiply by w/25 → contributes 0-w
    # H52: raw score 0-12, multiply by w/12 → contributes 0-w
    # OBV: raw score 0-8, multiply by w/8 → contributes 0-w
    # All weights sum to 100 → max total = 100
    scales = np.array([25.0, 25.0, 25.0, 25.0, 12.0, 8.0])
    wv = np.array(w, dtype=float) / scales          # scaled weights
    comp = np.einsum('k,pkt->pt', wv, sc6)
    nan_mask = np.any(np.isnan(sc6), axis=1)
    comp[nan_mask] = np.nan
    return comp

def optimise_fold_6ind(sc6: np.ndarray, price: np.ndarray,
                       train_end: int, wlist: list) -> dict:
    """Grid search over 6-indicator weight combos for training window."""
    P, T     = price.shape
    best     = {"sh": -999.0, "w": None}

    # Precompute forward returns for each hold period
    fwd_cache: dict = {}
    for hold in HOLD_DAYS:
        t0  = WARMUP
        t1  = train_end - hold
        fwd = np.full((P, T), np.nan)
        if t0 < t1:
            ep = price[:, t0:t1]
            xp = price[:, t0 + hold:t1 + hold]
            with np.errstate(invalid='ignore', divide='ignore'):
                fwd[:, t0:t1] = np.where(ep > 0, (xp - ep) / ep, np.nan)
        fwd_cache[hold] = (fwd[:, t0:t1].copy(), t0, t1)

    for w in wlist:
        comp = compute_6ind_composite(sc6, w)
        for hold in HOLD_DAYS:
            f_sl, t0, t1 = fwd_cache[hold]
            if t1 <= t0:
                continue
            c_sl    = comp[:, t0:t1]
            not_nan = ~np.isnan(c_sl) & ~np.isnan(f_sl)
            for thr in THRESHOLDS:
                sig = not_nan & (c_sl >= thr)
                n   = int(sig.sum())
                if n < 10:
                    continue
                returns = f_sl[sig]
                std     = float(returns.std(ddof=1))
                if std < 1e-12:
                    continue
                sh = float(returns.mean() / std * np.sqrt(252.0 / hold))
                if sh > best["sh"]:
                    best = {"sh": sh, "n": n, "wr": float((returns > 0).mean()),
                            "w": w, "thr": thr, "hold": hold}

    return best if best["w"] is not None else {}

# ─── Main per-group analysis ──────────────────────────────────────────────────

def run_group_analysis(gname: str, tickers: list) -> dict:
    t_start = time.time()
    print(f"\n  Simulating {gname.upper()} ({len(tickers)} stocks × {N_PATHS} paths)...")

    prices_list, volumes_list = [], []
    for ticker in tickers:
        if ticker not in STOCKS:
            continue
        cagr, vol = STOCKS[ticker]
        pm, vm = sim_batch(cagr, vol)
        prices_list.append(pm)
        volumes_list.append(vm)

    price_g  = np.concatenate(prices_list,  axis=0)
    volume_g = np.concatenate(volumes_list, axis=0)
    P_g = price_g.shape[0]
    print(f"    Series={P_g}  shape={price_g.shape}")

    # Compute 4-indicator scores
    print("    Computing 4-indicator scores...")
    mults4 = EXISTING_MULTS[gname]
    base_comp, sc_stack4 = base4_composite(price_g, mults4)

    # Compute new indicator scores
    print("    Computing score52wHigh...")
    s52w = sc_52w_high_vec(price_g)
    print("    Computing scoreOBV...")
    sobv = sc_obv_vec(price_g, volume_g)

    # Reference: evaluate base 4-indicator at fixed thr=65, hold=5
    REF_THR, REF_HOLD = 65, 5
    r_base = eval_composite(base_comp, price_g, REF_THR, REF_HOLD)
    print(f"    Base 4 only:  Sharpe={r_base['sh']:+.4f}  n={r_base['n']}")

    # +52wHigh bonus
    comp_52w = combined_composite(base_comp, s52w, sobv, use_52w=True,  use_obv=False)
    r_52w    = eval_composite(comp_52w, price_g, REF_THR, REF_HOLD)
    print(f"    +52wHigh:     Sharpe={r_52w['sh']:+.4f}  n={r_52w['n']}")

    # +OBV bonus
    comp_obv = combined_composite(base_comp, s52w, sobv, use_52w=False, use_obv=True)
    r_obv    = eval_composite(comp_obv, price_g, REF_THR, REF_HOLD)
    print(f"    +OBV:         Sharpe={r_obv['sh']:+.4f}  n={r_obv['n']}")

    # +Both bonuses
    comp_both = combined_composite(base_comp, s52w, sobv, use_52w=True, use_obv=True)
    r_both    = eval_composite(comp_both, price_g, REF_THR, REF_HOLD)
    print(f"    +Both:        Sharpe={r_both['sh']:+.4f}  n={r_both['n']}")

    # Walk-forward comparison (bonus approach)
    print("    Running walk-forward (bonus approach)...")
    fold_comparisons = []
    for fi, (train_days, test_days) in enumerate(FOLDS):
        test_end = train_days + test_days
        # find best thr+hold for base4 on training window
        best_thr, best_hold, _ = best_thr_hold(base_comp, price_g, train_days)
        # evaluate all 4 variants on test window using that thr+hold
        variants = {
            "base4":    eval_composite(base_comp,  price_g, best_thr, best_hold, train_days, test_end),
            "plus_52w": eval_composite(comp_52w,   price_g, best_thr, best_hold, train_days, test_end),
            "plus_obv": eval_composite(comp_obv,   price_g, best_thr, best_hold, train_days, test_end),
            "plus_both":eval_composite(comp_both,  price_g, best_thr, best_hold, train_days, test_end),
        }
        fold_comparisons.append({"fold": fi+1, "thr": best_thr, "hold": best_hold,
                                  "results": variants})

    # ── Full 6-indicator optimization ────────────────────────────────────────
    print("    Building 6-indicator stack...")
    # sc6 shape: (P, 6, T) = [rsi, macd, bb, ma, h52, obv]
    s_rsi  = sc_stack4[:, 0, :]
    s_macd = sc_stack4[:, 1, :]
    s_bb   = sc_stack4[:, 2, :]
    s_ma   = sc_stack4[:, 3, :]
    sc6 = np.stack([s_rsi, s_macd, s_bb, s_ma, s52w, sobv], axis=1)  # (P, 6, T)

    print("    Running 6-indicator walk-forward optimization...")
    wlist6 = weight_grid_6(step=20)
    print(f"    Weight grid: {len(wlist6)} combos")

    fold6_results = []
    for fi, (train_days, test_days) in enumerate(FOLDS):
        test_end = train_days + test_days
        best6 = optimise_fold_6ind(sc6, price_g, train_days, wlist6)
        if not best6:
            continue
        # evaluate on test window
        comp6_best = compute_6ind_composite(sc6, best6["w"])
        test_r = eval_composite(comp6_best, price_g, best6["thr"], best6["hold"],
                                 train_days, test_end)
        fold6_results.append({
            "fold": fi + 1,
            "train_sh": best6["sh"],
            "test_sh":  test_r["sh"],
            "w":        best6["w"],
            "thr":      best6["thr"],
            "hold":     best6["hold"],
        })

    # Best 6-indicator weights across folds (by train sharpe)
    best_fold6 = max(fold6_results, key=lambda x: x["train_sh"]) if fold6_results else None

    # Compute average test sharpes for comparison
    def avg_test_sh(fold_list, key="sh"):
        vals = [f["results"][key]["sh"] for f in fold_list
                if f["results"][key]["sh"] > -90]
        return float(np.mean(vals)) if vals else -99.0

    avg_base4     = avg_test_sh(fold_comparisons, "base4")
    avg_plus_52w  = avg_test_sh(fold_comparisons, "plus_52w")
    avg_plus_obv  = avg_test_sh(fold_comparisons, "plus_obv")
    avg_plus_both = avg_test_sh(fold_comparisons, "plus_both")
    avg_6ind_test = float(np.mean([f["test_sh"] for f in fold6_results
                                   if f["test_sh"] > -90])) if fold6_results else -99.0

    print(f"    WF avg test Sharpe → base4={avg_base4:.3f}  +52w={avg_plus_52w:.3f}  "
          f"+obv={avg_plus_obv:.3f}  +both={avg_plus_both:.3f}  6ind={avg_6ind_test:.3f}")
    print(f"    Elapsed: {time.time()-t_start:.0f}s")

    return {
        "ref_thr65_hold5": {
            "base4":     {"sh": round(r_base["sh"], 4), "n": r_base["n"]},
            "plus_52w":  {"sh": round(r_52w["sh"],  4), "n": r_52w["n"]},
            "plus_obv":  {"sh": round(r_obv["sh"],  4), "n": r_obv["n"]},
            "plus_both": {"sh": round(r_both["sh"], 4), "n": r_both["n"]},
        },
        "fold_avg_test_sharpe": {
            "base4":      round(avg_base4,     4),
            "plus_52w":   round(avg_plus_52w,  4),
            "plus_obv":   round(avg_plus_obv,  4),
            "plus_both":  round(avg_plus_both, 4),
            "6ind_full":  round(avg_6ind_test, 4),
        },
        "best_6ind_fold": {
            "fold": best_fold6["fold"]    if best_fold6 else None,
            "weights": {
                "rsi":  best_fold6["w"][0] if best_fold6 else None,
                "macd": best_fold6["w"][1] if best_fold6 else None,
                "bb":   best_fold6["w"][2] if best_fold6 else None,
                "ma":   best_fold6["w"][3] if best_fold6 else None,
                "h52":  best_fold6["w"][4] if best_fold6 else None,
                "obv":  best_fold6["w"][5] if best_fold6 else None,
            } if best_fold6 else None,
            "thr":       best_fold6["thr"]      if best_fold6 else None,
            "hold":      best_fold6["hold"]     if best_fold6 else None,
            "train_sh":  round(best_fold6["train_sh"], 4) if best_fold6 else None,
            "test_sh":   round(best_fold6["test_sh"],  4) if best_fold6 else None,
        },
        "all_fold6_results": fold6_results,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 68)
    print("  6-INDICATOR OPTIMIZATION (52wHigh + OBV bonus analysis)")
    print("=" * 68)
    print(f"  N_PATHS={N_PATHS}  N_DAYS={N_DAYS}  WARMUP={WARMUP}")
    print(f"  THRESHOLDS={THRESHOLDS}  HOLD_DAYS={HOLD_DAYS}")
    print(f"  FOLDS={len(FOLDS)}")

    all_results = {}
    for gname, tickers in SIZE_GROUPS.items():
        all_results[gname] = run_group_analysis(gname, tickers)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n")
    print("=" * 68)
    print("  === 6-INDICATOR OPTIMIZATION RESULTS ===")
    print("=" * 68)

    print("\n--- SHARPE COMPARISON: 4-indicator vs 6-indicator (swing, hold=5d, thr=65) ---")
    groups = ["large", "mid", "small"]
    hdr = f"{'':25s}  {'large':>8s}  {'mid':>8s}  {'small':>8s}"
    print(hdr)
    print("-" * 60)

    def fmt_row(label, key):
        vals = [all_results[g]["ref_thr65_hold5"][key]["sh"] for g in groups]
        return f"  {label:23s}  {vals[0]:>+8.3f}  {vals[1]:>+8.3f}  {vals[2]:>+8.3f}"

    base_vals = [all_results[g]["ref_thr65_hold5"]["base4"]["sh"] for g in groups]
    print(fmt_row("Base 4 only:", "base4"))

    for key, label in [("plus_52w",  "+52wHigh bonus:"),
                        ("plus_obv",  "+OBV bonus:"),
                        ("plus_both", "+Both bonuses:")]:
        vals = [all_results[g]["ref_thr65_hold5"][key]["sh"] for g in groups]
        deltas = [vals[i] - base_vals[i] for i in range(3)]
        delta_str = f"  (Δ {deltas[0]:+.3f} / {deltas[1]:+.3f} / {deltas[2]:+.3f})"
        print(f"  {label:23s}  {vals[0]:>+8.3f}  {vals[1]:>+8.3f}  {vals[2]:>+8.3f}{delta_str}")

    print("\n--- WALK-FORWARD AVG TEST SHARPE (bonus approach vs 6-ind full opt) ---")
    print(hdr)
    print("-" * 60)
    for key, label in [("base4",     "Base 4 (WF test avg):"),
                        ("plus_52w",  "+52wHigh (WF test):"),
                        ("plus_obv",  "+OBV (WF test):"),
                        ("plus_both", "+Both (WF test):"),
                        ("6ind_full", "6-ind full opt (WF):")]:
        vals = [all_results[g]["fold_avg_test_sharpe"][key] for g in groups]
        print(f"  {label:23s}  {vals[0]:>+8.3f}  {vals[1]:>+8.3f}  {vals[2]:>+8.3f}")

    print("\n--- OPTIMAL BONUS WEIGHTS (bonus approach) ---")
    print("  Using existing 4-indicator MULTS (sum~4.0 per group) as base.")
    print("  bonus_52w: 0-12 pts, bonus_obv: 0-8 pts  (score capped at 100)")
    for gname in groups:
        r = all_results[gname]
        best_bonus_key = max(["plus_52w", "plus_obv", "plus_both"],
                             key=lambda k: r["ref_thr65_hold5"][k]["sh"])
        best_bonus_sh  = r["ref_thr65_hold5"][best_bonus_key]["sh"]
        base_sh        = r["ref_thr65_hold5"]["base4"]["sh"]
        verdict = "BONUS WINS" if best_bonus_sh > base_sh else "NO IMPROVEMENT"
        print(f"  {gname:6s}: best={best_bonus_key}  "
              f"Sharpe={best_bonus_sh:+.4f} vs base={base_sh:+.4f}  [{verdict}]")

    print("\n--- FULL 6-INDICATOR MULTS (if optimizer finds them better) ---")
    for gname in groups:
        r = all_results[gname]["best_6ind_fold"]
        if r["weights"] is None:
            print(f"  {gname}: no valid 6-indicator solution found")
            continue
        w = r["weights"]
        base_ref  = all_results[gname]["ref_thr65_hold5"]["base4"]["sh"]
        bonus_ref = all_results[gname]["ref_thr65_hold5"]["plus_both"]["sh"]
        # compare 6-ind test sh vs bonus test sh (WF avg)
        six_test  = all_results[gname]["fold_avg_test_sharpe"]["6ind_full"]
        both_test = all_results[gname]["fold_avg_test_sharpe"]["plus_both"]
        verdict   = "6-IND WINS" if six_test > both_test else "BONUS WINS"
        print(f"  {gname:6s}: [RSI={w['rsi']}, MACD={w['macd']}, BB={w['bb']}, "
              f"MA={w['ma']}, H52={w['h52']}, OBV={w['obv']}]")
        print(f"          train_sh={r['train_sh']:+.4f}  test_sh={r['test_sh']:+.4f}  "
              f"WF-avg: 6ind={six_test:+.3f} vs bonus={both_test:+.3f}  [{verdict}]")

    print(f"\n  Total runtime: {time.time()-t0:.1f}s")
    print("=" * 68)

    # Save results
    out_path = Path(__file__).parent / "six_indicator_results.json"
    out_path.write_text(json.dumps({
        "method":    "6indicator_bonus_and_fullopt",
        "note":      "base4 MULTS from backtest_results.json; bonus: +52wHigh(0-12) +OBV(0-8)",
        "n_paths":   N_PATHS,
        "n_days":    N_DAYS,
        "thresholds": THRESHOLDS,
        "hold_days": HOLD_DAYS,
        "results":   all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out_path}")
    return all_results

if __name__ == "__main__":
    main()
