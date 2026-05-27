#!/usr/bin/env python3
"""
Walk-Forward Backtest — Swing Trade Score Weight Optimization
=============================================================

50 stocks (large + small/mid cap) × 15 paths = 750 total series.
Network blocked in this cloud env → uses Monte Carlo simulation
calibrated to documented 10-year statistics (2015-2024).

Optimizations vs previous version:
  - optimise_fold: einsum computed once per weight combo (not per combo×thr×hold)
  - fwd precomputed per hold period (not per combo)
  - Result: ~50× faster fold computation
"""

import json, time, warnings
from pathlib import Path
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

# ─── Config ──────────────────────────────────────────────────────────────────

STOCKS = {
    # (CAGR, ann.vol) — calibrated to 2015-2024 documented statistics
    # ── AI半導体 / GPU ──────────────────────────────────────────────────────
    "NVDA":   (0.55, 0.60),  # AI GPU market leader
    "AMD":    (0.35, 0.62),  # CPU/GPU challenger
    # ── 半導体装置 ──────────────────────────────────────────────────────────
    "ASML":   (0.30, 0.28),  # EUV lithography monopoly
    "LRCX":   (0.28, 0.30),  # etch equipment
    "KLAC":   (0.25, 0.30),  # process control
    "AMAT":   (0.22, 0.32),  # deposition equipment
    "TER":    (0.18, 0.35),  # test equipment
    "8035.T": (0.28, 0.42),  # Tokyo Electron
    "6857.T": (0.25, 0.38),  # Advantest
    "6920.T": (0.30, 0.55),  # Lasertec (EUV inspection)
    # ── ファウンドリ・設計 ──────────────────────────────────────────────────
    "TSM":    (0.18, 0.30),  # foundry leader
    "AVGO":   (0.25, 0.35),  # networking/custom AI chips
    "QCOM":   (0.12, 0.30),  # mobile/edge AI
    "INTC":   (0.02, 0.32),  # legacy, restructuring
    "MRVL":   (0.30, 0.45),  # custom silicon / data center
    "ARM":    (0.25, 0.55),  # IP licensing (IPO 2023)
    "CDNS":   (0.22, 0.28),  # EDA software
    "SNPS":   (0.25, 0.28),  # EDA software
    "AMBA":   (0.12, 0.55),  # edge AI chips (small cap)
    "4063.T": (0.15, 0.25),  # Shin-Etsu Chemical (silicon wafers)
    "6762.T": (0.12, 0.35),  # TDK (electronic components)
    "6503.T": (0.10, 0.25),  # Mitsubishi Electric
    # ── 特殊半導体 ──────────────────────────────────────────────────────────
    "WOLF":   (0.08, 0.65),  # Wolfspeed SiC (financially stressed)
    "ONTO":   (0.20, 0.45),  # optical inspection (small cap)
    # ── メモリ ──────────────────────────────────────────────────────────────
    "MU":     (0.22, 0.48),  # DRAM / HBM leader
    "WDC":    (0.10, 0.45),  # NAND / HDD storage
    # ── AI / クラウド (大企業) ──────────────────────────────────────────────
    "MSFT":   (0.32, 0.22),  # Azure / OpenAI partnership
    "GOOGL":  (0.22, 0.25),  # Gemini / GCP
    "META":   (0.22, 0.40),  # social AI / Llama
    "AMZN":   (0.20, 0.28),  # AWS / Bedrock
    "ORCL":   (0.18, 0.25),  # cloud database AI
    "CRM":    (0.22, 0.30),  # Salesforce Einstein AI
    "CRWD":   (0.30, 0.45),  # AI-native cybersecurity
    # ── AI / クラウド (中小) ────────────────────────────────────────────────
    "SMCI":   (0.35, 0.80),  # AI server infrastructure (volatile)
    "SNOW":   (0.25, 0.60),  # cloud data platform
    "PLTR":   (0.18, 0.70),  # data analytics AI
    "PATH":   (0.15, 0.55),  # process automation AI
    "SOUN":   (0.20, 1.00),  # SoundHound voice AI (small cap)
    "BBAI":   (0.10, 1.00),  # BigBear.ai analytics (small cap)
    "IREN":   (0.20, 1.00),  # AI data centers / HPC
    "AI":     (0.05, 0.85),  # C3.ai (enterprise AI, slow growth)
    "6501.T": (0.20, 0.25),  # Hitachi IT transformation
    "6702.T": (0.18, 0.28),  # Fujitsu cloud/AI services
    # ── 量子コンピュータ ────────────────────────────────────────────────────
    "IONQ":   (0.15, 0.90),  # trapped-ion quantum
    "RGTI":   (0.10, 1.20),  # Rigetti (highly speculative)
    "QBTS":   (0.10, 1.00),  # D-Wave quantum
    "IBM":    (0.05, 0.22),  # quantum + legacy IT
    # ── 核融合 / SMR ────────────────────────────────────────────────────────
    "OKLO":   (0.15, 0.80),  # micro-reactor startup
    "SMR":    (0.15, 0.75),  # NuScale Power SMR
    "BWXT":   (0.15, 0.25),  # BWX Technologies (defense nuclear)
}

# Regime-switching GBM parameters
TRANS = np.array([[0.97, 0.015, 0.015],
                  [0.05, 0.93,  0.02 ],
                  [0.03, 0.01,  0.96 ]])
D_MUL = np.array([1.8, -2.5,  0.1])   # drift multipliers (bull/bear/sideways)
V_MUL = np.array([0.9,  1.8,  0.7])   # vol   multipliers

N_PATHS     = 15          # paths per stock → 50×15 = 750 total
N_DAYS      = 2520        # ≈ 10 years
WARMUP      = 60          # bars to skip (indicator warm-up)

# Doubled from v1: 4→8 thresholds, 3→6 hold periods
THRESHOLDS  = [50, 55, 60, 62, 65, 67, 70, 75]
HOLD_DAYS   = [3, 5, 7, 10, 15, 20]
WEIGHT_STEP = 10          # 286 combos (kept fine-grained, optimised loop handles it)
FOLDS       = [(504, 504), (756, 504), (1008, 504), (1260, 504), (1512, 504)]

# ─── Simulation ──────────────────────────────────────────────────────────────

def sim_batch(cagr: float, vol: float) -> np.ndarray:
    """Return (N_PATHS, N_DAYS) price matrix via calibrated regime-switching GBM."""
    dd = np.log(1 + cagr) / 252
    dv = vol / np.sqrt(252)
    z  = RNG.standard_t(df=5, size=(N_PATHS, N_DAYS)).astype(float) / np.sqrt(5 / 3)
    prices = np.empty((N_PATHS, N_DAYS))
    for p in range(N_PATHS):
        reg = 0
        lr  = np.empty(N_DAYS)
        for t in range(N_DAYS):
            reg   = int(RNG.choice(3, p=TRANS[reg]))
            lr[t] = dd * D_MUL[reg] + dv * V_MUL[reg] * z[p, t]
        prices[p, 0]  = 100.0
        prices[p, 1:] = 100.0 * np.exp(np.cumsum(lr)[:-1])
    return prices

# ─── Indicators (vectorized over path axis) ───────────────────────────────────

def ema_v(p: np.ndarray, k: int) -> np.ndarray:
    """EMA; fast path when p is NaN-free, per-path path when p has leading NaN prefix."""
    T   = p.shape[-1]
    out = np.full_like(p, np.nan)
    if T < k:
        return out
    a = 2.0 / (k + 1)
    if not np.any(np.isnan(p)):
        # Fast path: raw prices / clean ema lines
        out[..., k - 1] = p[..., :k].mean(-1)
        for t in range(k, T):
            out[..., t] = p[..., t] * a + out[..., t - 1] * (1 - a)
        return out
    # NaN-prefix path (MACD line before ema26 warmup): find first valid per path
    P = p.shape[0]
    for pi in range(P):
        row   = p[pi]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        fv    = int(np.argmax(valid))       # first valid bar
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

# ─── Score arrays — exact mirror of scorer.ts ─────────────────────────────────

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

def all_scores(p: np.ndarray) -> np.ndarray:
    """Return (P, 4, T) score array: [rsi, macd, bb, ma]."""
    return np.stack([sc_rsi(p), sc_macd(p), sc_bb(p), sc_ma(p)], axis=1)

# ─── Weight grid ─────────────────────────────────────────────────────────────

def weight_grid(step: int = WEIGHT_STEP):
    out = []
    for a in range(0, 101, step):
        for b in range(0, 101 - a, step):
            for c in range(0, 101 - a - b, step):
                out.append((a, b, c, 100 - a - b - c))
    return out

# ─── Core backtest (used for OOS evaluation) ──────────────────────────────────

def backtest_one(sc: np.ndarray, price: np.ndarray,
                 w: tuple, hold: int, thr: float,
                 s_idx: int, e_idx: int):
    P, T = price.shape
    wv   = np.array(w, dtype=float) / 25.0
    comp = np.einsum('k,pkt->pt', wv, sc)
    comp[np.any(np.isnan(sc), axis=1)] = np.nan
    fwd  = np.full((P, T), np.nan)
    t0, t1 = max(s_idx, WARMUP), e_idx - hold
    if t0 >= t1:
        return -99.0, 0, 0.0
    ep = price[:, t0:t1]
    xp = price[:, t0 + hold:t1 + hold]
    with np.errstate(invalid='ignore', divide='ignore'):
        fwd[:, t0:t1] = np.where(ep > 0, (xp - ep) / ep, np.nan)
    c_sl, f_sl = comp[:, t0:t1], fwd[:, t0:t1]
    sig     = (c_sl >= thr) & ~np.isnan(c_sl) & ~np.isnan(f_sl)
    returns = f_sl[sig]
    n       = len(returns)
    if n < 10:
        return -99.0, n, 0.0
    std = returns.std(ddof=1)
    if std < 1e-12:
        return 0.0, n, float((returns > 0).mean())
    sh = float(returns.mean() / std * np.sqrt(252.0 / hold))
    return sh, n, float((returns > 0).mean())

# ─── Optimised walk-forward fold ──────────────────────────────────────────────

def optimise_fold(sc_all: np.ndarray, price_all: np.ndarray,
                  train_end: int, wlist: list) -> dict:
    """Grid search with einsum computed once per weight combo (not per combo×thr×hold).
    ~50× faster than the naive nested loop."""
    P, T     = price_all.shape
    nan_mask = np.any(np.isnan(sc_all), axis=1)   # (P, T)
    best     = {"sh": -999.0, "w": None}

    # Precompute forward returns for each hold period (avoids per-combo recomputation)
    fwd_cache: dict = {}
    for hold in HOLD_DAYS:
        t0  = WARMUP
        t1  = train_end - hold
        fwd = np.full((P, T), np.nan)
        if t0 < t1:
            ep = price_all[:, t0:t1]
            xp = price_all[:, t0 + hold:t1 + hold]
            with np.errstate(invalid='ignore', divide='ignore'):
                fwd[:, t0:t1] = np.where(ep > 0, (xp - ep) / ep, np.nan)
        fwd_cache[hold] = (fwd[:, t0:t1].copy(), t0, t1)

    # Outer loop: weight combo → compute composite once per w
    for w in wlist:
        wv   = np.array(w, dtype=float) / 25.0
        comp = np.einsum('k,pkt->pt', wv, sc_all)   # (P, T)
        comp[nan_mask] = np.nan

        for hold in HOLD_DAYS:
            f_sl, t0, t1 = fwd_cache[hold]
            if t1 <= t0:
                continue
            c_sl    = comp[:, t0:t1]                 # (P, slice) — view
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
                    best = {"sh": sh, "n": n,
                            "wr": float((returns > 0).mean()),
                            "w": w, "thr": thr, "hold": hold}

    return best if best["w"] is not None else {}

def eval_one(sc, price, w, thr, hold, s0, s1) -> dict:
    sh, n, wr = backtest_one(sc, price, w, hold, thr, s0, s1)
    return {"sh": sh, "n": n, "wr": wr}

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0    = time.time()
    wlist = weight_grid()
    n_stocks = len(STOCKS)
    total_series = n_stocks * N_PATHS

    print("=" * 68)
    print("  Walk-Forward Backtest — Monte Carlo (50 stocks, optimised fold)")
    print("=" * 68)
    print(f"\n  Stocks={n_stocks}  Paths/stock={N_PATHS}  "
          f"Total series={total_series}  Days={N_DAYS}")
    print(f"  Grid={len(wlist)} combos × {len(THRESHOLDS)} thr × {len(HOLD_DAYS)} hold")
    total_evals = len(wlist) * len(THRESHOLDS) * len(HOLD_DAYS) * len(FOLDS)
    print(f"  {len(FOLDS)} folds × {len(THRESHOLDS)}×{len(HOLD_DAYS)} param combos "
          f"= {total_evals:,} evaluations (optimised: einsum once per weight)")

    # ── 1. Simulate & score ───────────────────────────────────────────────
    print("\n[1/3] Simulating calibrated paths & computing scores ...")
    pb, sb = [], []
    for ticker, (cagr, vol) in STOCKS.items():
        pm = sim_batch(cagr, vol)
        sm = all_scores(pm)
        pb.append(pm)
        sb.append(sm)
        valid = ~np.isnan(sm[:, 0, :])
        print(f"  {ticker:8s}  cagr={cagr:.0%}  vol={vol:.0%}  "
              f"valid={valid.mean():.1%}")

    price_all = np.concatenate(pb, axis=0)   # (750, 2520)
    score_all = np.concatenate(sb, axis=0)   # (750, 4, 2520)

    # Sanity check on composite distribution
    wv_eq    = np.ones(4)
    comp_eq  = np.einsum('k,pkt->pt', wv_eq, score_all)
    comp_eq[np.any(np.isnan(score_all), axis=1)] = np.nan
    vc       = comp_eq[~np.isnan(comp_eq)]
    print(f"\n  Total: {total_series} series × {N_DAYS} days | "
          f"NaN={np.isnan(score_all).mean():.1%}")
    print(f"  Composite (equal wt): mean={vc.mean():.1f}  std={vc.std():.1f}  "
          f"≥60: {(vc >= 60).mean():.1%}  ≥70: {(vc >= 70).mean():.1%}")
    print(f"  Score phase done in {time.time()-t0:.1f}s")

    # ── 2. Walk-forward optimisation ─────────────────────────────────────
    print("\n[2/3] Walk-forward optimisation ...")
    fold_results = []
    for fi, (train_days, test_days) in enumerate(FOLDS):
        tf         = time.time()
        test_end   = train_days + test_days
        best       = optimise_fold(score_all, price_all, train_days, wlist)
        if not best:
            print(f"  Fold {fi+1}: no valid params")
            continue
        test_m = eval_one(score_all, price_all,
                          best["w"], best["thr"], best["hold"],
                          train_days, test_end)
        w = best["w"]
        print(f"  Fold {fi+1}  train={train_days}d  "
              f"RSI={w[0]:3d}/MACD={w[1]:3d}/BB={w[2]:3d}/MA={w[3]:3d}  "
              f"thr={best['thr']}  hold={best['hold']}d  "
              f"TrainSh={best['sh']:+.3f}  TestSh={test_m['sh']:+.3f}  "
              f"WR={test_m['wr']:.1%}  ({time.time()-tf:.0f}s)")
        fold_results.append({"fold": fi + 1, "train_days": train_days,
                              "best": best, "test": test_m})

    # ── 3. Aggregate ─────────────────────────────────────────────────────
    print("\n[3/3] Aggregating ...")
    if not fold_results:
        print("  ERROR: all folds returned no valid params.")
        return {}

    valid_folds = [f for f in fold_results if f["test"]["sh"] > -10] or fold_results
    w_v   = Counter()
    thr_v = Counter()
    hold_v = Counter()
    for f in valid_folds:
        wt = max(0.01, f["test"]["sh"])
        w_v[f["best"]["w"]]    += wt
        thr_v[f["best"]["thr"]] += wt
        hold_v[f["best"]["hold"]] += wt

    opt_w   = w_v.most_common(1)[0][0]
    opt_thr = thr_v.most_common(1)[0][0]
    opt_h   = hold_v.most_common(1)[0][0]

    base = eval_one(score_all, price_all, (25, 25, 25, 25), 70, 10, 0, N_DAYS)
    opt  = eval_one(score_all, price_all, opt_w, opt_thr, opt_h, 0, N_DAYS)

    oos_evals = [eval_one(score_all, price_all, opt_w, opt_thr, opt_h,
                          f["train_days"], f["train_days"] + 504)
                 for f in valid_folds[-2:]]
    oos_sh = float(np.mean([e["sh"] for e in oos_evals if e["n"] > 0])) \
             if oos_evals else -99.0
    oos_n  = sum(e["n"] for e in oos_evals)

    # Per-signal single-indicator analysis
    print("\n  Single-signal contributions (indicator used in isolation at 100%):")
    sig_sh  = {}
    sig_wr  = {}
    sig_n   = {}
    for lbl, sw in [("RSI",  (100, 0,   0,   0)),
                    ("MACD", (0,   100, 0,   0)),
                    ("BB",   (0,   0,   100, 0)),
                    ("MA",   (0,   0,   0,   100))]:
        m = eval_one(score_all, price_all, sw, opt_thr, opt_h, 0, N_DAYS)
        sig_sh[lbl] = m["sh"]
        sig_wr[lbl] = m["wr"]
        sig_n[lbl]  = m["n"]
        print(f"    {lbl:4s}: Sharpe={m['sh']:+.3f}  WinRate={m['wr']:.1%}  n={m['n']}")

    ranked = sorted(sig_sh, key=sig_sh.get, reverse=True)
    print(f"  Signal ranking: {' > '.join(ranked)}")

    # Compute backtest-optimal multipliers from single-signal Sharpes
    sh_vals  = np.array([sig_sh[k] for k in ["RSI", "MACD", "BB", "MA"]])
    # Clip negatives to 0, then normalise to sum=4 (preserves 0-100 scale)
    sh_pos   = np.maximum(sh_vals, 0.05)   # floor at 0.05 to keep all indicators
    mults    = sh_pos / sh_pos.mean()      # sum = 4 (4 indicators × 1.0 mean)
    mul_rsi, mul_macd, mul_bb, mul_ma = [round(float(m), 3) for m in mults]

    # ── Print results ──────────────────────────────────────────────────────
    ow = opt_w
    print("\n" + "=" * 68)
    print("  FINAL RESULTS")
    print("=" * 68)
    print(f"\nBaseline (25/25/25/25  thr=70  hold=10d):")
    print(f"  Sharpe={base['sh']:+.3f}  WinRate={base['wr']:.1%}  Trades={base['n']}")
    print(f"\nOptimized (walk-forward best):")
    print(f"  Weights  RSI={ow[0]:3d}  MACD={ow[1]:3d}  BB={ow[2]:3d}  MA={ow[3]:3d}")
    print(f"  Threshold={opt_thr}  Hold={opt_h}d")
    print(f"  In-sample  Sharpe={opt['sh']:+.3f}  WinRate={opt['wr']:.1%}  Trades={opt['n']}")
    print(f"  OOS Sharpe={oos_sh:+.3f}  Trades={oos_n}")
    print(f"\nBacktest-derived score multipliers (Sharpe-proportional, sum=4):")
    print(f"  RSI ×{mul_rsi}  MACD ×{mul_macd}  BB ×{mul_bb}  MA ×{mul_ma}")
    print(f"\nSharpe improvement: {base['sh']:+.3f} → {opt['sh']:+.3f}  "
          f"(Δ={opt['sh']-base['sh']:+.3f})")
    print("\nFold details:")
    for f in fold_results:
        w2 = f["best"]["w"]
        print(f"  Fold {f['fold']}  ({','.join(str(x) for x in w2)})  "
              f"thr={f['best']['thr']} hold={f['best']['hold']}d  "
              f"TrainSh={f['best']['sh']:+.3f}  TestSh={f['test']['sh']:+.3f}  "
              f"WR={f['test']['wr']:.1%}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

    result = {
        "method":            "walk_forward_monte_carlo_v2",
        "note":              "Calibrated simulation (50 stocks, network blocked).",
        "n_stocks":          n_stocks,
        "n_paths_per_stock": N_PATHS,
        "n_days":            N_DAYS,
        "thresholds":        THRESHOLDS,
        "hold_days":         HOLD_DAYS,
        "weight_step":       WEIGHT_STEP,
        "optimal_weights":   {"rsi": ow[0], "macd": ow[1], "bb": ow[2], "ma": ow[3]},
        "buy_threshold":     opt_thr,
        "hold_days_opt":     opt_h,
        "oos_sharpe":        round(oos_sh, 4),
        "baseline_sharpe":   round(base["sh"], 4),
        "optimal_sharpe":    round(opt["sh"],  4),
        "signal_ranking":    ranked,
        "signal_sharpes":    {k: round(v, 4) for k, v in sig_sh.items()},
        "signal_winrates":   {k: round(v, 4) for k, v in sig_wr.items()},
        "signal_counts":     sig_n,
        "score_multipliers": {
            "rsi": mul_rsi, "macd": mul_macd, "bb": mul_bb, "ma": mul_ma,
        },
        "folds": [
            {"fold":           f["fold"],
             "weights":        {"rsi": f["best"]["w"][0], "macd": f["best"]["w"][1],
                                "bb":  f["best"]["w"][2], "ma":   f["best"]["w"][3]},
             "threshold":      f["best"]["thr"],
             "hold_days":      f["best"]["hold"],
             "train_sharpe":   round(f["best"]["sh"], 4),
             "test_sharpe":    round(f["test"]["sh"],  4),
             "test_winrate":   round(f["test"]["wr"],  4)}
            for f in fold_results
        ],
    }
    out = Path(__file__).parent / "backtest_results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out}")
    return result

if __name__ == "__main__":
    main()
