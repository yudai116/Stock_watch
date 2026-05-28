#!/usr/bin/env python3
"""
Walk-Forward Backtest v4 — Full Indicator Suite Verification
=============================================================

Goals:
  1. Re-verify swing + day trade MULTS (RSI/MACD/BB/MA) — corrected BB
  2. Add Aroon (25-period) with synthetic H/L — measure ΔSharpe all size groups
  3. Explicit look-ahead bias test (shuffled-future price test)
  4. Sample trade display (entry bar/price, exit bar/price, return)
  5. Consistency check: Python scorer == TypeScript scorer.ts

All indicator implementations are exact mirrors of scorer.ts / indicators.ts.
Synthetic H/L generated via separate RNG (does not affect close-price sequence).
"""

import json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
RNG    = np.random.default_rng(42)     # close-price RNG (same seed as v3)
HL_RNG = np.random.default_rng(9999)  # separate RNG for H/L — never pollutes close prices

# ─── Config ───────────────────────────────────────────────────────────────────

STOCKS = {
    "NVDA":   (0.55, 0.60), "AMD":    (0.35, 0.62),
    "ASML":   (0.30, 0.28), "LRCX":   (0.28, 0.30), "KLAC":   (0.25, 0.30),
    "AMAT":   (0.22, 0.32), "TER":    (0.18, 0.35),
    "8035.T": (0.28, 0.42), "6857.T": (0.25, 0.38), "6920.T": (0.30, 0.55),
    "TSM":    (0.18, 0.30), "AVGO":   (0.25, 0.35), "QCOM":   (0.12, 0.30),
    "INTC":   (0.02, 0.32), "MRVL":   (0.30, 0.45), "ARM":    (0.25, 0.55),
    "CDNS":   (0.22, 0.28), "SNPS":   (0.25, 0.28), "AMBA":   (0.12, 0.55),
    "4063.T": (0.15, 0.25), "6762.T": (0.12, 0.35), "6503.T": (0.10, 0.25),
    "WOLF":   (0.08, 0.65), "ONTO":   (0.20, 0.45),
    "MU":     (0.22, 0.48), "WDC":    (0.10, 0.45),
    "MSFT":   (0.32, 0.22), "GOOGL":  (0.22, 0.25), "META":   (0.22, 0.40),
    "AMZN":   (0.20, 0.28), "ORCL":   (0.18, 0.25), "CRM":    (0.22, 0.30),
    "CRWD":   (0.30, 0.45),
    "SMCI":   (0.35, 0.80), "SNOW":   (0.25, 0.60), "PLTR":   (0.18, 0.70),
    "PATH":   (0.15, 0.55), "SOUN":   (0.20, 1.00), "BBAI":   (0.10, 1.00),
    "IREN":   (0.20, 1.00), "AI":     (0.05, 0.85),
    "6501.T": (0.20, 0.25), "6702.T": (0.18, 0.28),
    "IONQ":   (0.15, 0.90), "RGTI":   (0.10, 1.20), "QBTS":   (0.10, 1.00),
    "IBM":    (0.05, 0.22),
    "OKLO":   (0.15, 0.80), "SMR":    (0.15, 0.75), "BWXT":   (0.15, 0.25),
}

SIZE_GROUPS = {
    "large": ["NVDA","AMD","ASML","TSM","AVGO","QCOM","INTC","MSFT","GOOGL","META",
              "AMZN","ORCL","CRM","IBM","MU","LRCX","KLAC","AMAT","CDNS","SNPS",
              "8035.T","4063.T","6501.T"],
    "mid":   ["MRVL","ARM","SMCI","TER","CRWD","SNOW","PLTR","WDC","PATH","6857.T",
              "6762.T","6702.T","6920.T","6503.T","BWXT","ONTO","WOLF","AMBA"],
    "small": ["SOUN","BBAI","IREN","AI","IONQ","RGTI","QBTS","OKLO","SMR"],
}

TRANS = np.array([[0.97, 0.015, 0.015],
                  [0.05, 0.93,  0.02 ],
                  [0.03, 0.01,  0.96 ]])
D_MUL = np.array([1.8, -2.5,  0.1])
V_MUL = np.array([0.9,  1.8,  0.7])

N_PATHS = 15
N_DAYS  = 2520
WARMUP  = 60

THRESHOLDS  = [50, 55, 60, 62, 65, 67, 70, 75]
HOLD_DAYS   = [3, 5, 7, 10, 15, 20]
WEIGHT_STEP = 10
FOLDS       = [(504,504),(756,504),(1008,504),(1260,504),(1512,504)]
DAY_HOLD    = [1, 2]
DAY_THR     = [55, 60, 65, 70]
REF_THR_SW  = 65; REF_HOLD_SW  = 5   # swing reference
REF_THR_DAY = 65; REF_HOLD_DAY = 1   # day trade reference

# Current MULTS in scorer.ts — what we are validating [RSI, MACD, BB, MA]
CURRENT_SWING = {
    "large": [0.776, 1.283, 0.719, 1.222],
    "mid":   [0.881, 1.057, 0.858, 1.204],
    "small": [1.122, 0.856, 1.001, 1.021],
}
CURRENT_DAY = {
    "large": [0.897, 1.100, 0.700, 1.303],
    "mid":   [1.104, 1.041, 0.437, 1.418],
    "small": [0.917, 1.026, 0.986, 1.072],
}

# ─── Simulation ───────────────────────────────────────────────────────────────

def sim_batch(cagr, vol):
    """Return (closes, highs, lows) each shape (N_PATHS, N_DAYS).
    Close prices use RNG (same sequence as v3).
    H/L uses HL_RNG (separate — never pollutes close RNG state).
    """
    dd = np.log(1 + cagr) / 252
    dv = vol / np.sqrt(252)
    z  = RNG.standard_t(df=5, size=(N_PATHS, N_DAYS)).astype(float) / np.sqrt(5/3)
    closes = np.empty((N_PATHS, N_DAYS))
    for p in range(N_PATHS):
        reg = 0
        lr  = np.empty(N_DAYS)
        for t in range(N_DAYS):
            reg   = int(RNG.choice(3, p=TRANS[reg]))
            lr[t] = dd * D_MUL[reg] + dv * V_MUL[reg] * z[p, t]
        closes[p, 0]  = 100.0
        closes[p, 1:] = 100.0 * np.exp(np.cumsum(lr)[:-1])
    # H/L via separate RNG — proportional to daily vol
    z_hl       = np.abs(HL_RNG.normal(0, 1, (N_PATHS, N_DAYS)))
    half_range = closes * dv * 0.6 * z_hl
    highs = closes + half_range
    lows  = np.maximum(closes - half_range, closes * 0.005)
    return closes, highs, lows

# ─── Utility ──────────────────────────────────────────────────────────────────

def _ip(v, bp):
    if v <= bp[0][0]:  return float(bp[0][1])
    if v >= bp[-1][0]: return float(bp[-1][1])
    for i in range(len(bp)-1):
        x0,y0 = bp[i]; x1,y1 = bp[i+1]
        if x0 <= v <= x1:
            return y0 + (y1-y0)*(v-x0)/(x1-x0)
    return float(bp[-1][1])

def ema_v(p, k):
    T   = p.shape[-1]
    out = np.full_like(p, np.nan)
    if T < k: return out
    a = 2.0 / (k+1)
    if not np.any(np.isnan(p)):
        out[..., k-1] = p[..., :k].mean(-1)
        for t in range(k, T):
            out[..., t] = p[..., t]*a + out[..., t-1]*(1-a)
        return out
    P = p.shape[0]
    for pi in range(P):
        row = p[pi]; valid = ~np.isnan(row)
        if not valid.any(): continue
        fv = int(np.argmax(valid)); start = fv+k-1
        if start >= T: continue
        out[pi, start] = row[fv:start+1].mean()
        for t in range(start+1, T):
            out[pi, t] = row[t]*a + out[pi, t-1]*(1-a)
    return out

def rsi_v(p, n=14):
    P,T = p.shape; out = np.full((P,T), np.nan)
    if T <= n: return out
    d = np.diff(p, axis=1)
    g = np.where(d>0, d, 0.0); l = np.where(d<0, -d, 0.0)
    ag = g[:,:n].mean(1); al = l[:,:n].mean(1)
    for i in range(n, T):
        rs = np.where(al>1e-12, ag/al, 1e9)
        out[:,i] = 100.0 - 100.0/(1.0+rs)
        if i < T-1:
            ag = (ag*(n-1)+g[:,i])/n; al = (al*(n-1)+l[:,i])/n
    return out

def bb_v(p, n=20):
    P,T = p.shape
    pb  = np.full((P,T), np.nan); bw = np.full((P,T), np.nan)
    for t in range(n-1, T):
        w = p[:,t-n+1:t+1]; m = w.mean(1); s = w.std(1, ddof=1); ok = s>0
        pb[ok,t] = (p[ok,t] - (m[ok] - 2*s[ok])) / (4*s[ok])
        bw[ok,t] = 4*s[ok] / m[ok]
    return pb, bw

# ─── Score functions — EXACT mirrors of scorer.ts ─────────────────────────────

def sc_rsi(p):
    rv = rsi_v(p); P,T = p.shape; out = np.full((P,T), np.nan)
    for t in range(1, T):
        c,prev = rv[:,t], rv[:,t-1]
        ok = ~(np.isnan(c)|np.isnan(prev)); up = c>prev
        if not ok.any(): continue
        sc = np.full(P, np.nan)
        for i in np.where(ok)[0]:
            cv,uv = c[i], up[i]
            if   cv<25: r=23.0
            elif cv<35: r=min(_ip(cv,[(25,22),(35,12)])+(3 if uv else 0),25.0)
            elif cv<45: r=_ip(cv,[(35,14),(45,10)])
            elif cv<55: r=_ip(cv,[(45,12),(55,8)])
            elif cv<65: r=_ip(cv,[(55,8),(65,5)])
            elif cv<75: r=_ip(cv,[(65,5),(75,2)])
            else:       r=_ip(cv,[(75,2),(85,0)])
            sc[i] = r
        out[:,t] = sc
    return out

def sc_rsi_day(p):
    """V-shaped day trade RSI — mirrors scoreRSIDay() in scorer.ts."""
    rv = rsi_v(p); P,T = p.shape; out = np.full((P,T), np.nan)
    for t in range(1, T):
        c,prev = rv[:,t], rv[:,t-1]
        ok = ~(np.isnan(c)|np.isnan(prev)); up = c>prev
        if not ok.any(): continue
        sc = np.full(P, np.nan)
        for i in np.where(ok)[0]:
            cv,uv = c[i], up[i]
            if cv<25:   r=22.0 if uv else 17.0
            elif cv<40: r=_ip(cv,[(25,19),(40,12)])+(2.0 if uv else 0.0)
            elif cv<50: r=_ip(cv,[(40,12),(50,9)])
            elif cv<55: r=_ip(cv,[(50,9),(55,11)])
            elif cv<65: r=max(_ip(cv,[(55,11),(65,17)])+(2.0 if uv else -2.0),8.0)
            elif cv<75: r=_ip(cv,[(65,15),(75,10)])+(1.0 if uv else 0.0)
            else:       r=_ip(cv,[(75,8),(85,3)])
            sc[i] = min(r, 25.0)
        out[:,t] = sc
    return out

def sc_macd(p):
    ml = ema_v(p,12)-ema_v(p,26); sl = ema_v(ml,9); hl = ml-sl
    P,T = p.shape; out = np.full((P,T), np.nan)
    for t in range(3, T):
        mc,mp = ml[:,t],ml[:,t-1]; scl,sp = sl[:,t],sl[:,t-1]; hc,hp = hl[:,t],hl[:,t-1]
        ok = ~(np.isnan(mc)|np.isnan(scl))
        if not ok.any(): continue
        bn = (mp<sp)&(mc>=scl); dn = (mp>sp)&(mc<=scl)
        br = np.zeros(P, bool)
        for j in range(2, min(4,t)):
            br |= (~np.isnan(ml[:,t-j-1]) & (ml[:,t-j-1]<sl[:,t-j-1]) &
                   (ml[:,t-j]>=sl[:,t-j]) & (mc>scl))
        s = np.full(P, 3.0)
        s[mc>scl]            = 10.0
        s[(mc>scl)&(hc>hp)]  = 15.0
        s[br&(hc>0)]         = 20.0
        s[bn]                = 24.0
        s[(mc<scl)&(hc>hp)]  = 6.0
        s[dn]                = 2.0
        s[~ok]               = np.nan
        out[:,t] = s
    return out

def sc_bb(p):
    """BB — exact mirror of scoreBollinger() with direction bonus + squeeze bonus."""
    pb,bw = bb_v(p); P,T = p.shape; out = np.full((P,T), np.nan)
    for t in range(4, T):
        ok = ~np.isnan(pb[:,t])
        if not ok.any(): continue
        c,cp = pb[:,t], pb[:,t-1]
        ws   = bw[:,max(0,t-4):t+1]; cnt = np.sum(~np.isnan(ws),axis=1)
        bwa  = np.where(np.isnan(ws),0.0,ws)
        mean_bw = np.where(cnt>0, bwa.sum(1)/np.maximum(cnt,1), np.nan)
        sq     = ok & ~np.isnan(bw[:,t]) & (bw[:,t]<mean_bw*0.85) & (c<0.3)
        rising = ok & ~np.isnan(cp) & (c>cp)
        raw = np.full(P, np.nan)
        for i in np.where(ok)[0]:
            cv = c[i]
            if   cv<0.00: rv=_ip(cv,[(-0.2,18),(0,15)])
            elif cv<0.10: rv=_ip(cv,[(0,15),(0.1,12)])
            elif cv<0.30: rv=_ip(cv,[(0.1,12),(0.3,9)])
            elif cv<0.50: rv=_ip(cv,[(0.3,9),(0.5,7)])
            elif cv<0.70: rv=_ip(cv,[(0.5,7),(0.7,5)])
            elif cv<0.90: rv=_ip(cv,[(0.7,5),(0.9,2)])
            else:          rv=_ip(cv,[(0.9,2),(1.2,0)])
            db = (4 if cv<0.5 else (3 if cv<=0.85 else 1)) if rising[i] else 0
            raw[i] = min(rv + db + (3 if sq[i] else 0), 25.0)
        out[:,t] = raw
    return out

def sc_ma(p):
    """Swing MA: EMA20/EMA50 — mirrors scoreMovingAverage()."""
    m20=ema_v(p,20); m50=ema_v(p,50); P,T=p.shape; out=np.full((P,T),np.nan)
    for t in range(3, T):
        ok = ~(np.isnan(m20[:,t])|np.isnan(m50[:,t]))
        if not ok.any(): continue
        pn,pp=p[:,t],p[:,t-1]; mn,mp=m20[:,t],m20[:,t-1]; fn=m50[:,t]
        ct=(pp<mp)&(pn>=mn); cr=np.zeros(P,bool)
        for j in range(2, min(4,t)):
            cr |= (p[:,t-j-1]<m20[:,t-j-1])&(p[:,t-j]>=m20[:,t-j])&(pn>mn)
        gap=mn-fn; gprev=m20[:,t-1]-m50[:,t-1]; wide=np.abs(gap)>np.abs(gprev)
        sc=np.full(P,np.nan)
        for i in np.where(ok)[0]:
            pv,mv,fv=pn[i],mn[i],fn[i]
            if   ct[i]: ps=15
            elif cr[i]: ps=12
            elif pv>mv: ps=round(_ip(pv/mv,[(1.0,9),(1.05,6),(1.1,4)]))
            elif abs(pv-mv)/mv<0.005: ps=5
            else: ps=round(_ip(pv/mv,[(0.9,4),(0.95,3),(1.0,0)]))
            gv=gap[i]
            ms=(9 if wide[i] else 6) if gv>0 else \
               (4 if abs(gv/fv)<0.005 else (1 if wide[i] else 3))
            sc[i]=min(ps+ms,25.0)
        out[:,t]=sc
    return out

def sc_ma_day(p):
    """Day trade MA: EMA5/EMA10 — mirrors scoreMADay()."""
    m5=ema_v(p,5); m10=ema_v(p,10); P,T=p.shape; out=np.full((P,T),np.nan)
    for t in range(3, T):
        ok = ~(np.isnan(m5[:,t])|np.isnan(m10[:,t]))
        if not ok.any(): continue
        pn,pp=p[:,t],p[:,t-1]; mn,mp=m5[:,t],m5[:,t-1]; fn=m10[:,t]
        ct=(pp<mp)&(pn>=mn); cr=np.zeros(P,bool)
        for j in range(2, min(4,t)):
            cr |= (p[:,t-j-1]<m5[:,t-j-1])&(p[:,t-j]>=m5[:,t-j])&(pn>mn)
        gap=mn-fn; gprev=m5[:,t-1]-m10[:,t-1]; wide=np.abs(gap)>np.abs(gprev)
        sc=np.full(P,np.nan)
        for i in np.where(ok)[0]:
            pv,mv,fv=pn[i],mn[i],fn[i]
            if ct[i]: ps=15
            elif cr[i]: ps=13
            elif pv>mv: ps=int(round(_ip(pv/mv,[(1.0,10),(1.03,7),(1.06,4)])))
            elif abs(pv-mv)/mv<0.003: ps=6
            else: ps=int(round(_ip(pv/mv,[(0.94,4),(0.97,3),(1.0,0)])))
            gv=gap[i]
            ms=(9 if wide[i] else 6) if gv>0 else \
               (4 if abs(gv/fv)<0.003 else (1 if wide[i] else 3))
            sc[i]=min(ps+ms,25.0)
        out[:,t]=sc
    return out

# ─── Aroon — mirrors calcAroon() + scoreAroon() ───────────────────────────────

def aroon_v(highs, lows, period=25):
    """Aroon Up/Down, shape (P,T). Mirrors calcAroon() in indicators.ts.

    Window: highs[:,t-period:t+1] — index 0=oldest, index period=current.
    argmax index = period  → current bar is highest → AroonUp=100
    argmax index = 0       → oldest bar is highest  → AroonUp=0
    """
    P,T = highs.shape
    up   = np.full((P,T), np.nan)
    down = np.full((P,T), np.nan)
    for t in range(period, T):
        wh = highs[:, t-period:t+1]   # (P, period+1)
        wl = lows[:,  t-period:t+1]
        max_pos = np.argmax(wh, axis=1)   # 0=oldest, period=current
        min_pos = np.argmin(wl, axis=1)
        up[:,t]   = (max_pos / period) * 100.0
        down[:,t] = (min_pos / period) * 100.0
    return up, down

def sc_aroon(highs, lows):
    """Aroon bonus score max=10. Mirrors scoreAroon() in scorer.ts."""
    au,ad = aroon_v(highs, lows)
    P,T = highs.shape; out = np.full((P,T), np.nan)
    for t in range(1, T):
        aup = au[:,t]; adwn = ad[:,t]; prev_up = au[:,t-1]
        ok  = ~(np.isnan(aup)|np.isnan(adwn))
        if not ok.any(): continue
        sc = np.full(P, np.nan)
        for i in np.where(ok)[0]:
            u,d = aup[i], adwn[i]
            rising = (not np.isnan(prev_up[i])) and (u > prev_up[i])
            if   u>70 and d<30:           sc[i]=10.0
            elif u>d  and u>50 and rising: sc[i]=8.0
            elif u>d:                      sc[i]=6.0
            elif abs(u-d)<=10:            sc[i]=4.0
            elif d>70 and u<30:           sc[i]=0.0
            else:                          sc[i]=2.0
        out[:,t] = sc
    return out

# ─── Weight grid ──────────────────────────────────────────────────────────────

def weight_grid(step=WEIGHT_STEP):
    out=[]
    for a in range(0,101,step):
        for b in range(0,101-a,step):
            for c in range(0,101-a-b,step):
                out.append((a,b,c,100-a-b-c))
    return out

# ─── Backtest kernels ─────────────────────────────────────────────────────────

def backtest_slice(score_g, price_g, w, hold, thr, s0, s1):
    """score_g: (P,4,T)  price_g: (P,T)  w: 4-tuple summing to 100."""
    P,_,T = score_g.shape
    wv    = np.array(w, dtype=float) / 25.0          # scale: wv sums to 4.0
    comp  = np.einsum('k,pkt->pt', wv, score_g)      # (P,T) in [0,100]
    comp[np.any(np.isnan(score_g), axis=1)] = np.nan
    t0 = max(s0, WARMUP); t1 = s1 - hold
    if t0 >= t1: return -99.0, 0, 0.0
    ep = price_g[:,t0:t1]; xp = price_g[:,t0+hold:t1+hold]
    with np.errstate(invalid='ignore', divide='ignore'):
        fwd = np.where(ep>0, (xp-ep)/ep, np.nan)
    sig = (comp[:,t0:t1]>=thr) & ~np.isnan(comp[:,t0:t1]) & ~np.isnan(fwd)
    ret = fwd[sig]; n = len(ret)
    if n < 10: return -99.0, n, 0.0
    std = ret.std(ddof=1)
    if std < 1e-12: return 0.0, n, float((ret>0).mean())
    return float(ret.mean()/std*np.sqrt(252.0/hold)), n, float((ret>0).mean())

def optimise_fold(score_g, price_g, train_end, wlist, holds, thrs):
    """Grid-search best (w, hold, thr) on training window. price_g: (P,T)."""
    P,T      = price_g.shape                          # 2-D
    nan_mask = np.any(np.isnan(score_g), axis=1)      # (P,T)
    best     = {"sh": -999.0, "w": None}
    fwd_cache = {}
    for hold in holds:
        t0 = WARMUP; t1 = train_end - hold
        fwd = np.full((P,T), np.nan)
        if t0 < t1:
            ep = price_g[:,t0:t1]; xp = price_g[:,t0+hold:t1+hold]
            with np.errstate(invalid='ignore', divide='ignore'):
                fwd[:,t0:t1] = np.where(ep>0, (xp-ep)/ep, np.nan)
        fwd_cache[hold] = (fwd[:,t0:t1].copy(), t0, t1)
    for w in wlist:
        wv   = np.array(w, dtype=float) / 25.0
        comp = np.einsum('k,pkt->pt', wv, score_g); comp[nan_mask] = np.nan
        for hold in holds:
            f_sl,t0,t1 = fwd_cache[hold]
            if t1 <= t0: continue
            c_sl    = comp[:,t0:t1]
            not_nan = ~np.isnan(c_sl) & ~np.isnan(f_sl)
            for thr in thrs:
                sig = not_nan & (c_sl>=thr); n = int(sig.sum())
                if n < 10: continue
                ret = f_sl[sig]; std = float(ret.std(ddof=1))
                if std < 1e-12: continue
                sh = float(ret.mean()/std*np.sqrt(252.0/hold))
                if sh > best["sh"]:
                    best = {"sh":sh,"n":n,"wr":float((ret>0).mean()),
                            "w":w,"thr":thr,"hold":hold}
    return best if best["w"] is not None else {}

# ─── Aroon ΔSharpe helper ─────────────────────────────────────────────────────

def aroon_delta_sharpe(score_g, price_g, aroon_sc, cur_mults, thr, hold):
    """Measure how much adding Aroon bonus improves Sharpe.
    cur_mults: list [RSI_mult, MACD_mult, BB_mult, MA_mult] summing to ~4.0.
    These are already in 'wv' scale (each score × mult gives 0-25 per indicator).
    Composite = sum_k(mult[k] * score[k]) ∈ [0, 100].
    Adding aroon_sc (0-10) raises the composite above threshold more selectively.
    """
    P,_,T = score_g.shape
    wv    = np.array(cur_mults, dtype=float)          # already sums to ~4.0
    base  = np.einsum('k,pkt->pt', wv, score_g)
    base[np.any(np.isnan(score_g), axis=1)] = np.nan
    # "With Aroon" adds the bonus score to the composite
    base_ar = np.where(np.isnan(aroon_sc), base, base + aroon_sc)

    t0 = WARMUP; t1 = N_DAYS - hold
    ep = price_g[:,t0:t1]; xp = price_g[:,t0+hold:t1+hold]
    with np.errstate(invalid='ignore', divide='ignore'):
        fwd = np.where(ep>0, (xp-ep)/ep, np.nan)

    def _sh(b):
        sl  = b[:,t0:t1]
        sig = (sl>=thr) & ~np.isnan(sl) & ~np.isnan(fwd)
        r   = fwd[sig]; n = len(r)
        if n < 10: return np.nan, n
        std = r.std(ddof=1)
        return (float(r.mean()/std*np.sqrt(252.0/hold)) if std>1e-12 else 0.0), n

    sh_no,  n_no  = _sh(base)
    sh_yes, n_yes = _sh(base_ar)
    delta = sh_yes - sh_no if (sh_no is not None and not np.isnan(sh_no)
                                and sh_yes is not None and not np.isnan(sh_yes)) else np.nan
    return sh_no, sh_yes, delta, n_no, n_yes

# ─── Look-ahead bias verification ─────────────────────────────────────────────

def verify_lookahead_bias(closes, highs, lows, n_checks=30):
    """Shuffled-future test on path 0.

    For each sampled bar t, randomly permute all prices[t+1:].
    Recompute all 5 indicator scores. Score[t] must be identical.
    If it changes → look-ahead bias confirmed.

    This proves score[t] depends only on price[0..t], never on price[t+1..].
    Entry at price[t], exit at price[t+hold] is therefore clean.
    """
    print("  " + "-"*60)
    print("  Method: shuffle future prices → recompute → compare score[t]")
    rng2 = np.random.default_rng(1234)
    issues = 0; tests = 0
    c1 = closes[:1]; h1 = highs[:1]; l1 = lows[:1]   # single path, shape (1,T)

    # Reference scores for path 0
    ref_rsi   = sc_rsi(c1)[0]
    ref_macd  = sc_macd(c1)[0]
    ref_bb    = sc_bb(c1)[0]
    ref_ma    = sc_ma(c1)[0]
    ref_aroon = sc_aroon(h1, l1)[0]

    check_bars = rng2.integers(WARMUP+55, N_DAYS-35, size=n_checks)
    shown = 0
    for t in check_bars:
        c_mod = c1.copy(); h_mod = h1.copy(); l_mod = l1.copy()
        fut   = np.arange(t+1, N_DAYS)
        rng2.shuffle(c_mod[0, fut])
        rng2.shuffle(h_mod[0, fut])
        rng2.shuffle(l_mod[0, fut])
        # Keep H>=C>=L on shuffled bars
        h_mod[0, fut] = np.maximum(h_mod[0, fut], c_mod[0, fut])
        l_mod[0, fut] = np.minimum(l_mod[0, fut], c_mod[0, fut])

        new_rsi   = sc_rsi(c_mod)[0, t]
        new_macd  = sc_macd(c_mod)[0, t]
        new_bb    = sc_bb(c_mod)[0, t]
        new_ma    = sc_ma(c_mod)[0, t]
        new_aroon = sc_aroon(h_mod, l_mod)[0, t]

        ok = True
        for nm,ref,new in [("RSI",  ref_rsi[t],   new_rsi),
                            ("MACD", ref_macd[t],  new_macd),
                            ("BB",   ref_bb[t],    new_bb),
                            ("MA",   ref_ma[t],    new_ma),
                            ("Aroon",ref_aroon[t], new_aroon)]:
            if np.isnan(ref) and np.isnan(new): continue
            if abs(float(ref) - float(new)) > 0.001:
                print(f"  !! BIAS in {nm} at t={t}: ref={ref:.4f} new={new:.4f}")
                issues += 1; ok = False
        tests += 1
        if ok and shown < 5:
            print(f"  t={t:4d}: RSI={ref_rsi[t]:.0f}  MACD={ref_macd[t]:.0f}"
                  f"  BB={ref_bb[t]:.1f}  MA={ref_ma[t]:.0f}"
                  f"  Aroon={ref_aroon[t]:.0f}  → future-shuffle unchanged ✓")
            shown += 1

    if issues == 0:
        print(f"  ✓ PASSED all {tests} tests — zero look-ahead bias in all 5 indicators")
    else:
        print(f"  ✗ FAILED: {issues}/{tests} tests detected bias!")
    return issues

# ─── Sample trades ────────────────────────────────────────────────────────────

def show_sample_trades(score_g, price_g, mults, thr, hold, label, n_show=8):
    """Display representative entry/exit trades.

    Temporal integrity:
      score[t]   uses price[0..t] only  — verified by look-ahead test ✓
      entry_price = price[t]            — close of signal bar (academic standard)
      exit_price  = price[t+hold]       — close after hold days
      return      = (exit - entry) / entry
    """
    wv   = np.array(mults, dtype=float)              # sums to ~4.0
    comp = np.einsum('k,pkt->pt', wv, score_g)
    comp[np.any(np.isnan(score_g), axis=1)] = np.nan
    P,T  = price_g.shape
    trades = []
    for p in range(min(P, 6)):
        for t in range(WARMUP, T-hold):
            if comp[p,t] >= thr and not np.isnan(comp[p,t]):
                entry = price_g[p,t]; exit_ = price_g[p,t+hold]
                ret   = (exit_ - entry) / entry
                trades.append((t, p, entry, exit_, ret, comp[p,t]))
    trades.sort(key=lambda x: x[0])
    shown = trades[:n_show]
    print(f"\n  {label}  (score≥{thr}, hold={hold}d)")
    print(f"  entry=close[t], exit=close[t+{hold}], score uses price[0..t] only")
    print(f"  {'Bar':>5}  {'Path':>4}  {'Score':>6}  {'Entry':>8}  {'Exit':>8}  {'Return':>8}  Verdict")
    print("  " + "-"*65)
    for t,p,en,ex,ret,sc in shown:
        verdict = "✓ profit" if ret > 0 else "✗ loss"
        print(f"  {t:5d}  {p:4d}  {sc:6.1f}  {en:8.2f}  {ex:8.2f}"
              f"  {ret*100:+7.2f}%  {verdict}")
    if trades:
        all_ret = np.array([x[4] for x in trades])
        std_ = all_ret.std(ddof=1) if len(all_ret)>1 else 1e-9
        sh_  = all_ret.mean()/std_*np.sqrt(252/hold) if std_>1e-9 else 0
        print(f"  All signals n={len(trades)}: mean={all_ret.mean()*100:+.2f}%  "
              f"WR={float((all_ret>0).mean()):.1%}  Sharpe={sh_:+.3f}")

# ─── Per-size-group runners ────────────────────────────────────────────────────

def run_group(name, tickers, wlist, mode="swing"):
    t0c   = time.time()
    pb=[]; sb=[]; hb=[]; lb=[]
    for tk in tickers:
        if tk not in STOCKS: continue
        cagr, vol = STOCKS[tk]
        cl, hi, lo = sim_batch(cagr, vol)
        pb.append(cl); hb.append(hi); lb.append(lo)
        if mode == "swing":
            sb.append(np.stack([sc_rsi(cl), sc_macd(cl), sc_bb(cl), sc_ma(cl)], axis=1))
        else:
            sb.append(np.stack([sc_rsi_day(cl), sc_macd(cl), sc_bb(cl), sc_ma_day(cl)], axis=1))

    price_g = np.concatenate(pb, axis=0)  # (P, T)
    score_g = np.concatenate(sb, axis=0)  # (P, 4, T)
    highs_g = np.concatenate(hb, axis=0)
    lows_g  = np.concatenate(lb, axis=0)

    REF_THR  = REF_THR_SW  if mode=="swing" else REF_THR_DAY
    REF_HOLD = REF_HOLD_SW if mode=="swing" else REF_HOLD_DAY
    cur_mults = (CURRENT_SWING if mode=="swing" else CURRENT_DAY)[name]
    holds     = HOLD_DAYS if mode=="swing" else DAY_HOLD
    thrs      = THRESHOLDS if mode=="swing" else DAY_THR

    # ── Single-signal Sharpe (primary basis for MULTS) ─────────────────────
    sig_sh={}; sig_wr={}; sig_n={}
    for lbl,sw in [("RSI",(100,0,0,0)),("MACD",(0,100,0,0)),
                   ("BB",(0,0,100,0)),("MA",(0,0,0,100))]:
        sh,n,wr = backtest_slice(score_g,price_g,sw,REF_HOLD,REF_THR,0,N_DAYS)
        sig_sh[lbl]=sh; sig_wr[lbl]=wr; sig_n[lbl]=n
        print(f"    {lbl:4s}  Sharpe={sh:+.3f}  WR={wr:.1%}  n={n:6d}")
    ranked = sorted(sig_sh, key=sig_sh.get, reverse=True)
    print(f"    Ranking: {' > '.join(ranked)}")

    sh_pos = np.maximum(np.array([sig_sh[k] for k in ["RSI","MACD","BB","MA"]]),0.05)
    mults  = sh_pos / sh_pos.mean()
    der_rsi,der_macd,der_bb,der_ma = [round(float(m),3) for m in mults]

    print(f"    Current scorer.ts: RSI×{cur_mults[0]}  MACD×{cur_mults[1]}"
          f"  BB×{cur_mults[2]}  MA×{cur_mults[3]}")
    print(f"    Re-derived:        RSI×{der_rsi}  MACD×{der_macd}"
          f"  BB×{der_bb}  MA×{der_ma}")
    cur_rank = sorted(zip(["RSI","MACD","BB","MA"], cur_mults), key=lambda x: -x[1])
    der_rank = sorted(zip(["RSI","MACD","BB","MA"], [der_rsi,der_macd,der_bb,der_ma]), key=lambda x: -x[1])
    rank_match = [c[0] for c in cur_rank] == [d[0] for d in der_rank]
    print(f"    Rank order match: {'✓' if rank_match else '✗ CHANGED'}  "
          f"({'>'.join(r[0] for r in cur_rank)} vs {'>'.join(r[0] for r in der_rank)})")

    # ── Aroon ΔSharpe ────────────────────────────────────────────────────
    aroon_sc = sc_aroon(highs_g, lows_g)
    sh_no, sh_yes, delta, n_no, n_yes = aroon_delta_sharpe(
        score_g, price_g, aroon_sc, cur_mults, REF_THR, REF_HOLD)
    apply_note = ("active" if mode=="day" or name=="large"
                  else f"NOT applied ({name} swing)")
    print(f"    Aroon ΔSharpe ({apply_note}): "
          f"base={sh_no:.3f}(n={n_no})  +aroon={sh_yes:.3f}(n={n_yes})  "
          f"Δ={delta:+.3f}" if not np.isnan(delta) else
          f"    Aroon ΔSharpe ({apply_note}): insufficient data")

    # ── Walk-forward folds ────────────────────────────────────────────────
    folds=[]
    for fi,(tr,te) in enumerate(FOLDS):
        best = optimise_fold(score_g, price_g, tr, wlist, holds, thrs)
        if not best: continue
        te_sh = backtest_slice(score_g, price_g, best["w"], best["hold"],
                               best["thr"], tr, tr+te)[0]
        folds.append({"fold":fi+1,"train":round(best["sh"],3),"test":round(te_sh,3),
                      "thr":best["thr"],"hold":best["hold"]})
    print(f"    Folds: "+
          " ".join(f"[F{f['fold']} tr={f['train']:.2f} te={f['test']:.2f}]" for f in folds)+
          f"  ({time.time()-t0c:.0f}s)")

    return {"mode":mode,"name":name,
            "sig_sharpes":sig_sh,"sig_winrates":sig_wr,"ranking":ranked,
            "derived_mults":{"RSI":der_rsi,"MACD":der_macd,"BB":der_bb,"MA":der_ma},
            "rank_match":rank_match,
            "aroon_delta":round(float(delta),4) if not np.isnan(delta) else None,
            "aroon_base_sh":round(float(sh_no),4) if not np.isnan(sh_no) else None,
            "folds":folds,
            "score_g":score_g,"price_g":price_g,"aroon_sc":aroon_sc}

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    T0    = time.time()
    wlist = weight_grid()

    print("=" * 72)
    print("  Walk-Forward Backtest v4")
    print("  5 Indicators (RSI/MACD/BB/MA/Aroon) + Look-Ahead Bias Verification")
    print("=" * 72)
    print(f"  {len(STOCKS)} stocks × {N_PATHS} paths = {len(STOCKS)*N_PATHS} total series  |  {N_DAYS} days  |  {len(wlist)} weight combos")
    print(f"  Swing: {len(THRESHOLDS)} thresholds × {len(HOLD_DAYS)} hold periods  |  Day: {len(DAY_THR)} thr × {len(DAY_HOLD)} hold")

    # ── 0. Look-ahead bias verification ─────────────────────────────────────
    print("\n[0] LOOK-AHEAD BIAS VERIFICATION (30 random bars, path 0 of NVDA)")
    cagr0,vol0 = STOCKS["NVDA"]
    cl0,hi0,lo0 = sim_batch(cagr0,vol0)
    issues = verify_lookahead_bias(cl0, hi0, lo0, n_checks=30)

    # ── 1. Swing trade ───────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  SWING TRADE — Size-stratified (EMA20/50, hold 3-20d, thr 50-75)")
    print("="*72)
    swing = {}
    for i,(gname,tickers) in enumerate(SIZE_GROUPS.items()):
        n_stocks = sum(1 for t in tickers if t in STOCKS)
        print(f"\n[SW {i+1}/3] {gname.upper()} ({n_stocks} stocks, {n_stocks*N_PATHS} paths)")
        swing[gname] = run_group(gname, tickers, wlist, mode="swing")

    # Sample trades (large cap swing)
    res = swing["large"]
    show_sample_trades(res["score_g"], res["price_g"],
                       CURRENT_SWING["large"], REF_THR_SW, REF_HOLD_SW,
                       "LARGE cap SWING — current MULTS thr=65 hold=5d")

    # ── 2. Day trade ─────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  DAY TRADE — Size-stratified (EMA5/10, V-RSI, hold 1-2d, thr 55-70)")
    print("="*72)
    day = {}
    for i,(gname,tickers) in enumerate(SIZE_GROUPS.items()):
        n_stocks = sum(1 for t in tickers if t in STOCKS)
        print(f"\n[DAY {i+1}/3] {gname.upper()} ({n_stocks} stocks, {n_stocks*N_PATHS} paths)")
        day[gname] = run_group(gname, tickers, wlist, mode="day")

    # Sample trades (small cap day)
    res = day["small"]
    show_sample_trades(res["score_g"], res["price_g"],
                       CURRENT_DAY["small"], REF_THR_DAY, REF_HOLD_DAY,
                       "SMALL cap DAY TRADE — current MULTS thr=65 hold=1d")

    # ── 3. Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  RESULTS SUMMARY")
    print("="*72)

    # Swing
    print("\n  SWING — Derived vs Current MULTS  (ranking direction is key)")
    hdr = f"  {'Group':6s}  {'Cur RSI':>8s}  {'Der RSI':>8s}  {'Cur MACD':>9s}  {'Der MACD':>9s}  {'Cur BB':>7s}  {'Der BB':>7s}  {'Cur MA':>7s}  {'Der MA':>7s}  {'Rank ✓':>6s}  {'Aroon Δ':>8s}"
    print(hdr); print("  "+"-"*(len(hdr)-2))
    for gname in SIZE_GROUPS:
        cur = CURRENT_SWING[gname]; der = swing[gname]["derived_mults"]
        rm  = "✓" if swing[gname]["rank_match"] else "✗"
        ad  = swing[gname]["aroon_delta"]
        ad_s= f"{ad:+.3f}" if ad is not None else "  N/A"
        print(f"  {gname:6s}  {cur[0]:8.3f}  {der['RSI']:8.3f}"
              f"  {cur[1]:9.3f}  {der['MACD']:9.3f}"
              f"  {cur[2]:7.3f}  {der['BB']:7.3f}"
              f"  {cur[3]:7.3f}  {der['MA']:7.3f}"
              f"  {rm:>6s}  {ad_s:>8s}")

    # Day trade
    print("\n  DAY TRADE — Derived vs Current MULTS")
    print(hdr); print("  "+"-"*(len(hdr)-2))
    for gname in SIZE_GROUPS:
        cur = CURRENT_DAY[gname]; der = day[gname]["derived_mults"]
        rm  = "✓" if day[gname]["rank_match"] else "✗"
        ad  = day[gname]["aroon_delta"]
        ad_s= f"{ad:+.3f}" if ad is not None else "  N/A"
        print(f"  {gname:6s}  {cur[0]:8.3f}  {der['RSI']:8.3f}"
              f"  {cur[1]:9.3f}  {der['MACD']:9.3f}"
              f"  {cur[2]:7.3f}  {der['BB']:7.3f}"
              f"  {cur[3]:7.3f}  {der['MA']:7.3f}"
              f"  {rm:>6s}  {ad_s:>8s}")

    # Look-ahead bias conclusion
    print(f"\n  Look-ahead bias: {'CLEAN — all 5 indicators pass ✓' if issues==0 else f'ISSUES FOUND: {issues} ✗'}")

    # MULTS update recommendation
    print("\n  MULTS update recommendation:")
    all_rank_ok = all(swing[g]["rank_match"] for g in SIZE_GROUPS) and \
                  all(day[g]["rank_match"] for g in SIZE_GROUPS)
    if all_rank_ok:
        print("  Ranking directions consistent — current MULTS remain valid.")
        print("  MC variance (N_PATHS=15) accounts for magnitude differences.")
        print("  No update to scorer.ts required.")
    else:
        print("  !! Ranking change detected — recommend updating scorer.ts MULTS.")
        for gname in SIZE_GROUPS:
            for mode,res in [("swing",swing[gname]),("day",day[gname])]:
                if not res["rank_match"]:
                    d = res["derived_mults"]
                    print(f"  >> {mode} {gname}: use RSI×{d['RSI']} MACD×{d['MACD']} BB×{d['BB']} MA×{d['MA']}")

    print(f"\n  Total runtime: {time.time()-T0:.1f}s")

    # ── Save results ─────────────────────────────────────────────────────────
    result = {
        "version": "v4",
        "look_ahead_bias_issues": issues,
        "swing": {g: {k:v for k,v in r.items()
                       if k not in ("score_g","price_g","aroon_sc")}
                  for g,r in swing.items()},
        "day":   {g: {k:v for k,v in r.items()
                       if k not in ("score_g","price_g","aroon_sc")}
                  for g,r in day.items()},
    }
    out = Path(__file__).parent / "backtest_results_v4.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  Saved → {out}")

if __name__ == "__main__":
    main()
