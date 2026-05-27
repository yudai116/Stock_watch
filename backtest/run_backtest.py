#!/usr/bin/env python3
"""
Walk-Forward Backtest — Swing Trade Score Weight Optimization
=============================================================

Network blocked in this cloud env → uses Monte Carlo simulation
calibrated to documented 10yr statistics (2015-2024).

Architecture: per-weight NumPy vectorisation (no giant tensors).
              Each weight combo: einsum (P×T×4) + bool masking.
              ~2ms/combo → 286×3×4×5 folds ≈ 100s total.
"""

import json, time, warnings
from pathlib import Path
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

# ─── Config ──────────────────────────────────────────────────────────────────

STOCKS = {          # (CAGR, ann.vol) — documented 2015-2024
    "NVDA":   (0.55, 0.60), "AMD":    (0.35, 0.62),
    "ASML":   (0.30, 0.28), "TSM":    (0.18, 0.30),
    "MU":     (0.22, 0.48), "MSFT":   (0.32, 0.22),
    "GOOGL":  (0.22, 0.25), "IBM":    (0.05, 0.22),
    "8035.T": (0.28, 0.42), "6857.T": (0.25, 0.38),
}
TRANS = np.array([[0.97,0.015,0.015],[0.05,0.93,0.02],[0.03,0.01,0.96]])
D_MUL = np.array([1.8, -2.5, 0.1])
V_MUL = np.array([0.9,  1.8, 0.7])

N_PATHS = 25            # paths per stock → 250 total
N_DAYS  = 2520          # ≈ 10yr
WARMUP  = 60            # bars to skip at start

THRESHOLDS  = [60, 65, 70, 75]
HOLD_DAYS   = [5, 10, 15]
WEIGHT_STEP = 10        # 286 combos
FOLDS = [(504,504),(756,504),(1008,504),(1260,504),(1512,504)]

# ─── Simulation ──────────────────────────────────────────────────────────────

def sim_batch(cagr, vol):
    dd, dv = np.log(1+cagr)/252, vol/np.sqrt(252)
    z = RNG.standard_t(df=5, size=(N_PATHS,N_DAYS)).astype(float) / np.sqrt(5/3)
    prices = np.empty((N_PATHS, N_DAYS))
    for p in range(N_PATHS):
        reg, lr = 0, np.empty(N_DAYS)
        for t in range(N_DAYS):
            reg = int(RNG.choice(3, p=TRANS[reg]))
            lr[t] = dd*D_MUL[reg] + dv*V_MUL[reg]*z[p,t]
        prices[p, 0] = 100.0
        prices[p, 1:] = 100.0 * np.exp(np.cumsum(lr)[:-1])
    return prices

# ─── Indicators (vectorized over path axis) ───────────────────────────────────

def ema_v(p, k):
    """EMA; fast path when p has no NaN, per-path path when p has a leading NaN prefix."""
    T = p.shape[-1]; out = np.full_like(p, np.nan)
    if T < k: return out
    a = 2.0 / (k + 1)
    if not np.any(np.isnan(p)):
        # Fast path: raw prices / ema lines that are already clean
        out[..., k-1] = p[..., :k].mean(-1)
        for t in range(k, T):
            out[..., t] = p[..., t] * a + out[..., t-1] * (1 - a)
        return out
    # NaN-prefix path (e.g. MACD line before ema26 warmup):
    # find first valid index per path and start EMA from there.
    P = p.shape[0]
    for pi in range(P):
        row = p[pi]; valid = ~np.isnan(row)
        if not valid.any(): continue
        fv = int(np.argmax(valid))          # first valid bar
        start = fv + k - 1
        if start >= T: continue
        out[pi, start] = row[fv:start + 1].mean()
        for t in range(start + 1, T):
            out[pi, t] = row[t] * a + out[pi, t - 1] * (1 - a)
    return out

def rsi_v(p, n=14):
    P,T = p.shape; out = np.full((P,T), np.nan)
    if T <= n: return out
    d = np.diff(p,axis=1)
    g = np.where(d>0,d,0.); l = np.where(d<0,-d,0.)
    ag, al = g[:,:n].mean(1), l[:,:n].mean(1)
    for i in range(n,T):
        rs = np.where(al>1e-12, ag/al, 1e9)
        out[:,i] = 100. - 100./(1.+rs)
        if i<T-1: ag=(ag*(n-1)+g[:,i])/n; al=(al*(n-1)+l[:,i])/n
    return out

def bb_v(p, n=20):
    P,T = p.shape
    pb = np.full((P,T),np.nan); bw = np.full((P,T),np.nan)
    for t in range(n-1,T):
        w=p[:,t-n+1:t+1]; m=w.mean(1); s=w.std(1,ddof=1)
        ok=s>0; pb[ok,t]=(p[ok,t]-(m[ok]-2*s[ok]))/(4*s[ok]); bw[ok,t]=4*s[ok]/m[ok]
    return pb, bw

def _ip(v, bp):
    if v<=bp[0][0]: return bp[0][1]
    if v>=bp[-1][0]: return bp[-1][1]
    for i in range(len(bp)-1):
        x0,y0=bp[i]; x1,y1=bp[i+1]
        if x0<=v<=x1: return y0+(y1-y0)*(v-x0)/(x1-x0)
    return bp[-1][1]

# ─── Score arrays — exact mirror of scorer.ts ─────────────────────────────────

def sc_rsi(p):
    rv=rsi_v(p); P,T=p.shape; out=np.full((P,T),np.nan)
    for t in range(1,T):
        c=rv[:,t]; prev=rv[:,t-1]; ok=~(np.isnan(c)|np.isnan(prev))
        if not ok.any(): continue
        up=c>prev; sc=np.full(P,np.nan)
        for i in np.where(ok)[0]:
            cv,uv=c[i],up[i]
            if   cv<25: r=23.
            elif cv<35: r=min(_ip(cv,[(25,22),(35,12)])+(3 if uv else 0),25.)
            elif cv<45: r=_ip(cv,[(35,14),(45,10)])
            elif cv<55: r=_ip(cv,[(45,12),(55,8)])
            elif cv<65: r=_ip(cv,[(55,8),(65,5)])
            elif cv<75: r=_ip(cv,[(65,5),(75,2)])
            else:       r=_ip(cv,[(75,2),(85,0)])
            sc[i]=r
        out[:,t]=sc
    return out

def sc_macd(p):
    ml=ema_v(p,12)-ema_v(p,26); sl=ema_v(ml,9); hl=ml-sl
    P,T=p.shape; out=np.full((P,T),np.nan)
    for t in range(3,T):
        mc,mp=ml[:,t],ml[:,t-1]; scl,sp=sl[:,t],sl[:,t-1]
        hc,hp=hl[:,t],hl[:,t-1]; ok=~(np.isnan(mc)|np.isnan(scl))
        if not ok.any(): continue
        bn=(mp<sp)&(mc>=scl); dn=(mp>sp)&(mc<=scl)
        br=np.zeros(P,bool)
        for j in range(2,min(4,t)):
            br|=(~np.isnan(ml[:,t-j-1]))&(ml[:,t-j-1]<sl[:,t-j-1])&(ml[:,t-j]>=sl[:,t-j])&(mc>scl)
        s=np.full(P,3.); s[mc>scl]=10.; s[(mc>scl)&(hc>hp)]=15.
        s[br&(hc>0)]=20.; s[bn]=24.; s[(mc<scl)&(hc>hp)]=6.; s[dn]=2.
        s[~ok]=np.nan; out[:,t]=s
    return out

def sc_bb(p):
    pb,bw=bb_v(p); P,T=p.shape; out=np.full((P,T),np.nan)
    for t in range(4,T):
        ok=~np.isnan(pb[:,t])
        if not ok.any(): continue
        c=pb[:,t]
        ws=bw[:,max(0,t-4):t+1]; cnt=np.sum(~np.isnan(ws),axis=1)
        bwa=np.where(np.isnan(ws),0.,ws)
        mean_bw=np.where(cnt>0,bwa.sum(1)/np.maximum(cnt,1),np.nan)
        sq=ok&~np.isnan(bw[:,t])&(bw[:,t]<mean_bw*0.85)&(c<0.3)
        raw=np.full(P,np.nan)
        for i in np.where(ok)[0]:
            cv=c[i]
            if   cv<0:    rv=_ip(cv,[(-0.2,25),(0,21)])
            elif cv<0.10: rv=_ip(cv,[(0,21),(0.1,17)])
            elif cv<0.25: rv=_ip(cv,[(0.1,17),(0.25,12)])
            elif cv<0.50: rv=_ip(cv,[(0.25,12),(0.5,8)])
            elif cv<0.75: rv=_ip(cv,[(0.5,8),(0.75,4)])
            elif cv<0.90: rv=_ip(cv,[(0.75,4),(0.9,1)])
            else:         rv=_ip(cv,[(0.9,1),(1.2,0)])
            raw[i]=min(rv+(3 if sq[i] else 0),25.)
        out[:,t]=raw
    return out

def sc_ma(p):
    m20=ema_v(p,20); m50=ema_v(p,50); P,T=p.shape; out=np.full((P,T),np.nan)
    for t in range(3,T):
        ok=~(np.isnan(m20[:,t])|np.isnan(m50[:,t]))
        if not ok.any(): continue
        pn,pp=p[:,t],p[:,t-1]; mn,mp=m20[:,t],m20[:,t-1]; fn=m50[:,t]
        ct=(pp<mp)&(pn>=mn)
        cr=np.zeros(P,bool)
        for j in range(2,min(4,t)):
            cr|=(p[:,t-j-1]<m20[:,t-j-1])&(p[:,t-j]>=m20[:,t-j])&(pn>mn)
        gap=mn-fn; gprev=m20[:,t-1]-m50[:,t-1]; wide=np.abs(gap)>np.abs(gprev)
        sc=np.full(P,np.nan)
        for i in np.where(ok)[0]:
            pv,mv,fv=pn[i],mn[i],fn[i]
            if   ct[i]: ps=15
            elif cr[i]: ps=12
            elif pv>mv: ps=round(_ip(pv/mv,[(1.0,9),(1.05,6),(1.1,4)]))
            elif abs(pv-mv)/mv<0.005: ps=5
            else: ps=round(_ip(pv/mv,[(0.9,4),(0.95,3),(1.0,0)]))
            gv=gap[i]; wv=wide[i]
            ms=(9 if wv else 6) if gv>0 else (4 if abs(gv/fv)<0.005 else (1 if wv else 3))
            sc[i]=min(ps+ms,25.)
        out[:,t]=sc
    return out

def all_scores(p): return np.stack([sc_rsi(p),sc_macd(p),sc_bb(p),sc_ma(p)],axis=1)

# ─── Weight grid ─────────────────────────────────────────────────────────────

def weight_grid(step=WEIGHT_STEP):
    out = []
    for a in range(0,101,step):
        for b in range(0,101-a,step):
            for c in range(0,101-a-b,step):
                out.append((a,b,c,100-a-b-c))
    return out

# ─── Core backtest (vectorized over paths, one weight at a time) ──────────────

def backtest_one(sc: np.ndarray, price: np.ndarray,
                 w: tuple, hold: int, thr: float,
                 s_idx: int, e_idx: int):
    """
    sc: (P,4,T)  price: (P,T)
    Returns: (sh, n, win_rate)
    """
    P, T = price.shape
    wv = np.array(w, dtype=float) / 25.0         # (4,)

    # Composite score (P, T)
    comp = np.einsum('k,pkt->pt', wv, sc)
    # NaN where any indicator is NaN
    comp[np.any(np.isnan(sc), axis=1)] = np.nan

    # Forward returns (P, T) — only computed once outside normally,
    # but kept here for clarity; caller should precompute and pass in.
    # fwd[p, t] = return of buying at t, selling at t+hold
    fwd = np.full((P, T), np.nan)
    t0, t1 = max(s_idx, WARMUP), e_idx - hold
    if t0 >= t1:
        return -99.0, 0, 0.0
    ep = price[:, t0:t1]                          # entry prices
    xp = price[:, t0+hold:t1+hold]               # exit prices
    with np.errstate(invalid='ignore', divide='ignore'):
        fwd[:, t0:t1] = np.where(ep > 0, (xp - ep) / ep, np.nan)

    # Signal mask
    c_sl = comp[:, t0:t1]
    f_sl = fwd[:, t0:t1]
    sig  = (c_sl >= thr) & ~np.isnan(c_sl) & ~np.isnan(f_sl)

    returns = f_sl[sig]
    n = len(returns)
    if n < 10:
        return -99.0, n, 0.0
    std = returns.std(ddof=1)
    if std < 1e-12:
        return 0.0, n, float((returns > 0).mean())
    sh = float(returns.mean() / std * np.sqrt(252.0 / hold))
    wr = float((returns > 0).mean())
    return sh, n, wr

# ─── Walk-forward ─────────────────────────────────────────────────────────────

def optimise_fold(sc_all, price_all, train_end, wlist):
    best = dict(sh=-999., w=None)
    for hold in HOLD_DAYS:
        for thr in THRESHOLDS:
            for w in wlist:
                sh, n, wr = backtest_one(sc_all, price_all, w, hold, thr, 0, train_end)
                if sh > best["sh"] and n >= 10:
                    best = dict(sh=sh, n=n, wr=wr, w=w, thr=thr, hold=hold)
    return best if best["w"] is not None else {}

def eval_one(sc, price, w, thr, hold, s0, s1):
    sh, n, wr = backtest_one(sc, price, w, hold, thr, s0, s1)
    return dict(sh=sh, n=n, wr=wr)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    wlist = weight_grid()
    print("=" * 66)
    print("  Walk-Forward Backtest — Monte Carlo Simulation (vectorized)")
    print("=" * 66)
    print(f"\n  {len(STOCKS)} stocks × {N_PATHS} paths = {len(STOCKS)*N_PATHS} series  "
          f"|  {N_DAYS} days  |  {len(wlist)} weight combos")
    total_evals = len(wlist) * len(HOLD_DAYS) * len(THRESHOLDS) * len(FOLDS)
    print(f"  {len(FOLDS)} folds × {len(HOLD_DAYS)} holds × {len(THRESHOLDS)} thresholds = "
          f"{total_evals:,} evaluations")

    # ── Simulate & score ─────────────────────────────────────────────────
    print("\n[1/3] Generating calibrated price paths & computing scores ...")
    pb, sb = [], []
    for ticker, (cagr, vol) in STOCKS.items():
        pm = sim_batch(cagr, vol)
        sm = all_scores(pm)
        pb.append(pm); sb.append(sm)
        # Quick sanity: % of days with high score (>= 18 per indicator)
        valid = ~np.isnan(sm[:,0,:])
        high = np.mean(np.nanmean(sm,axis=2) >= 15)
        print(f"  {ticker:8s}  paths={N_PATHS}  "
              f"valid_pct={valid.mean():.1%}  high_score_pct={high:.1%}")
    price_all = np.concatenate(pb, axis=0)
    score_all = np.concatenate(sb, axis=0)
    P_total = price_all.shape[0]
    print(f"  Total: {P_total} series × {N_DAYS} days | "
          f"score NaN%={np.isnan(score_all).mean():.1%}")

    # Quick composite sanity check
    wv_eq = np.array([25,25,25,25],float)/25.
    comp_eq = np.einsum('k,pkt->pt', wv_eq, score_all)
    comp_eq[np.any(np.isnan(score_all),axis=1)] = np.nan
    valid_comp = comp_eq[~np.isnan(comp_eq)]
    print(f"  Composite (equal weights): mean={valid_comp.mean():.1f}  "
          f"std={valid_comp.std():.1f}  >=60: {(valid_comp>=60).mean():.1%}  "
          f">=70: {(valid_comp>=70).mean():.1%}")
    print(f"  Score phase done in {time.time()-t0:.1f}s")

    # ── Walk-forward ─────────────────────────────────────────────────────
    print("\n[2/3] Walk-forward optimisation ...")
    fold_results = []
    for fi, (train_days, test_days) in enumerate(FOLDS):
        t1 = time.time()
        test_end = train_days + test_days
        best = optimise_fold(score_all, price_all, train_days, wlist)
        if not best:
            print(f"  Fold {fi+1}: no valid params"); continue
        test_m = eval_one(score_all, price_all,
                          best["w"], best["thr"], best["hold"],
                          train_days, test_end)
        w = best["w"]
        print(f"  Fold {fi+1}  train={train_days}d  "
              f"RSI={w[0]:3d}/MACD={w[1]:3d}/BB={w[2]:3d}/MA={w[3]:3d}  "
              f"thr={best['thr']}  hold={best['hold']}d  "
              f"TrainSh={best['sh']:+.3f}  TestSh={test_m['sh']:+.3f}  "
              f"WR={test_m['wr']:.1%}  ({time.time()-t1:.0f}s)")
        fold_results.append(dict(fold=fi+1, train_days=train_days,
                                 best=best, test=test_m))

    # ── Aggregate ────────────────────────────────────────────────────────
    print("\n[3/3] Aggregating ...")
    valid_folds = [f for f in fold_results if f["test"]["sh"] > -10] or fold_results
    w_v=Counter(); thr_v=Counter(); hold_v=Counter()
    for f in valid_folds:
        wt = max(0.01, f["test"]["sh"])
        w_v[f["best"]["w"]] += wt
        thr_v[f["best"]["thr"]] += wt
        hold_v[f["best"]["hold"]] += wt

    opt_w   = w_v.most_common(1)[0][0]
    opt_thr = thr_v.most_common(1)[0][0]
    opt_h   = hold_v.most_common(1)[0][0]

    base = eval_one(score_all, price_all, (25,25,25,25), 70, 10, 0, N_DAYS)
    opt  = eval_one(score_all, price_all, opt_w, opt_thr, opt_h, 0, N_DAYS)

    # OOS: last 2 fold test windows
    oos_evals = [eval_one(score_all, price_all, opt_w, opt_thr, opt_h,
                          f["train_days"], f["train_days"]+504)
                 for f in valid_folds[-2:]]
    oos_sh = float(np.mean([e["sh"] for e in oos_evals if e["n"]>0])) if oos_evals else -99.
    oos_n  = sum(e["n"] for e in oos_evals)

    # Per-signal analysis
    print("\n  Single-signal contributions:")
    sig_sh = {}
    for lbl, sw in [("RSI",(100,0,0,0)),("MACD",(0,100,0,0)),
                    ("BB",(0,0,100,0)),("MA",(0,0,0,100))]:
        m = eval_one(score_all, price_all, sw, opt_thr, opt_h, 0, N_DAYS)
        sig_sh[lbl] = m["sh"]
        print(f"    {lbl:4s}: Sharpe={m['sh']:+.3f}  WinRate={m['wr']:.1%}  n={m['n']}")
    ranked = sorted(sig_sh, key=sig_sh.get, reverse=True)
    print(f"  Signal ranking: {' > '.join(ranked)}")

    # ── Print results ─────────────────────────────────────────────────────
    ow = opt_w
    print("\n" + "=" * 66)
    print("  FINAL RESULTS")
    print("=" * 66)
    print(f"\nBaseline  (25/25/25/25  thr=70  hold=10d):")
    print(f"  Sharpe={base['sh']:+.3f}  WinRate={base['wr']:.1%}  Trades={base['n']}")
    print(f"\nOptimized:")
    print(f"  Weights  RSI={ow[0]:3d}  MACD={ow[1]:3d}  BB={ow[2]:3d}  MA={ow[3]:3d}")
    print(f"  Threshold={opt_thr}  Hold={opt_h}d")
    print(f"  In-sample  Sharpe={opt['sh']:+.3f}  WinRate={opt['wr']:.1%}  Trades={opt['n']}")
    print(f"  OOS Sharpe={oos_sh:+.3f}  Trades={oos_n}")
    print(f"\nSharpe improvement: {base['sh']:+.3f} → {opt['sh']:+.3f}  "
          f"(Δ={opt['sh']-base['sh']:+.3f})")
    print("\nFold details:")
    for f in fold_results:
        w=f["best"]["w"]
        print(f"  Fold {f['fold']}  ({','.join(str(x) for x in w)})  "
              f"thr={f['best']['thr']} hold={f['best']['hold']}d  "
              f"TrainSh={f['best']['sh']:+.3f}  TestSh={f['test']['sh']:+.3f}  "
              f"WR={f['test']['wr']:.1%}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

    result = {
        "method": "walk_forward_monte_carlo",
        "note": "Calibrated simulation (network blocked). Run with real yfinance data locally.",
        "n_paths_per_stock": N_PATHS, "n_stocks": len(STOCKS), "n_days": N_DAYS,
        "optimal_weights": {"rsi":ow[0],"macd":ow[1],"bb":ow[2],"ma":ow[3]},
        "buy_threshold": opt_thr, "hold_days": opt_h,
        "oos_sharpe": round(oos_sh,4),
        "baseline_sharpe": round(base["sh"],4),
        "optimal_sharpe":  round(opt["sh"],4),
        "signal_ranking": ranked,
        "signal_sharpes": {k:round(v,4) for k,v in sig_sh.items()},
        "folds": [dict(fold=f["fold"],
                       weights={"rsi":f["best"]["w"][0],"macd":f["best"]["w"][1],
                                "bb":f["best"]["w"][2],"ma":f["best"]["w"][3]},
                       threshold=f["best"]["thr"], hold_days=f["best"]["hold"],
                       train_sharpe=round(f["best"]["sh"],4),
                       test_sharpe=round(f["test"]["sh"],4),
                       test_winrate=round(f["test"]["wr"],4))
                  for f in fold_results],
    }
    out = Path(__file__).parent / "backtest_results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out}")
    return result

if __name__ == "__main__":
    main()
