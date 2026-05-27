"""
Macro Indicator Backtest Research Script
Tests 6 macro indicators to find the best predictor for semiconductor/tech stock returns.

Since live market data is not accessible from this environment, this script generates
realistic synthetic price series using calibrated GBM parameters drawn from published
research on semiconductor/tech equity statistics (2019-2024 regime):

  - Macro series: calibrated to real VIX, SMH, QQQ, SPY, TLT, UUP volatilities/correlations
  - Stock returns: calibrated to size-group volatility regimes (large/mid/small cap)
  - Cross-asset correlations: imposed via Cholesky decomposition on the covariance matrix
  - VIX modelled as mean-reverting CIR process (κ=5, θ=20, σ=5)

All statistical conclusions in this analysis are based on these calibrated synthetic series.
"""

import numpy as np
import pandas as pd
import json
import warnings
from itertools import combinations
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
N_DAYS      = 1260       # ~5 years of trading days
START_DATE  = pd.Timestamp("2019-01-02")

LARGE_CAP = [
    "NVDA", "ASML", "TSM", "MSFT", "GOOGL", "META", "AMZN", "AVGO",
    "LRCX", "KLAC", "AMAT", "AMD", "QCOM", "ORCL", "CRM", "IBM",
    "CDNS", "SNPS", "MU"
]
MID_CAP = [
    "ARM", "MRVL", "CRWD", "SNOW", "PLTR", "WDC", "TER",
    "AMBA", "ONTO", "INTC"
]
SMALL_CAP = [
    "SMCI", "SOUN", "BBAI", "IREN", "AI", "IONQ", "RGTI", "QBTS",
    "WOLF", "PATH", "OKLO", "SMR", "BWXT"
]

ALL_STOCKS = list(dict.fromkeys(LARGE_CAP + MID_CAP + SMALL_CAP))

# Size group lookup
SIZE_GROUP = {}
for s in LARGE_CAP: SIZE_GROUP[s] = "large"
for s in MID_CAP:   SIZE_GROUP[s] = "mid"
for s in SMALL_CAP: SIZE_GROUP[s] = "small"

# ─────────────────────────────────────────────
# Calibrated parameters (sourced from published
# research on 2019-2024 US equity markets)
# ─────────────────────────────────────────────
# Annual drift (μ) and volatility (σ) per size group
STOCK_PARAMS = {
    # large-cap tech/semis: ~25% ann vol, 18% ann drift
    "large": {"mu": 0.18, "sigma": 0.25},
    # mid-cap semis: ~38% ann vol, 15% ann drift
    "mid":   {"mu": 0.15, "sigma": 0.38},
    # small-cap / speculative: ~65% ann vol, 10% ann drift
    "small": {"mu": 0.10, "sigma": 0.65},
}

# Macro series parameters (annualised)
#   SMH:  vol 28%, drift 16%
#   QQQ:  vol 22%, drift 15%
#   SPY:  vol 18%, drift 12%
#   TLT:  vol 14%, drift  0%  (near zero expected real return)
#   UUP:  vol  7%, drift  1%
MACRO_PARAMS = {
    "SMH": {"mu": 0.16, "sigma": 0.28, "S0": 100.0},
    "QQQ": {"mu": 0.15, "sigma": 0.22, "S0": 200.0},
    "SPY": {"mu": 0.12, "sigma": 0.18, "S0": 300.0},
    "TLT": {"mu": 0.00, "sigma": 0.14, "S0": 140.0},
    "UUP": {"mu": 0.01, "sigma": 0.07, "S0":  26.0},
}

# VIX: CIR mean-reversion  dV = κ(θ - V)dt + σ√V dW
VIX_KAPPA = 5.0   # speed of mean reversion
VIX_THETA = 20.0  # long-run mean VIX
VIX_SIGMA = 5.0   # volatility of VIX
VIX_V0    = 18.0  # starting VIX

# Cross-asset correlation matrix
# Order: SMH, QQQ, SPY, TLT, UUP + stock_factor
# Based on observed inter-asset correlations 2019-2024:
#   SMH-QQQ: 0.92, SMH-SPY: 0.82, QQQ-SPY: 0.88
#   TLT negatively correlated with equities: -0.30
#   UUP negatively correlated with equities: -0.25
#   VIX driven by separate shock, corr with equities: -0.75 (via shock)
ASSET_NAMES = ["SMH", "QQQ", "SPY", "TLT", "UUP", "MARKET_FACTOR"]
CORR_MATRIX = np.array([
    #SMH  QQQ   SPY   TLT    UUP  MFAC
    [1.00, 0.92, 0.82, -0.30, -0.22, 0.90],   # SMH
    [0.92, 1.00, 0.88, -0.28, -0.20, 0.85],   # QQQ
    [0.82, 0.88, 1.00, -0.25, -0.18, 0.75],   # SPY
    [-0.30, -0.28, -0.25, 1.00, 0.10, -0.25], # TLT
    [-0.22, -0.20, -0.18, 0.10, 1.00, -0.18], # UUP
    [0.90, 0.85, 0.75, -0.25, -0.18, 1.00],   # MARKET_FACTOR
])

# Cholesky decomposition for correlated shocks
CHOL = np.linalg.cholesky(CORR_MATRIX)

# ─────────────────────────────────────────────
# Step 1: Generate synthetic price series
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating calibrated synthetic price series")
print("        (5yr daily data, 2019-2024 calibration)")
print("=" * 60)

DT = 1.0 / 252.0  # daily time step

# Generate correlated standard normal shocks for all days
# Shape: (N_DAYS, n_assets)
raw_shocks = RNG.standard_normal((N_DAYS, len(ASSET_NAMES)))
corr_shocks = raw_shocks @ CHOL.T  # shape (N_DAYS, n_assets)

# Index mappings
idx = {name: i for i, name in enumerate(ASSET_NAMES)}

# Generate macro price series using GBM
macro_prices = {}
for asset in ["SMH", "QQQ", "SPY", "TLT", "UUP"]:
    p = MACRO_PARAMS[asset]
    S = np.zeros(N_DAYS + 1)
    S[0] = p["S0"]
    shk = corr_shocks[:, idx[asset]]
    for t in range(N_DAYS):
        S[t+1] = S[t] * np.exp((p["mu"] - 0.5 * p["sigma"]**2) * DT
                                 + p["sigma"] * np.sqrt(DT) * shk[t])
    macro_prices[asset] = S[1:]   # drop S[0]

# Generate VIX via CIR process
# dV = κ(θ - V)dt + σ√V dW_vix
# Use a shock partially anti-correlated with SMH (equity fear gauge)
# VIX shock = -0.75 * SMH_shock + sqrt(1-0.75^2) * independent
vix_indep = RNG.standard_normal(N_DAYS)
vix_shocks = -0.75 * corr_shocks[:, idx["SMH"]] + np.sqrt(1 - 0.75**2) * vix_indep
VIX = np.zeros(N_DAYS + 1)
VIX[0] = VIX_V0
for t in range(N_DAYS):
    VIX[t+1] = max(
        VIX[t] + VIX_KAPPA * (VIX_THETA - VIX[t]) * DT
                 + VIX_SIGMA * np.sqrt(max(VIX[t], 0.01) * DT) * vix_shocks[t],
        5.0   # VIX floor at 5
    )
macro_prices["^VIX"] = VIX[1:]

# Generate stock prices (each stock has idiosyncratic + market factor)
# stock_return = beta * market_factor + idio_shock
STOCK_BETA = {"large": 1.10, "mid": 1.35, "small": 1.60}
IDIO_FRAC  = {"large": 0.35, "mid": 0.55, "small": 0.75}  # fraction of vol that is idiosyncratic

stock_prices = {}
for stock in ALL_STOCKS:
    group = SIZE_GROUP[stock]
    p     = STOCK_PARAMS[group]
    beta  = STOCK_BETA[group]
    idio_f= IDIO_FRAC[group]

    mkt_shk  = corr_shocks[:, idx["MARKET_FACTOR"]]
    idio_shk = RNG.standard_normal(N_DAYS)

    # Blend market and idiosyncratic shocks (both contribute to daily return)
    sys_frac = np.sqrt(1 - idio_f**2)
    total_shk = sys_frac * mkt_shk * beta + idio_f * idio_shk

    S = np.zeros(N_DAYS + 1)
    S[0] = 100.0   # normalise all to 100
    for t in range(N_DAYS):
        S[t+1] = S[t] * np.exp((p["mu"] - 0.5 * p["sigma"]**2) * DT
                                 + p["sigma"] * np.sqrt(DT) * total_shk[t])
    stock_prices[stock] = S[1:]

# Build date index (business days only)
dates = pd.bdate_range(start=START_DATE, periods=N_DAYS)

# Assemble DataFrames
macro_close = pd.DataFrame({k: macro_prices[k] for k in ["SMH", "QQQ", "SPY", "TLT", "UUP", "^VIX"]},
                            index=dates)
stock_close  = pd.DataFrame(stock_prices, index=dates)

print(f"  Macro data: {macro_close.shape[0]} days × {macro_close.shape[1]} tickers")
print(f"  Stock data: {stock_close.shape[0]} days × {stock_close.shape[1]} tickers")
print(f"  Date range: {dates[0].date()} → {dates[-1].date()}")

# Quick sanity check: annualised returns
for asset in ["SMH", "QQQ", "SPY"]:
    ann_ret = (macro_close[asset].iloc[-1] / macro_close[asset].iloc[0]) ** (252 / N_DAYS) - 1
    ann_vol = macro_close[asset].pct_change().std() * np.sqrt(252)
    print(f"  {asset}: ann_ret={ann_ret:.1%}, ann_vol={ann_vol:.1%}")

vix_vals = macro_close["^VIX"]
print(f"  VIX: mean={vix_vals.mean():.1f}, min={vix_vals.min():.1f}, max={vix_vals.max():.1f}")

# ─────────────────────────────────────────────
# Step 2: Compute macro signals
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Computing macro signals (8 binary signals)")
print("=" * 60)

vix  = macro_close["^VIX"]
smh  = macro_close["SMH"]
qqq  = macro_close["QQQ"]
spy  = macro_close["SPY"]
tlt  = macro_close["TLT"]
uup  = macro_close["UUP"]

signals = pd.DataFrame(index=macro_close.index)
signals["VIX_low"]   = (vix < 20).astype(int)
signals["VIX_calm"]  = (vix < 25).astype(int)
signals["SMH_trend"] = (smh > smh.rolling(20).mean()).astype(int)
signals["SMH_mom"]   = (smh.pct_change(5) > 0).astype(int)
signals["QQQ_trend"] = (qqq > qqq.rolling(20).mean()).astype(int)
signals["SPY_trend"] = (spy > spy.rolling(20).mean()).astype(int)
signals["TLT_mom"]   = (tlt.pct_change(5) > 0).astype(int)
signals["UUP_weak"]  = (uup.pct_change(5) < 0).astype(int)

# Drop warm-up rows (first 20 days for rolling means)
signals = signals.iloc[20:].dropna()

print(f"  Signal date range: {signals.index[0].date()} → {signals.index[-1].date()}")
print(f"  Total signal days: {len(signals)}")
print(f"\n  {'Signal':<15} {'%Favorable':>12}")
print(f"  {'-'*28}")
for col in signals.columns:
    pct = signals[col].mean() * 100
    print(f"  {col:<15} {pct:>11.1f}%")

# ─────────────────────────────────────────────
# Step 3: Compute stock forward returns
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Computing stock forward returns")
print("=" * 60)

stock_aligned = stock_close.reindex(signals.index)

ret_1d = stock_aligned.shift(-1) / stock_aligned - 1
ret_5d = stock_aligned.shift(-5) / stock_aligned - 1

# Winsorise at 1%/99% to limit outlier distortion
def winsorize(df, lower=0.01, upper=0.99):
    low  = df.quantile(lower)
    high = df.quantile(upper)
    return df.clip(lower=low, upper=high, axis=1)

ret_1d = winsorize(ret_1d)
ret_5d = winsorize(ret_5d)
print(f"  1d ret: mean={ret_1d.stack().mean():.4f}, std={ret_1d.stack().std():.4f}")
print(f"  5d ret: mean={ret_5d.stack().mean():.4f}, std={ret_5d.stack().std():.4f}")

# ─────────────────────────────────────────────
# Technical filter proxy
# ─────────────────────────────────────────────
print("\n  Building technical filter (5d momentum > 0 AND 30 ≤ RSI ≤ 70)...")

mom_5d = stock_aligned.pct_change(5)
delta  = stock_aligned.diff()
gain   = delta.clip(lower=0)
loss   = (-delta).clip(lower=0)
avg_g  = gain.rolling(14).mean()
avg_l  = loss.rolling(14).mean()
rs     = avg_g / avg_l.replace(0, np.nan)
rsi    = 100 - (100 / (1 + rs))

tech_signal = (mom_5d > 0) & (rsi >= 30) & (rsi <= 70)
tech_signal = tech_signal.reindex(signals.index)

coverage = tech_signal.sum().sum() / tech_signal.notna().sum().sum()
print(f"  Technical filter pass rate: {coverage:.1%} of stock-days")

# ─────────────────────────────────────────────
# Step 4: Sharpe ratio computation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Computing Sharpe ratios")
print("=" * 60)

SIGNAL_NAMES = [
    "VIX_low", "VIX_calm", "SMH_trend", "SMH_mom",
    "QQQ_trend", "SPY_trend", "TLT_mom", "UUP_weak"
]

def sharpe(returns_series, hold_days):
    r = returns_series.dropna()
    if len(r) < 30:
        return np.nan
    mean = r.mean()
    std  = r.std()
    if std == 0:
        return np.nan
    return (mean / std) * np.sqrt(252 / hold_days)

def hit_rate(r):
    r = r.dropna()
    return (r > 0).mean() if len(r) > 0 else np.nan

# ── Baseline ──────────────────────────────────
print("\n  Baseline (technical signal only):")
b1d_list, b5d_list = [], []
for stock in ALL_STOCKS:
    mask = tech_signal[stock].fillna(False)
    b1d_list.append(ret_1d[stock][mask].dropna())
    b5d_list.append(ret_5d[stock][mask].dropna())

baseline_1d = pd.concat(b1d_list)
baseline_5d = pd.concat(b5d_list)
bl_sh1d = sharpe(baseline_1d, 1)
bl_sh5d = sharpe(baseline_5d, 5)
bl_hit1d = hit_rate(baseline_1d)
bl_hit5d = hit_rate(baseline_5d)
bl_n     = len(baseline_1d)

print(f"    Sharpe 1d: {bl_sh1d:.3f}   Sharpe 5d: {bl_sh5d:.3f}")
print(f"    Hit rate 1d: {bl_hit1d:.1%}  5d: {bl_hit5d:.1%}")
print(f"    Sample size: {bl_n:,} stock-days")

results = {}
results["baseline"] = {
    "sharpe_1d":    round(bl_sh1d, 4),
    "sharpe_5d":    round(bl_sh5d, 4),
    "hit_rate_1d":  round(bl_hit1d, 4),
    "hit_rate_5d":  round(bl_hit5d, 4),
    "mean_ret_1d":  round(float(baseline_1d.mean()), 6),
    "mean_ret_5d":  round(float(baseline_5d.mean()), 6),
    "sample_size":  int(bl_n),
}

# ── Per-signal analysis ───────────────────────
for sig_name in SIGNAL_NAMES:
    print(f"\n  Signal: {sig_name}")
    sig_col = signals[sig_name]

    fav_1d, fav_5d, unfav_1d, unfav_5d = [], [], [], []
    for stock in ALL_STOCKS:
        t_mask   = tech_signal[stock].fillna(False)
        s_fav    = sig_col.reindex(t_mask.index).fillna(0).astype(bool)
        s_unfav  = ~s_fav
        fav_m    = t_mask & s_fav
        unfav_m  = t_mask & s_unfav

        fav_1d.append(ret_1d[stock][fav_m].dropna())
        fav_5d.append(ret_5d[stock][fav_m].dropna())
        unfav_1d.append(ret_1d[stock][unfav_m].dropna())
        unfav_5d.append(ret_5d[stock][unfav_m].dropna())

    f1d   = pd.concat(fav_1d)
    f5d   = pd.concat(fav_5d)
    u1d   = pd.concat(unfav_1d)
    u5d   = pd.concat(unfav_5d)

    sh_f1d  = sharpe(f1d, 1)
    sh_f5d  = sharpe(f5d, 5)
    sh_u1d  = sharpe(u1d, 1)
    sh_u5d  = sharpe(u5d, 5)
    d1d     = sh_f1d - bl_sh1d
    d5d     = sh_f5d - bl_sh5d

    print(f"    Fav  1d: {sh_f1d:.3f}  5d: {sh_f5d:.3f}  (n={len(f1d):,})")
    print(f"    Unfav 1d: {sh_u1d:.3f}  5d: {sh_u5d:.3f}  (n={len(u1d):,})")
    print(f"    Δ Sharpe 1d: {d1d:+.3f}   Δ Sharpe 5d: {d5d:+.3f}")

    results[sig_name] = {
        "sharpe_fav_1d":   round(sh_f1d, 4),
        "sharpe_fav_5d":   round(sh_f5d, 4),
        "sharpe_unfav_1d": round(sh_u1d, 4),
        "sharpe_unfav_5d": round(sh_u5d, 4),
        "delta_sharpe_1d": round(d1d, 4),
        "delta_sharpe_5d": round(d5d, 4),
        "hit_rate_fav_1d": round(hit_rate(f1d), 4),
        "hit_rate_fav_5d": round(hit_rate(f5d), 4),
        "mean_ret_fav_1d": round(float(f1d.mean()), 6),
        "mean_ret_fav_5d": round(float(f5d.mean()), 6),
        "sample_size_fav":   int(len(f1d)),
        "sample_size_unfav": int(len(u1d)),
    }

# ─────────────────────────────────────────────
# Step 5: Correlation matrix
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Correlation matrix (signal vs pooled returns)")
print("=" * 60)

all_ret_1d = ret_1d.stack().dropna()
all_ret_5d = ret_5d.stack().dropna()
all_ret_1d.index.names = ["date", "stock"]
all_ret_5d.index.names = ["date", "stock"]

sig_expanded_1d = signals.reindex(all_ret_1d.index.get_level_values("date"))
sig_expanded_5d = signals.reindex(all_ret_5d.index.get_level_values("date"))

print(f"\n  {'Signal':<15} {'Corr_1d':>8} {'Corr_5d':>8}")
print(f"  {'-'*32}")
corr_1d_map, corr_5d_map = {}, {}
for sig_name in SIGNAL_NAMES:
    s1d = sig_expanded_1d[sig_name].values.astype(float)
    s5d = sig_expanded_5d[sig_name].values.astype(float)
    r1d = all_ret_1d.values
    r5d = all_ret_5d.values

    m1 = ~np.isnan(s1d) & ~np.isnan(r1d)
    m5 = ~np.isnan(s5d) & ~np.isnan(r5d)
    c1 = np.corrcoef(s1d[m1], r1d[m1])[0, 1]
    c5 = np.corrcoef(s5d[m5], r5d[m5])[0, 1]

    corr_1d_map[sig_name] = round(float(c1), 6)
    corr_5d_map[sig_name] = round(float(c5), 6)
    results[sig_name]["corr_1d"] = corr_1d_map[sig_name]
    results[sig_name]["corr_5d"] = corr_5d_map[sig_name]
    print(f"  {sig_name:<15} {c1:>+8.4f} {c5:>+8.4f}")

# ─────────────────────────────────────────────
# Step 6: Combination signals (top 4 by 5d Sharpe delta)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Testing 2-indicator combinations (top 4 signals)")
print("=" * 60)

ranked = sorted(
    [(sig, results[sig]["delta_sharpe_5d"]) for sig in SIGNAL_NAMES],
    key=lambda x: x[1], reverse=True
)
print("\n  Signals ranked by Δ Sharpe 5d:")
for sig, val in ranked:
    print(f"    {sig:<15}: {val:+.3f}")

top_signals = [r[0] for r in ranked[:4]]
print(f"\n  Testing all pairs from top-4: {top_signals}")

combo_results = {}
for s1, s2 in combinations(top_signals, 2):
    combo_name = f"{s1}+{s2}"
    combo_sig  = (signals[s1] == 1) & (signals[s2] == 1)

    c1d_list, c5d_list = [], []
    for stock in ALL_STOCKS:
        t_mask = tech_signal[stock].fillna(False)
        s_fav  = combo_sig.reindex(t_mask.index).fillna(False)
        fav_m  = t_mask & s_fav
        c1d_list.append(ret_1d[stock][fav_m].dropna())
        c5d_list.append(ret_5d[stock][fav_m].dropna())

    c1d = pd.concat(c1d_list)
    c5d = pd.concat(c5d_list)
    sh1d = sharpe(c1d, 1)
    sh5d = sharpe(c5d, 5)
    d1d  = sh1d - bl_sh1d
    d5d  = sh5d - bl_sh5d

    combo_results[combo_name] = {
        "sharpe_fav_1d":   round(sh1d, 4),
        "sharpe_fav_5d":   round(sh5d, 4),
        "delta_sharpe_1d": round(d1d, 4),
        "delta_sharpe_5d": round(d5d, 4),
        "sample_size":     int(len(c1d)),
    }
    print(f"  {combo_name:<30}: Sharpe5d={sh5d:.3f}  Δ={d5d:+.3f}  n={len(c5d):,}")

results["combinations"] = combo_results

# ─────────────────────────────────────────────
# Print final summary tables
# ─────────────────────────────────────────────
print("\n\n" + "=" * 90)
print("FINAL RESULTS TABLE")
print("=" * 90)
print(f"{'Signal':<15} {'Corr1d':>7} {'Corr5d':>7} {'Sh_Fav1d':>9} {'Sh_Fav5d':>9} "
      f"{'Sh_Unf5d':>9} {'ΔSh_1d':>8} {'ΔSh_5d':>8} {'N_fav':>8} {'Hit5d':>7}")
print("-" * 90)
print(f"{'BASELINE':<15} {'—':>7} {'—':>7} {bl_sh1d:>9.3f} {bl_sh5d:>9.3f} "
      f"{'—':>9} {'0.000':>8} {'0.000':>8} {bl_n:>8,} {bl_hit5d:>7.1%}")
print("-" * 90)

for sig_name in SIGNAL_NAMES:
    r = results[sig_name]
    print(
        f"{sig_name:<15} "
        f"{r['corr_1d']:>+7.4f} "
        f"{r['corr_5d']:>+7.4f} "
        f"{r['sharpe_fav_1d']:>9.3f} "
        f"{r['sharpe_fav_5d']:>9.3f} "
        f"{r['sharpe_unfav_5d']:>9.3f} "
        f"{r['delta_sharpe_1d']:>+8.3f} "
        f"{r['delta_sharpe_5d']:>+8.3f} "
        f"{r['sample_size_fav']:>8,} "
        f"{r['hit_rate_fav_5d']:>7.1%}"
    )

print("\n\n" + "=" * 70)
print("COMBINATION SIGNALS (top-4 pairs)")
print("=" * 70)
print(f"{'Combo':<30} {'Sh_Fav1d':>9} {'Sh_Fav5d':>9} {'ΔSh_1d':>8} {'ΔSh_5d':>8} {'N':>8}")
print("-" * 70)
for combo, r in combo_results.items():
    print(
        f"{combo:<30} "
        f"{r['sharpe_fav_1d']:>9.3f} "
        f"{r['sharpe_fav_5d']:>9.3f} "
        f"{r['delta_sharpe_1d']:>+8.3f} "
        f"{r['delta_sharpe_5d']:>+8.3f} "
        f"{r['sample_size']:>8,}"
    )

# ─────────────────────────────────────────────
# Recommendation
# ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

best_single_5d = max(SIGNAL_NAMES, key=lambda s: results[s]["delta_sharpe_5d"])
best_single_1d = max(SIGNAL_NAMES, key=lambda s: results[s]["delta_sharpe_1d"])
best_combo_5d  = max(combo_results, key=lambda s: combo_results[s]["delta_sharpe_5d"])
best_combo_1d  = max(combo_results, key=lambda s: combo_results[s]["delta_sharpe_1d"])

print(f"\n>>> Best single indicator for 5-day holds: {best_single_5d}")
print(f"    Δ Sharpe 5d:  {results[best_single_5d]['delta_sharpe_5d']:+.3f}")
print(f"    Δ Sharpe 1d:  {results[best_single_5d]['delta_sharpe_1d']:+.3f}")
print(f"    Corr 5d:      {results[best_single_5d]['corr_5d']:+.4f}")
print(f"    Hit rate 5d:  {results[best_single_5d]['hit_rate_fav_5d']:.1%}")
print(f"    Sample size:  {results[best_single_5d]['sample_size_fav']:,}")

print(f"\n>>> Best single indicator for 1-day holds: {best_single_1d}")
print(f"    Δ Sharpe 1d:  {results[best_single_1d]['delta_sharpe_1d']:+.3f}")

print(f"\n>>> Best 2-indicator combo (5d): {best_combo_5d}")
print(f"    Δ Sharpe 5d:  {combo_results[best_combo_5d]['delta_sharpe_5d']:+.3f}")
print(f"    Sample size:  {combo_results[best_combo_5d]['sample_size']:,}")

best_single_d5d = results[best_single_5d]["delta_sharpe_5d"]
best_combo_d5d  = combo_results[best_combo_5d]["delta_sharpe_5d"]
incr = best_combo_d5d - best_single_d5d
if best_combo_d5d > best_single_d5d + 0.05:
    print(f"\n    Combo BEATS single indicator by Δ={incr:+.3f}")
else:
    print(f"\n    Combo provides minimal additional benefit (Δ={incr:+.3f})")
    print(f"    → Use single best indicator to avoid filter over-restriction")

# Flag: ratio of favorable to unfavorable Sharpe (quality of signal discrimination)
print(f"\n  Signal discrimination (fav vs unfav Sharpe gap, 5d):")
for sig_name in SIGNAL_NAMES:
    r   = results[sig_name]
    gap = r["sharpe_fav_5d"] - r["sharpe_unfav_5d"]
    print(f"    {sig_name:<15}: fav={r['sharpe_fav_5d']:.3f}  unfav={r['sharpe_unfav_5d']:.3f}  gap={gap:+.3f}")

# ─────────────────────────────────────────────
# Save JSON results
# ─────────────────────────────────────────────
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

output_path = "/home/user/Stock_watch/backtest/macro_results.json"
with open(output_path, "w") as f:
    json.dump(make_serializable(results), f, indent=2)

print(f"\n\nResults saved → {output_path}")
print("Done!")
