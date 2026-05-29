#!/usr/bin/env python3
"""
strategy_search.py v3 — DOE (Taguchi L18) → GA → Walk-Forward 最適化

【モード】
  --swing : スイング (1h足, RSI14/MACD12-26/BB20/EMA20-50/Aroon25/Stoch14/CCI20/ROC10)
  --day   : デイトレ (1h足, RSI9/MACD5-13/BB10/EMA9-21/Stoch5/ROC5/CCI14/VWAP偏差)
  --both  : 両方順番に実行

【パイプライン】
  1. DOE  (Taguchi L18 直交表) → 指標の主効果ランキング
  2. GA   (pop=200, gen=120)   → 訓練データ (0-80%) でウェイト最適化
  3. Walk-Forward 検証         → テストデータ (80-100%) で汎化確認

【その他】
  - 取引コスト: US 0.16% / JP 0.30% (往復)
  - ボリュームフィルター: vol >= 0.7 × SMA(vol,20)
  - エッジバーフィルター: 各日の最初・最後の1hバーは除外 (day モードのみ)
  - エントリー価格: シグナルバー翌バーの始値 (先読みバイアスなし)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE          = Path(__file__).parent
DATA_FILE     = HERE / "price_data_intraday.json"
RESULTS_SWING = HERE / "strategy_results_swing.json"
RESULTS_DAY   = HERE / "strategy_results_day.json"

MIN_TRADES    = 15
BARS_PER_YEAR = 1500   # 1h bars/year (US+JP平均)

# ── 取引コスト (往復) ────────────────────────────────────────────────────────
JP_COST = 0.0030   # 東証: 0.30%
US_COST = 0.0016   # 米国: 0.16%

def cost_rate(ticker: str) -> float:
    return JP_COST if ticker.endswith(".T") else US_COST

# ── DOE Taguchi L18 直交表 (2^1 × 3^7) ──────────────────────────────────────
#   列 0: 2水準  列 1-7: 3水準
L18 = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1],
    [0,0,2,2,2,2,2,2],
    [0,1,0,0,1,1,2,2],
    [0,1,1,1,2,2,0,0],
    [0,1,2,2,0,0,1,1],
    [0,2,0,1,0,2,1,2],
    [0,2,1,2,1,0,2,0],
    [0,2,2,0,2,1,0,1],
    [1,0,0,2,2,1,1,0],
    [1,0,1,0,0,2,2,1],
    [1,0,2,1,1,0,0,2],
    [1,1,0,1,2,0,2,1],
    [1,1,1,2,0,1,0,2],
    [1,1,2,0,1,2,1,0],
    [1,2,0,2,1,2,0,1],
    [1,2,1,0,2,0,1,2],
    [1,2,2,1,0,1,2,0],
], dtype=np.int32)

DOE_LEVELS_2 = [0.1, 1.5]            # 2水準列 (列0)
DOE_LEVELS_3 = [0.1, 0.5, 1.5]       # 3水準列 (列1-7)

def doe_weight_row(row: np.ndarray) -> np.ndarray:
    w = np.empty(8, dtype=np.float64)
    w[0] = DOE_LEVELS_2[row[0]]
    for i in range(1, 8):
        w[i] = DOE_LEVELS_3[row[i]]
    return w

# ── 売りルール定義 ────────────────────────────────────────────────────────────
SWING_SELL_RULES: dict[str, dict] = {
    "hold_21b":        {"type": "hold",        "bars": 21},
    "hold_35b":        {"type": "hold",        "bars": 35},
    "hold_70b":        {"type": "hold",        "bars": 70},
    "hold_105b":       {"type": "hold",        "bars": 105},
    "hold_140b":       {"type": "hold",        "bars": 140},
    "target5_stop3":   {"type": "target_stop", "target": 5,  "stop": 3},
    "target10_stop5":  {"type": "target_stop", "target": 10, "stop": 5},
    "target15_stop5":  {"type": "target_stop", "target": 15, "stop": 5},
    "target20_stop7":  {"type": "target_stop", "target": 20, "stop": 7},
    "target15_stop7":  {"type": "target_stop", "target": 15, "stop": 7},
    "trail_5pct":      {"type": "trailing",    "trail": 5},
    "trail_10pct":     {"type": "trailing",    "trail": 10},
}

DAY_SELL_RULES: dict[str, dict] = {
    "hold_2b":        {"type": "hold",        "bars": 2},
    "hold_4b":        {"type": "hold",        "bars": 4},
    "hold_6b":        {"type": "hold",        "bars": 6},
    "hold_8b":        {"type": "hold",        "bars": 8},
    "target3_stop2":  {"type": "target_stop", "target": 3,  "stop": 2},
    "target5_stop3":  {"type": "target_stop", "target": 5,  "stop": 3},
    "target7_stop4":  {"type": "target_stop", "target": 7,  "stop": 4},
    "target10_stop5": {"type": "target_stop", "target": 10, "stop": 5},
    "trail_2pct":     {"type": "trailing",    "trail": 2},
    "trail_3pct":     {"type": "trailing",    "trail": 3},
}

SWING_SELL_HOLD = {
    "hold_21b": 21, "hold_35b": 35, "hold_70b": 70, "hold_105b": 105, "hold_140b": 140,
    "target5_stop3": 35,  "target10_stop5": 50, "target15_stop5": 70,
    "target20_stop7": 80, "target15_stop7": 70,
    "trail_5pct": 50,     "trail_10pct": 70,
}

DAY_SELL_HOLD = {
    "hold_2b": 2, "hold_4b": 4, "hold_6b": 6, "hold_8b": 8,
    "target3_stop2": 4,  "target5_stop3": 6,
    "target7_stop4": 8,  "target10_stop5": 10,
    "trail_2pct": 4,     "trail_3pct": 6,
}

SWING_SELL_JA = {
    "hold_21b":       "固定保有 21h(≈3日)",
    "hold_35b":       "固定保有 35h(≈5日)",
    "hold_70b":       "固定保有 70h(≈10日)",
    "hold_105b":      "固定保有 105h(≈15日)",
    "hold_140b":      "固定保有 140h(≈20日)",
    "target5_stop3":  "利確+5% / ストップ-3%",
    "target10_stop5": "利確+10% / ストップ-5%",
    "target15_stop5": "利確+15% / ストップ-5%",
    "target20_stop7": "利確+20% / ストップ-7%",
    "target15_stop7": "利確+15% / ストップ-7%",
    "trail_5pct":     "トレーリングストップ 5%",
    "trail_10pct":    "トレーリングストップ 10%",
}

DAY_SELL_JA = {
    "hold_2b":        "固定保有 2h",
    "hold_4b":        "固定保有 4h",
    "hold_6b":        "固定保有 6h",
    "hold_8b":        "固定保有 8h",
    "target3_stop2":  "利確+3% / ストップ-2%",
    "target5_stop3":  "利確+5% / ストップ-3%",
    "target7_stop4":  "利確+7% / ストップ-4%",
    "target10_stop5": "利確+10% / ストップ-5%",
    "trail_2pct":     "トレーリングストップ 2%",
    "trail_3pct":     "トレーリングストップ 3%",
}

IND_NAMES_SWING = ["RSI", "MACD", "BB", "MA", "Aroon", "Stoch", "CCI", "ROC"]
IND_NAMES_DAY   = ["RSI", "MACD", "BB", "MA", "Stoch", "ROC", "CCI", "VWAP"]

BUY_THRESHOLDS = [55, 60, 65, 70]

MAX_HOLD_BARS_SWING = 200
MAX_HOLD_BARS_DAY   = 20

# ══════════════════════════════════════════════════════════════════════════════
# 1. データ読み込み
# ══════════════════════════════════════════════════════════════════════════════

def load_data(mode: str) -> dict[str, dict]:
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size < 100:
        print(f"ERROR: {DATA_FILE.name} が見つかりません。")
        print("  node backtest/fetch_data.mjs --intraday  を実行してください")
        sys.exit(1)
    raw = json.loads(DATA_FILE.read_text())
    result = {}
    for ticker, rows in raw.items():
        if len(rows) < 300:
            print(f"  SKIP {ticker}: {len(rows)} bars < 300")
            continue
        result[ticker] = {
            "closes":  np.array([r["close"]  for r in rows], dtype=np.float64),
            "opens":   np.array([r["open"]   for r in rows], dtype=np.float64),
            "highs":   np.array([r["high"]   for r in rows], dtype=np.float64),
            "lows":    np.array([r["low"]    for r in rows], dtype=np.float64),
            "volumes": np.array([r.get("volume", 0) for r in rows], dtype=np.float64),
            "dates":   [r["date"] for r in rows],
            "cost":    cost_rate(ticker),
        }
    return result

# ══════════════════════════════════════════════════════════════════════════════
# 2. フィルターマスク
# ══════════════════════════════════════════════════════════════════════════════

def vol_ok_mask(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """ボリュームフィルター: vol >= 0.7 × SMA(vol, period)"""
    T = len(volumes)
    ok = np.ones(T, dtype=bool)
    sma = np.full(T, np.nan)
    for i in range(period - 1, T):
        sma[i] = volumes[i - period + 1:i + 1].mean()
    ok[:period - 1] = False
    ok[period - 1:] = volumes[period - 1:] >= 0.7 * np.nan_to_num(sma[period - 1:], nan=0.)
    return ok

def edge_bar_mask(dates: list[str], T: int) -> np.ndarray:
    """エッジバーフィルター: 各日の最初・最後のバーは True (除外対象)"""
    edge = np.zeros(T, dtype=bool)
    day_map: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        key = d[:10]
        day_map.setdefault(key, []).append(i)
    for indices in day_map.values():
        if len(indices) >= 2:
            edge[indices[0]]  = True
            edge[indices[-1]] = True
        elif len(indices) == 1:
            edge[indices[0]] = True
    return edge

# ══════════════════════════════════════════════════════════════════════════════
# 3. テクニカル指標計算
# ══════════════════════════════════════════════════════════════════════════════

def _ema(x: np.ndarray, k: int) -> np.ndarray:
    T = len(x); out = np.full(T, np.nan)
    fv = np.where(~np.isnan(x))[0]
    if len(fv) < k: return out
    sv = fv[0]
    if sv + k > T: return out
    out[sv + k - 1] = np.nanmean(x[sv:sv + k])
    a = 2.0 / (k + 1)
    for t in range(sv + k, T):
        out[t] = (x[t] if not np.isnan(x[t]) else out[t-1]) * a + out[t-1] * (1 - a)
    return out

def _sma(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(k - 1, len(x)):
        out[i] = x[i - k + 1:i + 1].mean()
    return out

def calc_rsi(c: np.ndarray, p: int = 14) -> np.ndarray:
    T = len(c); out = np.full(T, np.nan)
    if T < p + 1: return out
    d = np.diff(c)
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ag, al = up[:p].mean(), dn[:p].mean()
    for i in range(p, T - 1):
        ag = (ag * (p-1) + up[i]) / p; al = (al * (p-1) + dn[i]) / p
        out[i+1] = 100. if al == 0 else 100. - 100. / (1. + ag / al)
    return out

def calc_macd(c: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9):
    ef = _ema(c, fast); es = _ema(c, slow)
    ml = ef - es; sl = _ema(ml, sig)
    return ml, sl, ml - sl

def calc_bb(c: np.ndarray, p: int = 20):
    T = len(c); pb = np.full(T, np.nan)
    for i in range(p - 1, T):
        w = c[i - p + 1:i + 1]; m = w.mean(); s = w.std(ddof=1)
        u = m + 2*s; lo = m - 2*s
        pb[i] = (c[i] - lo) / (u - lo) if (u - lo) > 1e-10 else 0.5
    return pb

def calc_aroon(h: np.ndarray, lo: np.ndarray, p: int = 25):
    T = len(h); up = np.full(T, np.nan); dn = np.full(T, np.nan)
    for i in range(p, T):
        wh = h[i - p:i + 1]; wl = lo[i - p:i + 1]
        up[i] = np.argmax(wh) / p * 100
        dn[i] = np.argmin(wl) / p * 100
    return up, dn

def calc_stoch(c: np.ndarray, h: np.ndarray, lo: np.ndarray,
               k_period: int = 14, d_period: int = 3):
    T = len(c); k = np.full(T, np.nan)
    for i in range(k_period - 1, T):
        hi = h[i - k_period + 1:i + 1].max()
        li = lo[i - k_period + 1:i + 1].min()
        k[i] = (c[i] - li) / (hi - li) * 100 if (hi - li) > 1e-10 else 50.
    d = _sma(k, d_period)
    return k, d

def calc_cci(c: np.ndarray, h: np.ndarray, lo: np.ndarray, p: int = 20):
    T = len(c); out = np.full(T, np.nan)
    tp = (c + h + lo) / 3.0
    for i in range(p - 1, T):
        w = tp[i - p + 1:i + 1]; m = w.mean()
        md = np.abs(w - m).mean()
        out[i] = (tp[i] - m) / (0.015 * md) if md > 1e-10 else 0.
    return out

def calc_roc(c: np.ndarray, p: int = 10) -> np.ndarray:
    out = np.full(len(c), np.nan)
    for i in range(p, len(c)):
        if c[i - p] > 0:
            out[i] = (c[i] / c[i - p] - 1) * 100
    return out

def calc_vwap_dev(closes: np.ndarray, volumes: np.ndarray, dates: list[str]) -> np.ndarray:
    """1h足でのVWAP偏差: (close - cumVWAP) / cumVWAP (日別リセット)"""
    T = len(closes); out = np.full(T, np.nan)
    day_map: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        day_map.setdefault(d[:10], []).append(i)
    for indices in day_map.values():
        cum_pv = 0.; cum_v = 0.
        for i in indices:
            v = volumes[i] if volumes[i] > 0 else 1.
            cum_pv += closes[i] * v
            cum_v  += v
            vwap = cum_pv / cum_v
            out[i] = (closes[i] - vwap) / vwap if vwap > 1e-10 else 0.
    return out

# ══════════════════════════════════════════════════════════════════════════════
# 4. スコア計算 (各 0-25 点)
# ══════════════════════════════════════════════════════════════════════════════

def score_rsi(r: np.ndarray) -> np.ndarray:
    s = np.where(r < 25, 23.,
        np.where(r < 35, 19 - (r-25)*0.4,
        np.where(r < 45, 12 - (r-35)*0.2,
        np.where(r < 55, 10 - (r-45)*0.2,
        np.where(r < 65,  7 - (r-55)*0.2,
                           np.maximum(0., 4-(r-65)*0.1))))))
    return np.clip(np.where(np.isnan(r), 0., s), 0., 25.)

def score_macd(ml: np.ndarray, sl: np.ndarray, hl: np.ndarray) -> np.ndarray:
    v = ~np.isnan(ml) & ~np.isnan(sl)
    pml = np.roll(ml, 1); psl = np.roll(sl, 1); phl = np.roll(hl, 1)
    gc = v & (ml > sl)  & (pml <= psl); gc[0] = False
    dc = v & (ml <= sl) & (pml > psl);  dc[0] = False
    ab = v & (ml > sl)  & ~gc
    ex = ab & ~np.isnan(hl) & ~np.isnan(phl) & (hl > phl) & (hl > 0); ex[0] = False
    s  = np.where(gc, 24., np.where(ex, 15., np.where(ab, 10., np.where(dc, 2., 0.))))
    return np.where(v, s, 0.)

def score_bb(pb: np.ndarray) -> np.ndarray:
    s = np.where(pb < 0.,  20.,
        np.where(pb < 0.1, 15 + (0.1-pb)/0.1*5,
        np.where(pb < 0.3,  9 + (0.3-pb)/0.2*6,
        np.where(pb < 0.5,  6 + (0.5-pb)/0.2*3,
        np.where(pb < 0.7,  3 + (0.7-pb)/0.2*3,
        np.where(pb < 0.9,  1 + (0.9-pb)/0.2*2, 0.))))))
    return np.clip(np.where(np.isnan(pb), 0., s), 0., 25.)

def score_ma(c: np.ndarray, ef: np.ndarray, es: np.ndarray) -> np.ndarray:
    v  = ~np.isnan(ef) & ~np.isnan(es)
    rt = np.where(v, c / np.where(ef != 0, ef, 1.), 1.)
    gc = v & (ef > es)
    pc = np.roll(c, 1); pe = np.roll(ef, 1)
    cr = v & (c > ef) & (pc <= pe); cr[0] = False
    base = np.where(cr, 15.,
           np.where(rt > 1., np.minimum(12., 4. + (rt-1.)*100.),
                    np.maximum(0., 3. - (1.-rt)*50.)))
    return np.clip(np.where(v, base + np.where(gc, 8., 0.), 0.), 0., 25.)

def score_aroon(aup: np.ndarray, adn: np.ndarray) -> np.ndarray:
    diff = aup - adn
    s = np.where(diff > 70,  20 + (diff-70)/30*5,
        np.where(diff > 30,  13 + (diff-30)/40*7,
        np.where(diff > 0,    8 + diff/30*5,
        np.where(diff > -30,  4 + (diff+30)/30*4,
        np.where(diff > -70,  2 + (diff+70)/40*2, 0.)))))
    return np.clip(np.where(np.isnan(aup)|np.isnan(adn), 0., s), 0., 25.)

def score_stoch(k: np.ndarray, d: np.ndarray) -> np.ndarray:
    s = np.where(k < 20,  20 + (20-k)/20*5,
        np.where(k < 35,  13 + (35-k)/15*7,
        np.where(k < 50,   7 + (50-k)/15*6,
        np.where(k < 70,   3 + (70-k)/20*4,
        np.where(k < 85,   1 + (85-k)/15*2, 0.)))))
    pk = np.roll(k, 1); pd_ = np.roll(d, 1)
    gc = ~np.isnan(k) & ~np.isnan(d) & (k > d) & (pk <= pd_); gc[0] = False
    s  = np.where(gc, s + 3., s)
    return np.clip(np.where(np.isnan(k), 0., s), 0., 25.)

def score_cci(cci: np.ndarray) -> np.ndarray:
    s = np.where(cci < -200, 22 + np.minimum((cci+300)/100, 1.)*3,
        np.where(cci < -100, 14 + (cci+200)/100*8,
        np.where(cci < 0,     8 + (cci+100)/100*6,
        np.where(cci < 100,   3 + (100-cci)/100*5,
        np.where(cci < 200,   1 + (200-cci)/100*2, 0.)))))
    return np.clip(np.where(np.isnan(cci), 0., s), 0., 25.)

def score_roc(roc: np.ndarray) -> np.ndarray:
    s = np.where(roc > 30,  18 + np.minimum((roc-30)/20, 1.)*7,
        np.where(roc > 15,  12 + (roc-15)/15*6,
        np.where(roc > 5,    7 + (roc-5)/10*5,
        np.where(roc > 0,    4 + roc/5*3,
        np.where(roc > -10,  1 + (roc+10)/10*3, 0.)))))
    return np.clip(np.where(np.isnan(roc), 0., s), 0., 25.)

def score_vwap(vwap_dev: np.ndarray) -> np.ndarray:
    """VWAP偏差: 負 (VWAP以下) ほど買いシグナル"""
    dev_pct = vwap_dev * 100
    s = np.where(dev_pct < -3.,  20 + np.minimum((-dev_pct-3)/2, 1.)*5,
        np.where(dev_pct < -1.5, 13 + (-dev_pct-1.5)/1.5*7,
        np.where(dev_pct < 0,     7 + (-dev_pct)/1.5*6,
        np.where(dev_pct < 1.5,   3 + (1.5-dev_pct)/1.5*4,
        np.where(dev_pct < 3.,    1 + (3.-dev_pct)/1.5*2, 0.)))))
    return np.clip(np.where(np.isnan(vwap_dev), 0., s), 0., 25.)

# ══════════════════════════════════════════════════════════════════════════════
# 5. 指標スコアをまとめて計算
# ══════════════════════════════════════════════════════════════════════════════

def compute_ind_scores(td: dict, mode: str) -> np.ndarray:
    c, h, lo = td["closes"], td["highs"], td["lows"]
    volumes   = td["volumes"]
    dates     = td["dates"]

    if mode == "swing":
        rsi_v            = calc_rsi(c, 14)
        ml, sl, hl       = calc_macd(c, 12, 26, 9)
        pb               = calc_bb(c, 20)
        ef               = _ema(c, 20); es = _ema(c, 50)
        aup, adn         = calc_aroon(h, lo, 25)
        sk, sd           = calc_stoch(c, h, lo, 14, 3)
        cci              = calc_cci(c, h, lo, 20)
        roc              = calc_roc(c, 10)
        return np.stack([
            score_rsi(rsi_v), score_macd(ml, sl, hl),
            score_bb(pb),     score_ma(c, ef, es),
            score_aroon(aup, adn), score_stoch(sk, sd),
            score_cci(cci),   score_roc(roc),
        ], axis=0)
    else:  # day
        rsi_v            = calc_rsi(c, 9)
        ml, sl, hl       = calc_macd(c, 5, 13, 4)
        pb               = calc_bb(c, 10)
        ef               = _ema(c, 9); es = _ema(c, 21)
        sk, sd           = calc_stoch(c, h, lo, 5, 3)
        roc              = calc_roc(c, 5)
        cci              = calc_cci(c, h, lo, 14)
        vwap_dev         = calc_vwap_dev(c, volumes, dates)
        return np.stack([
            score_rsi(rsi_v), score_macd(ml, sl, hl),
            score_bb(pb),     score_ma(c, ef, es),
            score_stoch(sk, sd), score_roc(roc),
            score_cci(cci),   score_vwap(vwap_dev),
        ], axis=0)

# ══════════════════════════════════════════════════════════════════════════════
# 6. 売りルール事前計算 (先読みバイアスなし)
# ══════════════════════════════════════════════════════════════════════════════

def precompute_sell_outcomes(closes, opens, highs, lows, cr: float,
                              sell_rules: dict, max_hold: int) -> dict[str, np.ndarray]:
    T = len(closes)
    outcomes = {}

    for name, rule in sell_rules.items():
        if rule["type"] == "hold":
            N = rule["bars"]
            ret = np.full(T, np.nan)
            valid = T - N - 1
            if valid > 0:
                entry = opens[1:1 + valid]
                exit_ = closes[N:N + valid]
                vm = (entry > 0) & ~np.isnan(entry) & (exit_ > 0) & ~np.isnan(exit_)
                ret[:valid] = np.where(vm, exit_ / entry - 1. - cr, np.nan)
            outcomes[name] = ret

        elif rule["type"] == "target_stop":
            P = rule["target"] / 100.0; S = rule["stop"] / 100.0
            ret = np.full(T, np.nan)
            for t in range(T - 2):
                entry = opens[t + 1]
                if entry <= 0 or np.isnan(entry): continue
                tp = entry * (1 + P); sl = entry * (1 - S)
                result = None
                for i in range(t + 1, min(t + 1 + max_hold, T)):
                    if highs[i] >= tp: result = P - cr;  break
                    if lows[i]  <= sl: result = -S - cr; break
                if result is None:
                    xt = t + 1 + max_hold
                    if xt < T:
                        result = closes[xt] / entry - 1. - cr
                ret[t] = result if result is not None else np.nan
            outcomes[name] = ret

        else:  # trailing
            trail  = rule["trail"] / 100.0
            ret = np.full(T, np.nan)
            for t in range(T - 2):
                entry = opens[t + 1]
                if entry <= 0 or np.isnan(entry): continue
                peak   = entry; result = None
                for i in range(t + 1, min(t + 1 + max_hold, T)):
                    peak = max(peak, highs[i])
                    if lows[i] <= peak * (1 - trail):
                        result = peak * (1 - trail) / entry - 1. - cr; break
                if result is None:
                    xt = t + 1 + max_hold
                    if xt < T:
                        result = closes[xt] / entry - 1. - cr
                ret[t] = result if result is not None else np.nan
            outcomes[name] = ret

    return outcomes

# ══════════════════════════════════════════════════════════════════════════════
# 7. バッチ Sharpe 評価
# ══════════════════════════════════════════════════════════════════════════════

def batch_sharpe(ticker_data: dict, wm: np.ndarray, sell_name: str,
                 thresholds: list, hold_bars: int,
                 t_start: int, t_end: int) -> np.ndarray:
    """
    wm: (N, 8) 重み行列
    返り値: (N, len(thresholds)) Sharpe
    """
    N = len(wm); n_thr = len(thresholds)
    acc_n   = np.zeros((N, n_thr), dtype=np.float64)
    acc_sum = np.zeros((N, n_thr), dtype=np.float64)
    acc_sq  = np.zeros((N, n_thr), dtype=np.float64)

    for td in ticker_data.values():
        ind  = td["ind_scores"][:, t_start:t_end].astype(np.float32)
        sout = td["sell_outcomes"][sell_name][t_start:t_end].astype(np.float32)
        vmask = td["vol_ok"][t_start:t_end]
        if "edge_bar" in td:
            vmask = vmask & ~td["edge_bar"][t_start:t_end]
        valid = ~np.isnan(sout) & vmask

        comp = (wm.astype(np.float32) @ ind)  # (N, T_slice)

        for ti, thr in enumerate(thresholds):
            mask  = (comp >= thr) & valid
            tr    = np.where(mask, sout, np.float32(0.))
            is_t  = mask
            n_    = is_t.sum(1).astype(np.float64)
            s_    = (tr * is_t).sum(1).astype(np.float64)
            sq_   = (tr**2 * is_t).sum(1).astype(np.float64)
            acc_n[:,  ti] += n_
            acc_sum[:, ti] += s_
            acc_sq[:,  ti] += sq_

    shp = np.full((N, n_thr), np.nan)
    for ti in range(n_thr):
        n = acc_n[:, ti]; s = acc_sum[:, ti]; sq = acc_sq[:, ti]
        ok  = n >= MIN_TRADES
        avg = np.where(ok, s / np.where(n > 0, n, 1.), np.nan)
        var = np.where(ok & (n > 1), sq / np.where(n > 0, n, 1.) - avg**2, np.nan)
        std = np.sqrt(np.maximum(var, 0.))
        factor = np.sqrt(BARS_PER_YEAR / max(hold_bars, 1))
        shp[:, ti] = np.where(ok & (std > 1e-10), avg / std * factor, np.nan)

    return shp

def detailed_eval_single(ticker_data: dict, w: np.ndarray, sell_name: str,
                          threshold: int, hold_bars: int,
                          t_start: int, t_end: int) -> dict:
    """単一重みベクトルの詳細評価 (n_trades / win_rate / avg_return / max_dd)"""
    n = 0; s = 0.; sq = 0.; wins = 0; min_ret = np.inf
    wm = w.reshape(1, 8).astype(np.float32)
    for td in ticker_data.values():
        ind   = td["ind_scores"][:, t_start:t_end].astype(np.float32)
        sout  = td["sell_outcomes"][sell_name][t_start:t_end].astype(np.float32)
        vmask = td["vol_ok"][t_start:t_end]
        if "edge_bar" in td:
            vmask = vmask & ~td["edge_bar"][t_start:t_end]
        valid = ~np.isnan(sout) & vmask
        comp  = (wm @ ind)[0]
        mask  = (comp >= threshold) & valid
        tr    = sout[mask]
        if len(tr) == 0: continue
        n    += len(tr)
        s    += float(tr.sum())
        sq   += float((tr**2).sum())
        wins += int((tr > 0).sum())
        min_ret = min(min_ret, float(tr.min()))
    if n == 0:
        return {"sharpe": 0., "n_trades": 0, "win_rate": 0., "avg_return": 0., "max_dd": 0.}
    avg = s / n
    var = sq / n - avg**2
    std = np.sqrt(max(var, 0.))
    factor = np.sqrt(BARS_PER_YEAR / max(hold_bars, 1))
    sharpe = (avg / std * factor) if std > 1e-10 else 0.
    return {
        "sharpe":      round(float(sharpe), 4),
        "n_trades":    n,
        "win_rate":    round(wins / n, 4),
        "avg_return":  round(avg * 100, 3),
        "max_dd":      round((min_ret if min_ret < np.inf else 0.) * 100, 3),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 8. DOE 主効果分析 (Taguchi L18)
# ══════════════════════════════════════════════════════════════════════════════

def run_doe(ticker_data: dict, mode: str, t_train_end: int) -> tuple[dict, list]:
    sell_rules = SWING_SELL_RULES if mode == "swing" else DAY_SELL_RULES
    sell_hold  = SWING_SELL_HOLD  if mode == "swing" else DAY_SELL_HOLD
    ind_names  = IND_NAMES_SWING  if mode == "swing" else IND_NAMES_DAY

    print("  [DOE] L18 直交表実験 (18行) ...")
    doe_sharpes = []   # (18,)
    for exp_i, row in enumerate(L18):
        w = doe_weight_row(row)
        wm = w.reshape(1, 8)
        best_shp = -np.inf
        for sname, rule in sell_rules.items():
            hb  = sell_hold[sname]
            shp_mat = batch_sharpe(ticker_data, wm, sname, BUY_THRESHOLDS, hb, 0, t_train_end)
            best_shp = max(best_shp, float(np.nanmax(shp_mat)))
        doe_sharpes.append(best_shp if np.isfinite(best_shp) else 0.)
        print(f"    exp {exp_i+1:2d}/18  best_sharpe={doe_sharpes[-1]:.3f}")
    doe_sharpes = np.array(doe_sharpes)

    # 主効果: 各列ごとに各水準の平均シャープの範囲 (range)
    effects = {}
    for col_i, name in enumerate(ind_names):
        if col_i == 0:
            levels = [0, 1]; n_lvl = 2
        else:
            levels = [0, 1, 2]; n_lvl = 3
        means = []
        for lv in levels:
            idx = np.where(L18[:, col_i] == lv)[0]
            if len(idx) > 0:
                means.append(doe_sharpes[idx].mean())
            else:
                means.append(0.)
        effects[name] = float(max(means) - min(means))

    ranked = sorted(ind_names, key=lambda n: effects[n], reverse=True)
    print(f"  [DOE] 指標ランキング: {ranked}")
    return effects, ranked

# ══════════════════════════════════════════════════════════════════════════════
# 9. 遺伝的アルゴリズム
# ══════════════════════════════════════════════════════════════════════════════

def run_ga(ticker_data: dict, mode: str, doe_effects: dict,
           best_sell: str, best_thresh: int,
           t_train_end: int) -> tuple[np.ndarray, float, list]:
    sell_rules = SWING_SELL_RULES if mode == "swing" else DAY_SELL_RULES
    sell_hold  = SWING_SELL_HOLD  if mode == "swing" else DAY_SELL_HOLD
    ind_names  = IND_NAMES_SWING  if mode == "swing" else IND_NAMES_DAY

    POP = 200; GENS = 120; ELITE = 10
    TOURN_SIZE = 5; MUT_SIGMA = 0.12; MUT_PROB = 0.20

    # Dirichlet 初期化: DOE 効果に比例した alpha
    effects_arr = np.array([doe_effects.get(n, 0.1) for n in ind_names])
    effects_arr = np.maximum(effects_arr, 0.05)
    alpha = effects_arr / effects_arr.sum() * 8.
    np.random.seed(42)
    pop = np.random.dirichlet(alpha, POP) * 4.0  # (POP, 8), sum≈4

    hb = sell_hold[best_sell]
    thresholds_single = [best_thresh]

    def fitness_batch(wm_: np.ndarray) -> np.ndarray:
        shp = batch_sharpe(ticker_data, wm_, best_sell, thresholds_single, hb, 0, t_train_end)
        return np.where(np.isnan(shp[:, 0]), -np.inf, shp[:, 0])

    convergence = []
    for gen in range(GENS):
        fit = fitness_batch(pop)
        elite_idx = np.argsort(fit)[::-1][:ELITE]
        elite     = pop[elite_idx].copy()
        best_fit  = fit[elite_idx[0]]
        convergence.append(float(best_fit) if np.isfinite(best_fit) else 0.)

        if gen % 20 == 0 or gen == GENS - 1:
            print(f"    [GA] gen {gen+1:3d}/{GENS}  best_sharpe={best_fit:.4f}")

        # トーナメント選択
        parents = []
        for _ in range(POP - ELITE):
            cand_idx = np.random.choice(POP, TOURN_SIZE, replace=False)
            winner   = cand_idx[np.argmax(fit[cand_idx])]
            parents.append(pop[winner])

        # 算術交叉
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            a = np.random.uniform(0.3, 0.7)
            c1 = a * parents[i] + (1-a) * parents[i+1]
            c2 = (1-a) * parents[i] + a * parents[i+1]
            offspring.extend([c1, c2])
        if len(offspring) < POP - ELITE:
            offspring.append(parents[-1])
        offspring = np.array(offspring[:POP - ELITE])

        # Gaussian 突然変異
        mut_mask = np.random.rand(*offspring.shape) < MUT_PROB
        offspring += np.where(mut_mask, np.random.normal(0, MUT_SIGMA, offspring.shape), 0.)
        offspring  = np.clip(offspring, 0., 5.)

        pop = np.vstack([elite, offspring])

    final_fit = fitness_batch(pop)
    best_idx  = np.argmax(final_fit)
    best_w    = pop[best_idx]
    best_shp  = float(final_fit[best_idx]) if np.isfinite(final_fit[best_idx]) else 0.

    return best_w, best_shp, convergence

# ══════════════════════════════════════════════════════════════════════════════
# 10. 全体評価パイプライン
# ══════════════════════════════════════════════════════════════════════════════

def full_evaluation(ticker_data: dict, mode: str) -> dict:
    sell_rules = SWING_SELL_RULES if mode == "swing" else DAY_SELL_RULES
    sell_hold  = SWING_SELL_HOLD  if mode == "swing" else DAY_SELL_HOLD
    sell_ja    = SWING_SELL_JA    if mode == "swing" else DAY_SELL_JA
    ind_names  = IND_NAMES_SWING  if mode == "swing" else IND_NAMES_DAY

    # 全銘柄の最小バー数で分割
    T_min = min(len(td["ind_scores"][0]) for td in ticker_data.values())
    t_train_end = int(T_min * 0.80)
    print(f"  総バー数(最小): {T_min},  訓練終端: {t_train_end},  テスト: {t_train_end}〜{T_min}")

    # ── Step 1: DOE ──────────────────────────────────────────────────────────
    print("\n  ===== STEP 1: DOE (Taguchi L18) =====")
    doe_effects, doe_ranked = run_doe(ticker_data, mode, t_train_end)

    # ── Step 2: 事前プローブ (MC 1000サンプル) → best_sell, best_thresh ──────
    print("\n  ===== STEP 2: 事前プローブ (MC 1000) =====")
    np.random.seed(0)
    effects_arr = np.array([doe_effects.get(n, 0.1) for n in ind_names])
    effects_arr = np.maximum(effects_arr, 0.05)
    alpha = effects_arr / effects_arr.sum() * 8.
    probe_w = np.random.dirichlet(alpha, 1000) * 4.0

    best_sell = None; best_thresh = BUY_THRESHOLDS[0]; best_probe_shp = -np.inf
    for sname, rule in sell_rules.items():
        hb = sell_hold[sname]
        shp_mat = batch_sharpe(ticker_data, probe_w, sname, BUY_THRESHOLDS, hb, 0, t_train_end)
        mx = float(np.nanmax(shp_mat))
        if mx > best_probe_shp:
            best_probe_shp = mx
            idx = np.unravel_index(np.nanargmax(shp_mat), shp_mat.shape)
            best_thresh = BUY_THRESHOLDS[idx[1]]
            best_sell   = sname
        print(f"    {sname:<20s} best={mx:.4f}")

    print(f"  → 売りルール: {best_sell}  閾値: {best_thresh}")

    # ── Step 3: GA ───────────────────────────────────────────────────────────
    print("\n  ===== STEP 3: GA (pop=200, gen=120) =====")
    best_w, train_shp, convergence = run_ga(
        ticker_data, mode, doe_effects, best_sell, best_thresh, t_train_end)

    best_weights = {nm: round(float(best_w[i]), 4) for i, nm in enumerate(ind_names)}
    print(f"  [GA 完了] 訓練シャープ={train_shp:.4f}")
    print(f"  最適ウェイト: {best_weights}")

    # ── Step 4: MC最終評価 (5000サンプル, 訓練データ) ───────────────────────
    print("\n  ===== STEP 4: MC最終評価 (5000) =====")
    np.random.seed(99)
    mc_w = np.random.dirichlet(np.ones(8), 5000) * 4.0
    mc_w = np.vstack([mc_w, best_w.reshape(1, 8)])

    all_results = []
    hb = sell_hold[best_sell]
    shp_mat = batch_sharpe(ticker_data, mc_w, best_sell, BUY_THRESHOLDS, hb, 0, t_train_end)
    for wi in range(len(mc_w)):
        for ti, thr in enumerate(BUY_THRESHOLDS):
            sv = shp_mat[wi, ti]
            if np.isnan(sv): continue
            all_results.append({"weights": mc_w[wi], "thr": thr, "shp": float(sv)})

    all_results.sort(key=lambda x: x["shp"], reverse=True)
    top100 = all_results[:100]

    # 指標重み相関 (MC 上位 2000 件)
    if len(all_results) >= 10:
        mc_slice     = all_results[:min(2000, len(all_results))]
        weights_arr  = np.array([r["weights"] for r in mc_slice])
        sharpes_arr  = np.array([r["shp"]     for r in mc_slice])
        ind_corr: dict[str, float] = {}
        for i, nm in enumerate(ind_names):
            cv = float(np.corrcoef(weights_arr[:, i], sharpes_arr)[0, 1])
            ind_corr[nm] = round(cv if np.isfinite(cv) else 0., 4)
    else:
        ind_corr = {nm: 0. for nm in ind_names}

    # top100 中央値重みと重要度ランキング
    if top100:
        tw_arr = np.array([r["weights"] for r in top100])
        med_w  = {nm: round(float(np.median(tw_arr[:, i])), 3) for i, nm in enumerate(ind_names)}
    else:
        med_w = {nm: round(float(best_w[i]), 3) for i, nm in enumerate(ind_names)}
    weight_ranking = sorted(ind_names, key=lambda n: med_w[n], reverse=True)

    # ── Step 5: Walk-Forward テスト ──────────────────────────────────────────
    print("\n  ===== STEP 5: Walk-Forward テスト =====")
    wm_best = best_w.reshape(1, 8)
    thresholds_single = [best_thresh]
    hb = sell_hold[best_sell]

    # テストデータで評価
    test_shp_mat = batch_sharpe(ticker_data, wm_best, best_sell,
                                thresholds_single, hb, t_train_end, T_min)
    test_shp = float(test_shp_mat[0, 0]) if not np.isnan(test_shp_mat[0, 0]) else 0.

    # テスト期間の詳細統計
    test_n = 0; test_sum = 0.; test_wins = 0; test_sq = 0.; test_min = np.inf
    for td in ticker_data.values():
        ind  = td["ind_scores"][:, t_train_end:T_min].astype(np.float32)
        sout = td["sell_outcomes"][best_sell][t_train_end:T_min].astype(np.float32)
        vmask = td["vol_ok"][t_train_end:T_min]
        if "edge_bar" in td:
            vmask = vmask & ~td["edge_bar"][t_train_end:T_min]
        valid = ~np.isnan(sout) & vmask
        comp  = (wm_best @ ind)[0]
        mask  = (comp >= best_thresh) & valid
        trade_rets = sout[mask]
        test_n    += int(mask.sum())
        test_sum  += float(trade_rets.sum())
        test_wins += int((trade_rets > 0).sum())
        test_sq   += float((trade_rets**2).sum())
        if len(trade_rets) > 0:
            test_min = min(test_min, float(trade_rets.min()))

    test_avg_ret = (test_sum / test_n * 100) if test_n > 0 else 0.
    test_win_rate = (test_wins / test_n) if test_n > 0 else 0.
    test_max_dd   = test_min * 100 if test_n > 0 else 0.

    print(f"  テスト: N={test_n}  シャープ={test_shp:.4f}  "
          f"勝率={test_win_rate*100:.1f}%  avg={test_avg_ret:+.2f}%")

    # ── 売りルールランキング (訓練データ、GA最良重みで評価) ─────────────────
    print("  売りルールランキング評価中...")
    sell_stats = []
    for sname in sell_rules:
        hb_ = sell_hold[sname]
        sm  = batch_sharpe(ticker_data, wm_best, sname, BUY_THRESHOLDS, hb_, 0, t_train_end)
        if not np.all(np.isnan(sm)):
            best_ti    = int(np.nanargmax(sm[0]))
            best_thr_s = BUY_THRESHOLDS[best_ti]
            best_shp_s = float(sm[0, best_ti])
            avg_shp    = float(np.nanmean(sm[0]))
            dstats     = detailed_eval_single(ticker_data, best_w, sname,
                                               best_thr_s, hb_, 0, t_train_end)
        else:
            best_thr_s = BUY_THRESHOLDS[0]; best_shp_s = 0.; avg_shp = 0.
            dstats = {"win_rate": 0., "n_trades": 0, "avg_return": 0., "max_dd": 0.}
        sell_stats.append({
            "sell_rule":       sname,
            "sell_rule_ja":    sell_ja[sname],
            "best_sharpe":     round(best_shp_s, 4),
            "avg_sharpe":      round(avg_shp, 4),
            "best_win_rate":   dstats["win_rate"],
            "best_n_trades":   dstats["n_trades"],
            "best_avg_return": dstats["avg_return"],
            "best_weights":    best_weights,
            "best_threshold":  best_thr_s,
        })
    sell_stats.sort(key=lambda x: x["best_sharpe"], reverse=True)

    # ── top100 詳細評価 ───────────────────────────────────────────────────────
    print("  top100 詳細評価中...")
    hb = sell_hold[best_sell]
    top100_out = []
    for r in top100:
        ds = detailed_eval_single(ticker_data, r["weights"], best_sell,
                                   r["thr"], hb, 0, t_train_end)
        top100_out.append({
            "buy_weights":   {nm: round(float(r["weights"][i]), 3) for i, nm in enumerate(ind_names)},
            "buy_threshold": r["thr"],
            "sell_rule":     best_sell,
            "sharpe":        ds["sharpe"],
            "n_trades":      ds["n_trades"],
            "win_rate":      ds["win_rate"],
            "avg_return":    ds["avg_return"],
            "max_dd":        ds["max_dd"],
        })

    return {
        "version":       5,
        "generated_at":  time.strftime("%Y-%m-%d %H:%M"),
        "mode":          mode,
        "data_source":   "price_data_intraday.json (1h bars)",
        "n_tickers":     len(ticker_data),
        "tickers":       list(ticker_data.keys()),
        "n_evaluated":   len(all_results),
        "transaction_costs": {"US_pct": round(US_COST * 100, 2), "JP_pct": round(JP_COST * 100, 2)},
        "lookahead_bias_fixed": True,
        "entry_price":   "open of bar t+1 (signal at close of bar t)",
        "top100":        top100_out,
        "sell_rule_ranking": sell_stats,
        "indicator_weight_corr_with_sharpe": ind_corr,
        "top100_median_weights":   med_w,
        "top100_weight_ranking":   weight_ranking,
        "summary": {
            "best_sharpe":       round(train_shp, 4),
            "best_sharpe_train": round(train_shp, 4),
            "best_sell_rule":    best_sell,
            "best_weights":      best_weights,
            "best_threshold":    best_thresh,
        },
        "doe_results": {
            "n_experiments":      18,
            "indicator_effects":  {k: round(v, 4) for k, v in doe_effects.items()},
            "ranking":            doe_ranked,
        },
        "ga_results": {
            "population":        200,
            "generations":       120,
            "best_weights":      best_weights,
            "best_sharpe_train": round(train_shp, 4),
            "best_sell_rule":    best_sell,
            "best_threshold":    best_thresh,
            "convergence":       [round(v, 4) for v in convergence],
        },
        "walk_forward": {
            "train_ratio":    0.80,
            "train_bars":     t_train_end,
            "test_bars":      T_min - t_train_end,
            "train_sharpe":   round(train_shp, 4),
            "test_sharpe":    round(test_shp, 4),
            "test_win_rate":  round(test_win_rate, 4),
            "test_n_trades":  test_n,
            "test_avg_return": round(test_avg_ret, 3),
            "test_max_dd":    round(test_max_dd, 3),
        },
    }

# ══════════════════════════════════════════════════════════════════════════════
# 11. モード実行
# ══════════════════════════════════════════════════════════════════════════════

def run_mode(mode: str):
    sell_rules = SWING_SELL_RULES if mode == "swing" else DAY_SELL_RULES
    max_hold   = MAX_HOLD_BARS_SWING if mode == "swing" else MAX_HOLD_BARS_DAY
    out_file   = RESULTS_SWING if mode == "swing" else RESULTS_DAY
    mode_ja    = "スイングトレード" if mode == "swing" else "デイトレード"

    t0 = time.time()
    print("=" * 65)
    print(f"  バックテスト v3 — {mode_ja} ({mode}) mode")
    print(f"  DOE (L18) → GA (200×120) → Walk-Forward (80/20)")
    print("=" * 65)

    print("\n[1/3] データ読み込み中...")
    raw = load_data(mode)
    tickers = list(raw.keys())
    print(f"  {len(tickers)} 銘柄")

    print("\n[2/3] 指標計算・売りルール事前計算中...")
    ticker_data: dict[str, dict] = {}
    for tkr, td in raw.items():
        c, o = td["closes"], td["opens"]
        h, lo = td["highs"], td["lows"]
        cr = td["cost"]

        ind_scores   = compute_ind_scores(td, mode)
        vol_ok       = vol_ok_mask(td["volumes"])

        print(f"  {tkr}: 売りルール計算中 ({len(sell_rules)}件)...", end="", flush=True)
        sell_out = precompute_sell_outcomes(c, o, h, lo, cr, sell_rules, max_hold)
        print(" done")

        entry_dict: dict = {
            "ind_scores":    ind_scores,
            "sell_outcomes": sell_out,
            "vol_ok":        vol_ok,
        }
        if mode == "day":
            entry_dict["edge_bar"] = edge_bar_mask(td["dates"], len(c))

        ticker_data[tkr] = entry_dict

    print("\n[3/3] DOE → GA → Walk-Forward 実行中...")
    result = full_evaluation(ticker_data, mode)

    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    elapsed = time.time() - t0
    print(f"\n結果保存: {out_file}  ({out_file.stat().st_size // 1024} KB)")
    print(f"総実行時間: {elapsed:.0f}秒\n")

    wf = result["walk_forward"]
    ga = result["ga_results"]
    doe = result["doe_results"]
    print("=" * 65)
    print(f"  【{mode_ja} 最終結果】")
    print("=" * 65)
    print(f"  訓練シャープ: {wf['train_sharpe']:.4f}  テストシャープ: {wf['test_sharpe']:.4f}")
    print(f"  テスト勝率: {wf['test_win_rate']*100:.1f}%  "
          f"平均リターン: {wf['test_avg_return']:+.3f}%  "
          f"取引数: {wf['test_n_trades']}")
    print(f"  売りルール: {ga['best_sell_rule']}  閾値: {ga['best_threshold']}")
    print(f"\n  指標重要度 (DOE): {doe['ranking']}")
    ind_names = IND_NAMES_SWING if mode == "swing" else IND_NAMES_DAY
    print("  最適ウェイト:")
    for nm in ind_names:
        eff = doe["indicator_effects"].get(nm, 0.)
        w   = ga["best_weights"].get(nm, 0.)
        print(f"    {nm:<8s} ウェイト={w:.4f}  主効果={eff:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 12. エントリーポイント
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="バックテスト v3 (DOE→GA→Walk-Forward)")
    parser.add_argument("--swing", action="store_true", help="スイングトレードモード")
    parser.add_argument("--day",   action="store_true", help="デイトレードモード")
    parser.add_argument("--both",  action="store_true", help="両方実行")
    args = parser.parse_args()

    if not (args.swing or args.day or args.both):
        print("引数が必要です: --swing / --day / --both")
        parser.print_help()
        sys.exit(1)

    if args.both:
        run_mode("swing")
        run_mode("day")
    elif args.swing:
        run_mode("swing")
    else:
        run_mode("day")


if __name__ == "__main__":
    main()
