#!/usr/bin/env python3
"""
strategy_search.py v3 — DOE (Taguchi L18) → GA × 全売りルール → Walk-Forward 3折

【モード】
  --swing : スイング (1h足, RSI14/MACD12-26/BB20/EMA20-50/Aroon25/Stoch14/CCI20/ROC10)
  --day   : デイトレ (1h足, RSI9/MACD5-13/BB10/EMA9-21/Stoch5/ROC5/CCI14/VWAP偏差)
  --both  : 両方順番に実行

【パイプライン】
  1. DOE   (Taguchi L18 直交表) → 指標の主効果ランキング
  2. WF    (4折拡大窓, 訓練データ内) → 各Fold: 全売りルール × GA (pop=2000, gen=500)
  3. 最終GA (全売りルール × pop=2000 × gen=500) → best (sell, weights)
  4. MC最終評価 (5000サンプル) → top100
  5. ホールドアウト評価 (後ろ20%)
  ※ --both で約8時間 (swing/dayそれぞれ約4時間)

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
ARTIFACTS_DIR = HERE / "phase_artifacts"

MIN_TRADES     = 30   # 訓練期間の最小取引数
MIN_TRADES_OOS = 5    # OOS/ホールドアウト評価の最小取引数（短い窓でも記録する）
BARS_PER_YEAR  = 1500  # 1h bars/year (US+JP平均)

# スイング専用パラメータ
SWING_MIN_TRADES      = 20   # OOS最小トレード数（閾値引き上げで取引数が減るため緩和）
SWING_BUY_THRESHOLDS  = [60, 65, 70, 75, 80]  # 高閾値のみ: 厳選シグナルで勝率向上

# IC+Lift スイング専用パラメータ (GAの代替)
SWING_IC_WIN        = 400         # ローリングIC窓サイズ
SWING_IC_STEP       = 100         # ローリングICスライドステップ
SWING_IC_MIN_ICIR   = 0.03        # 最小IC情報比 (mean/std)
SWING_LIFT_SIGNAL   = 12.0        # シグナルゾーン下限 (0-25スコア中12以上)
SWING_LIFT_TARGET   = 0.005       # Lift計算ターゲットリターン (0.5%)
SWING_LIFT_MIN      = 1.02        # 最小Lift比率
SWING_IC_MIN_VALID  = 4           # 有効指標の最低採用数
SWING_IC_TREND_INDS = {"MA", "Aroon", "MACD"}  # トレンド系指標: 必ず1件以上採用
# 注: per-sell-rule IC+Lift — 各売りルールの実現損益でICを計算するため
# SWING_FORWARD_KEY は廃止 (run_ic_lift_sell_probe 内で sell_outcomes[sname] を直接使用)

# GA ハイパーパラメータ
GA_POP   = 2000   # 個体数
GA_GENS  = 500    # 世代数
GA_ELITE = 40     # エリート保存数
GA_TOURN = 7      # トーナメントサイズ
GA_SIGMA = 0.10   # Gaussian突然変異σ
GA_MPROB = 0.25   # 突然変異確率
GA_L2_LAMBDA = 0.5  # L2正則化強度: 1指標への過度な集中を抑制しOOS汎化性を向上

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
    # 時間固定保有は除外（方向性がなくシグナルと無関係に終了するため）
    # ストップ/トレール付きルールのみ残す → リスク管理が明確
    "target5_stop3":   {"type": "target_stop", "target": 5,  "stop": 3},
    "target10_stop5":  {"type": "target_stop", "target": 10, "stop": 5},
    "target15_stop5":  {"type": "target_stop", "target": 15, "stop": 5},
    "target20_stop7":  {"type": "target_stop", "target": 20, "stop": 7},
    "target15_stop7":  {"type": "target_stop", "target": 15, "stop": 7},
    "target25_stop10": {"type": "target_stop", "target": 25, "stop": 10},
    "trail_5pct":      {"type": "trailing",    "trail": 5},
    "trail_10pct":     {"type": "trailing",    "trail": 10},
    "trail_15pct":     {"type": "trailing",    "trail": 15},
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
    "hold_1b":        {"type": "hold",        "bars": 1},
    "target2_stop1":  {"type": "target_stop", "target": 2,  "stop": 1},
    "trail_1pct":     {"type": "trailing",    "trail": 1},
}

SWING_SELL_HOLD = {
    "target5_stop3": 35,  "target10_stop5": 50, "target15_stop5": 70,
    "target20_stop7": 80, "target15_stop7": 70, "target25_stop10": 100,
    "trail_5pct": 50,     "trail_10pct": 70,    "trail_15pct": 90,
}

DAY_SELL_HOLD = {
    "hold_2b": 2, "hold_4b": 4, "hold_6b": 6, "hold_8b": 8,
    "target3_stop2": 4,  "target5_stop3": 6,
    "target7_stop4": 8,  "target10_stop5": 10,
    "trail_2pct": 4,     "trail_3pct": 6,
    "hold_1b": 1, "target2_stop1": 2, "trail_1pct": 3,
}

SWING_SELL_JA = {
    "target5_stop3":   "利確+5% / ストップ-3%",
    "target10_stop5":  "利確+10% / ストップ-5%",
    "target15_stop5":  "利確+15% / ストップ-5%",
    "target20_stop7":  "利確+20% / ストップ-7%",
    "target15_stop7":  "利確+15% / ストップ-7%",
    "target25_stop10": "利確+25% / ストップ-10%",
    "trail_5pct":      "トレーリングストップ 5%",
    "trail_10pct":     "トレーリングストップ 10%",
    "trail_15pct":     "トレーリングストップ 15%",
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
    "hold_1b":       "固定保有 1h",
    "target2_stop1": "利確+2% / ストップ-1%",
    "trail_1pct":    "トレーリングストップ 1%",
}

IND_NAMES_SWING = ["RSI", "MACD", "BB", "MA", "Aroon", "Stoch", "CCI", "ROC"]
IND_NAMES_DAY   = ["RSI", "MACD", "BB", "MA", "Stoch", "ROC", "CCI", "VWAP"]

BUY_THRESHOLDS = [50, 55, 60, 65, 70, 75]

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

def _load_ticker_data(mode: str) -> tuple[dict, int, int]:
    """Load raw data, compute indicators & sell outcomes. Returns (ticker_data, T_min, t_holdout)."""
    sell_rules = SWING_SELL_RULES if mode == "swing" else DAY_SELL_RULES
    max_hold   = MAX_HOLD_BARS_SWING if mode == "swing" else MAX_HOLD_BARS_DAY

    raw = load_data(mode)
    ticker_data: dict = {}
    for tkr, td in raw.items():
        c, o   = td["closes"], td["opens"]
        h, lo  = td["highs"],  td["lows"]
        cr     = td["cost"]

        ind_scores = compute_ind_scores(td, mode)
        vol_ok     = vol_ok_mask(td["volumes"])

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

    T_min     = min(len(td["ind_scores"][0]) for td in ticker_data.values())
    t_holdout = int(T_min * 0.80)
    return ticker_data, T_min, t_holdout


def _wf_splits(T_min: int, t_holdout: int) -> list[tuple[int, int]]:
    """4-fold expanding WF splits within training data [0, t_holdout]."""
    return [
        (int(T_min * 0.30), int(T_min * 0.43)),
        (int(T_min * 0.43), int(T_min * 0.56)),
        (int(T_min * 0.56), int(T_min * 0.69)),
        (int(T_min * 0.69), t_holdout),
    ]


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
                 t_start: int, t_end: int,
                 min_trades: int = MIN_TRADES) -> np.ndarray:
    """
    wm: (N, 8) 重み行列
    返り値: (N, len(thresholds)) Sharpe
    min_trades: Sharpe計算に必要な最小取引数 (OOS評価ではMIN_TRADES_OOSを指定)
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
        ok  = n >= min_trades
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
# 9. 遺伝的アルゴリズム (1売りルール)
# ══════════════════════════════════════════════════════════════════════════════

def _make_alpha(doe_effects: dict, ind_names: list) -> np.ndarray:
    effects_arr = np.array([doe_effects.get(n, 0.1) for n in ind_names])
    effects_arr = np.maximum(effects_arr, 0.05)
    return effects_arr / effects_arr.sum() * 8.

def run_ga(ticker_data: dict, mode: str, doe_effects: dict,
           best_sell: str, best_thresh: int,
           t_train_end: int) -> tuple[np.ndarray, float, list]:
    sell_hold  = SWING_SELL_HOLD  if mode == "swing" else DAY_SELL_HOLD
    ind_names  = IND_NAMES_SWING  if mode == "swing" else IND_NAMES_DAY

    POP = GA_POP; GENS = GA_GENS; ELITE = GA_ELITE
    TOURN_SIZE = GA_TOURN; MUT_SIGMA = GA_SIGMA; MUT_PROB = GA_MPROB

    alpha = _make_alpha(doe_effects, ind_names)
    np.random.seed(42)
    pop = np.random.dirichlet(alpha, POP) * 4.0  # (POP, 8), sum≈4

    hb = sell_hold[best_sell]
    thresholds_single = [best_thresh]

    def fitness_batch(wm_: np.ndarray) -> np.ndarray:
        min_t = SWING_MIN_TRADES if mode == "swing" else MIN_TRADES
        shp = batch_sharpe(ticker_data, wm_, best_sell, thresholds_single, hb, 0, t_train_end,
                           min_trades=min_t)
        fit = np.where(np.isnan(shp[:, 0]), -np.inf, shp[:, 0])
        # L2正則化: 重みの集中を抑制しOOS汎化性を向上（指標数で正規化）
        n_ind = wm_.shape[1]
        return fit - GA_L2_LAMBDA * np.sum(wm_**2, axis=1) / n_ind

    convergence = []
    no_improve = 0
    prev_best  = -np.inf
    GA_PATIENCE = 60  # この世代数改善がなければ早期停止
    GA_MIN_GENS = 80  # 最低保証世代数

    for gen in range(GENS):
        fit = fitness_batch(pop)
        elite_idx = np.argsort(fit)[::-1][:ELITE]
        elite     = pop[elite_idx].copy()
        best_fit  = fit[elite_idx[0]]
        convergence.append(float(best_fit) if np.isfinite(best_fit) else 0.)

        if gen % 50 == 0 or gen == GENS - 1:
            print(f"    [GA] gen {gen+1:3d}/{GENS}  best_sharpe={best_fit:.4f}")

        # 早期停止: GA_MIN_GENS世代以降、GA_PATIENCE世代連続で0.01%未満の改善なら終了
        cur = float(best_fit) if np.isfinite(best_fit) else 0.
        if cur > prev_best * 1.0001:
            no_improve = 0
            prev_best  = cur
        else:
            no_improve += 1
        if gen >= GA_MIN_GENS and no_improve >= GA_PATIENCE:
            print(f"    [GA] 早期停止: gen {gen+1} (改善停滞 {GA_PATIENCE}世代)")
            break

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
# 10. 全売りルール GA 最適化
# ══════════════════════════════════════════════════════════════════════════════

def run_ga_all_sells(ticker_data: dict, mode: str, doe_effects: dict,
                     t_train_end: int,
                     checkpoint_dir: "Path | None" = None,
                     time_limit_s: "float | None" = None,
                     ) -> "tuple[dict | None, dict, bool]":
    """
    全売りルールそれぞれでGAを走らせ、最良の (sell, weights) を返す。

    checkpoint_dir: 指定時、売りルール完了ごとにJSONを保存し、再実行時に再開する。
    time_limit_s:   指定秒数を超えたら残りをスキップし、all_completed=False で返る。
    戻り値: (best_overall, all_ga_results, all_completed)
    """
    sell_rules = SWING_SELL_RULES if mode == "swing" else DAY_SELL_RULES
    sell_hold  = SWING_SELL_HOLD  if mode == "swing" else DAY_SELL_HOLD
    ind_names  = IND_NAMES_SWING  if mode == "swing" else IND_NAMES_DAY

    alpha = _make_alpha(doe_effects, ind_names)

    # ── チェックポイント読み込み ──────────────────────────────────────────────
    completed_sells: set[str] = set()
    all_ga_results: dict = {}
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prog_file = checkpoint_dir / "progress.json"
        if prog_file.exists():
            prog = json.loads(prog_file.read_text())
            completed_sells = set(prog.get("completed_sells", []))
            for sname in list(completed_sells):
                rf = checkpoint_dir / f"ga_{sname}.json"
                if rf.exists():
                    gr = json.loads(rf.read_text())
                    gr["weights"] = np.array(gr["weights_list"])
                    all_ga_results[sname] = gr
                else:
                    completed_sells.discard(sname)
            print(f"  チェックポイント復元: {len(completed_sells)}/{len(sell_rules)} 売りルール完了")

    # MC probe (300サンプル) で各売りルールの best_thresh を事前決定
    thresholds = SWING_BUY_THRESHOLDS if mode == "swing" else BUY_THRESHOLDS
    min_t_train = SWING_MIN_TRADES if mode == "swing" else MIN_TRADES
    np.random.seed(0)
    probe_w = np.random.dirichlet(alpha, 300) * 4.0
    best_thresh_per_sell: dict[str, int] = {}
    for sname in sell_rules:
        hb = sell_hold[sname]
        shp_mat = batch_sharpe(ticker_data, probe_w, sname, thresholds, hb, 0, t_train_end,
                               min_trades=min_t_train)
        col_means = np.nanmean(shp_mat, axis=0)
        best_ti = int(np.nanargmax(col_means)) if not np.all(np.isnan(col_means)) else 0
        best_thresh_per_sell[sname] = thresholds[best_ti]

    # 全売りルールでGA
    n_rules = len(sell_rules)
    start_time = time.time()
    time_over = False

    for rule_i, sname in enumerate(sell_rules, 1):
        if sname in completed_sells:
            print(f"  [{rule_i}/{n_rules}] {sname}: スキップ (チェックポイント済み)")
            continue

        # タイムバジェットチェック
        if time_limit_s is not None and (time.time() - start_time) > time_limit_s:
            print(f"  [{rule_i}/{n_rules}] タイムバジェット超過 — {sname} 以降をスキップ")
            time_over = True
            break

        thresh = best_thresh_per_sell[sname]
        print(f"  [{rule_i}/{n_rules}] 売りルール: {sname} (thresh={thresh})")
        best_w, train_shp, convergence = run_ga(
            ticker_data, mode, doe_effects, sname, thresh, t_train_end)

        ga_result: dict = {
            "weights":      best_w,
            "weights_list": best_w.tolist(),
            "sharpe":       train_shp,
            "threshold":    thresh,
            "convergence":  [round(v, 4) for v in convergence],
        }
        all_ga_results[sname] = ga_result

        # 売りルール完了ごとにチェックポイント保存
        if checkpoint_dir is not None:
            (checkpoint_dir / f"ga_{sname}.json").write_text(json.dumps({
                "weights_list": best_w.tolist(),
                "sharpe":       train_shp,
                "threshold":    thresh,
                "convergence":  [round(v, 4) for v in convergence],
            }))
            completed_sells.add(sname)
            (checkpoint_dir / "progress.json").write_text(json.dumps({
                "completed_sells": list(completed_sells),
                "last_updated":    time.strftime("%Y-%m-%d %H:%M"),
            }))

        best_shp_now = max(gr["sharpe"] for gr in all_ga_results.values())
        print(f"    → train_sharpe={train_shp:.4f}  (current best: {best_shp_now:.4f})")

    all_completed = (not time_over) and (len(all_ga_results) >= n_rules)

    if not all_ga_results:
        return None, {}, False

    # ベスト選択
    best_overall: dict | None = None
    for sname, gr in all_ga_results.items():
        w = gr["weights"] if isinstance(gr["weights"], np.ndarray) else np.array(gr["weights_list"])
        train_shp = gr["sharpe"]
        if best_overall is None or train_shp > best_overall["sharpe"]:
            best_overall = {
                "sell":        sname,
                "weights":     w,
                "sharpe":      train_shp,
                "threshold":   gr["threshold"],
                "convergence": gr["convergence"],
            }

    return best_overall, all_ga_results, all_completed

# ══════════════════════════════════════════════════════════════════════════════
# 9b. IC+Lift分析 (スイング専用, GAの代替)
# ══════════════════════════════════════════════════════════════════════════════

def pearson_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson相関係数 (NaN除外)。サンプル不足時は0.0を返す。"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = int(mask.sum())
    if n < 20:
        return 0.0
    xm = x[mask]; ym = y[mask]
    xc = xm - xm.mean(); yc = ym - ym.mean()
    denom = float(np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    return float(np.dot(xc, yc) / denom) if denom > 1e-10 else 0.0


def compute_swing_ic_lift(ticker_data: dict, t_train_end: int,
                          fwd_key: str = "hold_35b") -> dict:
    """
    スイング専用: IC情報比(ICIR) + Lift分析で有効指標と重みを決定する。

    fwd_key: sell_outcomes のキー。各売りルールの実現損益を使うことで
             「その売りルールに対して本当に効く指標」を選択できる。

    IC: 各指標スコア vs fwd_keyフォワードリターン のローリングPearson相関
        ICIR = mean(rolling_ICs) / std(rolling_ICs)
    Lift: P(ret>LIFT_TARGET | score>=SIGNAL) / P(ret>LIFT_TARGET) を銘柄ごとに計算

    採用条件: ICIR >= SWING_IC_MIN_ICIR AND Lift >= SWING_LIFT_MIN
    重み: 採用指標のICIRに比例 (sum=4.0)、非採用は0

    戻り値 dict:
      "valid_indicators": list[str]
      "weights":          np.ndarray (8,)
      "ic_scores":        dict[str, float]   ICIR値
      "lift_scores":      dict[str, float]   Lift比率
      "n_valid":          int
    """
    n_ind    = len(IND_NAMES_SWING)
    WIN      = SWING_IC_WIN
    STEP     = SWING_IC_STEP
    FWD_KEY  = fwd_key

    # ── per-ticker ローリングIC収集 ──────────────────────────────────────
    all_ics: dict[int, list[float]] = {i: [] for i in range(n_ind)}

    for td in ticker_data.values():
        ind  = td["ind_scores"][:, :t_train_end]               # (8, T)
        fret = td["sell_outcomes"][FWD_KEY][:t_train_end]      # (T,)
        vol  = td["vol_ok"][:t_train_end]
        mask = ~np.isnan(fret) & vol
        if mask.sum() < 30:
            continue
        ind_f  = ind[:, mask]   # (8, N)
        fret_f = fret[mask]     # (N,)
        N = fret_f.shape[0]

        if N < WIN:
            # 窓が取れないケースは全体でIC1点
            for i in range(n_ind):
                ic = pearson_ic(ind_f[i], fret_f)
                all_ics[i].append(ic)
        else:
            for start in range(0, N - WIN + 1, STEP):
                end = start + WIN
                fw  = fret_f[start:end]
                for i in range(n_ind):
                    ic = pearson_ic(ind_f[i, start:end], fw)
                    all_ics[i].append(ic)

    # ── ICIR計算 ─────────────────────────────────────────────────────────
    ic_scores: dict[str, float] = {}
    icir_vals: dict[int, float] = {}
    for i, name in enumerate(IND_NAMES_SWING):
        arr  = np.array(all_ics[i]) if all_ics[i] else np.array([0.0])
        mean = float(arr.mean())
        std  = float(arr.std()) + 1e-6
        icir = mean / std
        ic_scores[name] = round(icir, 4)
        icir_vals[i]    = icir

    # ── per-ticker Lift計算 ──────────────────────────────────────────────
    lift_scores: dict[str, float] = {}
    for i, name in enumerate(IND_NAMES_SWING):
        lifts_per_ticker: list[float] = []
        for td in ticker_data.values():
            ind_col = td["ind_scores"][i, :t_train_end]
            fret    = td["sell_outcomes"][FWD_KEY][:t_train_end]
            vol     = td["vol_ok"][:t_train_end]
            mask    = ~np.isnan(fret) & ~np.isnan(ind_col) & vol
            if mask.sum() < 20:
                continue
            fret_v = fret[mask]; ind_v = ind_col[mask]
            brate  = float((fret_v > SWING_LIFT_TARGET).mean())
            if brate < 1e-6:
                continue
            sig = ind_v >= SWING_LIFT_SIGNAL
            if sig.sum() < 5:
                continue
            crate = float((fret_v[sig] > SWING_LIFT_TARGET).mean())
            lifts_per_ticker.append(crate / brate)
        lift_scores[name] = round(float(np.mean(lifts_per_ticker)) if lifts_per_ticker else 1.0, 4)

    # ── 有効指標選択 ─────────────────────────────────────────────────────
    valid_names: list[str] = []
    for i, name in enumerate(IND_NAMES_SWING):
        passes_ic   = icir_vals[i] >= SWING_IC_MIN_ICIR
        passes_lift = lift_scores[name] >= SWING_LIFT_MIN
        tag = "✓" if (passes_ic and passes_lift) else "✗"
        note = ""
        if not passes_ic:   note += f" ICIR={icir_vals[i]:.3f}<{SWING_IC_MIN_ICIR}"
        if not passes_lift: note += f" Lift={lift_scores[name]:.3f}<{SWING_LIFT_MIN}"
        print(f"    {tag} {name}: ICIR={icir_vals[i]:.3f}  Lift={lift_scores[name]:.3f}{note}")
        if passes_ic and passes_lift:
            valid_names.append(name)

    # 有効指標がMIN_VALID件未満の場合、正ICIRの上位指標で補充
    if len(valid_names) < SWING_IC_MIN_VALID:
        candidates = sorted(
            [i for i in range(n_ind) if icir_vals[i] > 0 and IND_NAMES_SWING[i] not in valid_names],
            key=lambda i: icir_vals[i], reverse=True
        )
        needed = SWING_IC_MIN_VALID - len(valid_names)
        added = [IND_NAMES_SWING[i] for i in candidates[:needed]]
        if added:
            print(f"  補充: {added} (有効指標{len(valid_names)}件 → {SWING_IC_MIN_VALID}件確保)")
            valid_names = valid_names + added
        if not valid_names:
            valid_names = list(IND_NAMES_SWING)

    # ── トレンド指標を必ず1件以上含める ──────────────────────────────────
    # 下落トレンド中の逆張り連発を防ぐためMA/Aroon/MACDを最低1件確保
    has_trend = any(n in SWING_IC_TREND_INDS for n in valid_names)
    if not has_trend:
        trend_candidates = sorted(
            [i for i, n in enumerate(IND_NAMES_SWING)
             if n in SWING_IC_TREND_INDS and n not in valid_names],
            key=lambda i: icir_vals[i], reverse=True
        )
        if trend_candidates:
            added = IND_NAMES_SWING[trend_candidates[0]]
            valid_names.append(added)
            print(f"  トレンドフィルタ補充: {added} (ICIR={icir_vals[trend_candidates[0]]:.3f})")

    # ── 重み: ICIR比例 (sum=4.0) ─────────────────────────────────────────
    raw_w = np.zeros(n_ind)
    for i, name in enumerate(IND_NAMES_SWING):
        if name in valid_names:
            raw_w[i] = max(icir_vals[i], 0.0)
    total = raw_w.sum()
    weights = raw_w / total * 4.0 if total > 1e-6 else np.ones(n_ind) * (4.0 / n_ind)

    return {
        "valid_indicators": valid_names,
        "weights":          weights,
        "ic_scores":        ic_scores,
        "lift_scores":      lift_scores,
        "n_valid":          len(valid_names),
    }


def run_ic_lift_sell_probe(ticker_data: dict, t_train_end: int,
                            checkpoint_dir: "Path | None" = None,
                            time_limit_s: "float | None" = None,
                            ) -> "tuple[dict | None, dict, bool]":
    """
    Per-sell-rule IC+Lift: 各売りルールの実現損益でICを計算し、
    そのルールに最適な重みを導出してグリッドサーチする。

    GAの代替。チェックポイント機能付き。

    戻り値: (best_result, all_sell_results_compat, all_completed)
      best_result: {"sell", "threshold", "weights", "sharpe", "convergence"}
      all_sell_results_compat: run_ga_all_sells互換形式 (assembleで共通利用可能)
    """
    sell_rules = SWING_SELL_RULES
    sell_hold  = SWING_SELL_HOLD
    thresholds = SWING_BUY_THRESHOLDS
    min_t      = SWING_MIN_TRADES
    n_rules    = len(sell_rules)

    # チェックポイント読み込み
    completed_sells: set[str] = set()
    sell_results: dict = {}
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prog_file = checkpoint_dir / "progress.json"
        if prog_file.exists():
            prog = json.loads(prog_file.read_text())
            completed_sells = set(prog.get("completed_sells", []))
            for sname in list(completed_sells):
                rf = checkpoint_dir / f"sell_{sname}.json"
                if rf.exists():
                    sell_results[sname] = json.loads(rf.read_text())
                else:
                    completed_sells.discard(sname)
            if completed_sells:
                print(f"  チェックポイント復元: {len(completed_sells)}/{n_rules} 売りルール完了")

    wm_fixed   = weights.reshape(1, 8)
    start_time = time.time()
    time_over  = False

    for rule_i, sname in enumerate(sell_rules, 1):
        if sname in completed_sells:
            print(f"  [{rule_i}/{n_rules}] {sname}: スキップ (チェックポイント済み)")
            continue

        if time_limit_s is not None and (time.time() - start_time) > time_limit_s:
            print(f"  [{rule_i}/{n_rules}] タイムバジェット超過")
            time_over = True
            break

        # per-sell-rule IC+Lift: このルールの実現損益でICを計算し最適重みを導出
        ic_lift_rule = compute_swing_ic_lift(ticker_data, t_train_end, fwd_key=sname)
        w_rule = ic_lift_rule["weights"]
        wm_rule = w_rule.reshape(1, 8)

        hb      = sell_hold[sname]
        shp_row = batch_sharpe(
            ticker_data, wm_rule, sname, thresholds, hb, 0, t_train_end,
            min_trades=min_t)[0]  # (len(thresholds),)

        if np.all(np.isnan(shp_row)):
            best_ti = 0; best_shp = -np.inf
        else:
            best_ti  = int(np.nanargmax(shp_row))
            best_shp = float(shp_row[best_ti])

        result = {
            "sharpe":    round(best_shp, 4),
            "threshold": thresholds[best_ti],
            "weights":   w_rule.tolist(),
            "ic_lift":   {
                "valid_indicators": ic_lift_rule["valid_indicators"],
                "ic_scores":        ic_lift_rule["ic_scores"],
                "lift_scores":      ic_lift_rule["lift_scores"],
                "n_valid":          ic_lift_rule["n_valid"],
            },
        }
        sell_results[sname] = result

        if checkpoint_dir is not None:
            (checkpoint_dir / f"sell_{sname}.json").write_text(json.dumps(result))
            completed_sells.add(sname)
            (checkpoint_dir / "progress.json").write_text(json.dumps({
                "completed_sells": list(completed_sells),
                "last_updated":    time.strftime("%Y-%m-%d %H:%M"),
            }))

        print(f"  [{rule_i}/{n_rules}] {sname}: sharpe={best_shp:.4f}  thresh={thresholds[best_ti]}")

    all_completed = (not time_over) and (len(sell_results) >= n_rules)

    if not sell_results:
        return None, {}, False

    # ベスト売りルール選択 (各ルール固有の重みを使用)
    finite = {s: r for s, r in sell_results.items() if np.isfinite(r["sharpe"])}
    best_sname = max(finite, key=lambda s: finite[s]["sharpe"]) if finite else next(iter(sell_results))
    best_r = sell_results[best_sname]
    best_result = {
        "sell":        best_sname,
        "threshold":   best_r["threshold"],
        "weights":     np.array(best_r["weights"]),
        "sharpe":      best_r["sharpe"],
        "convergence": [],
    }

    # run_ga_all_sells互換形式 (assemble側で共通利用)
    compat = {
        sname: {
            "sharpe":     r["sharpe"],
            "threshold":  r["threshold"],
            "weights":    np.array(r["weights"]),
            "convergence": [],
        }
        for sname, r in sell_results.items()
    }

    # ベスト売りルールのIC+Lift詳細
    best_ic_lift = best_r.get("ic_lift")

    return best_result, compat, all_completed, best_ic_lift


# ══════════════════════════════════════════════════════════════════════════════
# 11. フェーズ実行 (16ジョブ並列用)
# ══════════════════════════════════════════════════════════════════════════════

def run_phase_doe(mode: str) -> None:
    """Phase DOE: L18直交表実験 → phase_artifacts/doe_{mode}.json"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[PHASE: DOE] mode={mode}")

    ticker_data, T_min, t_holdout = _load_ticker_data(mode)
    doe_effects, doe_ranked = run_doe(ticker_data, mode, t_holdout)

    artifact = {
        "mode":        mode,
        "T_min":       T_min,
        "t_holdout":   t_holdout,
        "doe_effects": {k: round(v, 6) for k, v in doe_effects.items()},
        "doe_ranked":  doe_ranked,
    }
    out_path = ARTIFACTS_DIR / f"doe_{mode}.json"
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"[PHASE: DOE] 保存: {out_path}")


def run_phase_wf_fold(mode: str, fold_i: int,
                      time_limit_s: "float | None" = None) -> None:
    """Phase WF-Fold: 1fold分の全売りルール×GA → phase_artifacts/wf_{mode}_fold_{fold_i}.json
    time_limit_s: 指定秒数を超えたらチェックポイント保存して exit(1)。
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    doe_path = ARTIFACTS_DIR / f"doe_{mode}.json"
    if not doe_path.exists():
        print(f"ERROR: {doe_path} が見つかりません"); sys.exit(1)
    doe_art     = json.loads(doe_path.read_text())
    T_min       = doe_art["T_min"]
    t_holdout   = doe_art["t_holdout"]
    doe_effects = doe_art["doe_effects"]

    splits = _wf_splits(T_min, t_holdout)
    if fold_i < 0 or fold_i >= len(splits):
        print(f"ERROR: fold_i={fold_i} 範囲外 (0〜{len(splits)-1})"); sys.exit(1)

    t_train_f, t_oos_end_f = splits[fold_i]
    print(f"[PHASE: WF Fold {fold_i}] mode={mode}  train=[0,{t_train_f}] oos=[{t_train_f},{t_oos_end_f}]")
    if time_limit_s:
        print(f"  タイムリミット: {time_limit_s/60:.0f}分")

    ticker_data, _, _ = _load_ticker_data(mode)
    sell_hold = SWING_SELL_HOLD if mode == "swing" else DAY_SELL_HOLD
    ind_names = IND_NAMES_SWING if mode == "swing" else IND_NAMES_DAY

    checkpoint_dir = ARTIFACTS_DIR / f"checkpoint_{mode}_fold_{fold_i}"

    if mode == "swing":
        print(f"  [per-sell-rule IC+Lift] 全売りルールを検証中 (訓練 0〜{t_train_f} バー) ...")
        best_fold, all_ga_results, all_completed, ic_lift_details = run_ic_lift_sell_probe(
            ticker_data, t_train_f,
            checkpoint_dir=checkpoint_dir, time_limit_s=time_limit_s)
    else:
        best_fold, all_ga_results, all_completed = run_ga_all_sells(
            ticker_data, mode, doe_effects, t_train_f,
            checkpoint_dir=checkpoint_dir, time_limit_s=time_limit_s)
        ic_lift_details = None

    if not all_completed:
        print(f"[PHASE: WF Fold {fold_i}] 未完了: タイムバジェット超過。"
              f"チェックポイントは {checkpoint_dir} に保存済み。")
        print("  → GitHub Actions で 'Re-run failed jobs' を実行すると続きから再開します。")
        sys.exit(1)

    hb_f     = sell_hold[best_fold["sell"]]
    oos_stats = detailed_eval_single(ticker_data, best_fold["weights"],
                                     best_fold["sell"], best_fold["threshold"],
                                     hb_f, t_train_f, t_oos_end_f)
    wm_f      = best_fold["weights"].reshape(1, 8)
    oos_shp_m = batch_sharpe(ticker_data, wm_f, best_fold["sell"],
                              [best_fold["threshold"]], hb_f, t_train_f, t_oos_end_f,
                              min_trades=MIN_TRADES_OOS)
    oos_shp_f = float(oos_shp_m[0, 0]) if not np.isnan(oos_shp_m[0, 0]) else 0.

    all_sell_oos: dict = {}
    for sname, gr in all_ga_results.items():
        hb_ = sell_hold[sname]
        wm_ = gr["weights"].reshape(1, 8)
        sm  = batch_sharpe(ticker_data, wm_, sname, [gr["threshold"]], hb_, t_train_f, t_oos_end_f,
                           min_trades=MIN_TRADES_OOS)
        os_ = float(sm[0, 0]) if not np.isnan(sm[0, 0]) else 0.
        ds  = detailed_eval_single(ticker_data, gr["weights"], sname, gr["threshold"],
                                    hb_, t_train_f, t_oos_end_f)
        all_sell_oos[sname] = {
            "oos_sharpe":    round(os_, 4),
            "oos_n_trades":  ds["n_trades"],
            "oos_win_rate":  ds["win_rate"],
            "oos_avg_return": ds["avg_return"],
        }

    artifact = {
        "mode":           mode,
        "fold":           fold_i,
        "t_train_f":      t_train_f,
        "t_oos_end_f":    t_oos_end_f,
        "best_sell":      best_fold["sell"],
        "best_threshold": best_fold["threshold"],
        "best_weights":   best_fold["weights"].tolist(),
        "train_sharpe":   round(float(best_fold["sharpe"]), 4),
        "oos_sharpe":     round(oos_shp_f, 4),
        "oos_n_trades":   oos_stats["n_trades"],
        "oos_win_rate":   oos_stats["win_rate"],
        "oos_avg_return": oos_stats["avg_return"],
        "oos_max_dd":     oos_stats["max_dd"],
        "all_sell_oos":   all_sell_oos,
        "ic_lift_details": ic_lift_details,
    }
    out_path = ARTIFACTS_DIR / f"wf_{mode}_fold_{fold_i}.json"
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"[PHASE: WF Fold {fold_i}] 保存: {out_path}  "
          f"train={best_fold['sharpe']:.4f}  oos={oos_shp_f:.4f}  N={oos_stats['n_trades']}")


def run_phase_final(mode: str, time_limit_s: "float | None" = None) -> None:
    """Phase Final: 訓練全体で全売りルール×GA → phase_artifacts/final_{mode}.json
    time_limit_s: 指定秒数を超えたらチェックポイント保存して exit(1)。
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    doe_path = ARTIFACTS_DIR / f"doe_{mode}.json"
    if not doe_path.exists():
        print(f"ERROR: {doe_path} が見つかりません"); sys.exit(1)
    doe_art     = json.loads(doe_path.read_text())
    t_holdout   = doe_art["t_holdout"]
    doe_effects = doe_art["doe_effects"]

    print(f"[PHASE: FINAL] mode={mode}  t_holdout={t_holdout}")
    if time_limit_s:
        print(f"  タイムリミット: {time_limit_s/60:.0f}分")
    ticker_data, _, _ = _load_ticker_data(mode)

    checkpoint_dir = ARTIFACTS_DIR / f"checkpoint_{mode}_final"

    if mode == "swing":
        print(f"  [per-sell-rule IC+Lift] 全売りルールを検証中 (訓練 0〜{t_holdout} バー) ...")
        best_overall, all_ga_results, all_completed, ic_lift_details = run_ic_lift_sell_probe(
            ticker_data, t_holdout,
            checkpoint_dir=checkpoint_dir, time_limit_s=time_limit_s)
    else:
        best_overall, all_ga_results, all_completed = run_ga_all_sells(
            ticker_data, mode, doe_effects, t_holdout,
            checkpoint_dir=checkpoint_dir, time_limit_s=time_limit_s)
        ic_lift_details = None

    if not all_completed:
        print(f"[PHASE: FINAL] 未完了: タイムバジェット超過。"
              f"チェックポイントは {checkpoint_dir} に保存済み。")
        print("  → GitHub Actions で 'Re-run failed jobs' を実行すると続きから再開します。")
        sys.exit(1)

    artifact = {
        "mode":           mode,
        "t_holdout":      t_holdout,
        "best_sell":      best_overall["sell"],
        "best_threshold": best_overall["threshold"],
        "best_weights":   best_overall["weights"].tolist(),
        "train_sharpe":   round(float(best_overall["sharpe"]), 4),
        "convergence":    [round(v, 4) for v in best_overall["convergence"]],
        "all_ga_results": {
            sname: {
                "sharpe":     round(float(gr["sharpe"]), 4),
                "threshold":  gr["threshold"],
                "weights":    gr["weights"].tolist() if isinstance(gr["weights"], np.ndarray) else gr["weights"],
                "convergence": [round(v, 4) for v in gr["convergence"]],
            }
            for sname, gr in all_ga_results.items()
        },
        "ic_lift_details": ic_lift_details,
    }
    out_path = ARTIFACTS_DIR / f"final_{mode}.json"
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"[PHASE: FINAL] 保存: {out_path}  sell={best_overall['sell']}  "
          f"thresh={best_overall['threshold']}  train={best_overall['sharpe']:.4f}")


def run_phase_assemble(mode: str) -> None:
    """Phase Assemble: 全アーティファクト集約 → strategy_results_{mode}.json"""
    sell_rules = SWING_SELL_RULES if mode == "swing" else DAY_SELL_RULES
    sell_hold  = SWING_SELL_HOLD  if mode == "swing" else DAY_SELL_HOLD
    sell_ja    = SWING_SELL_JA    if mode == "swing" else DAY_SELL_JA
    ind_names  = IND_NAMES_SWING  if mode == "swing" else IND_NAMES_DAY
    out_file   = RESULTS_SWING if mode == "swing" else RESULTS_DAY

    print(f"[PHASE: ASSEMBLE] mode={mode}")

    doe_path   = ARTIFACTS_DIR / f"doe_{mode}.json"
    final_path = ARTIFACTS_DIR / f"final_{mode}.json"
    for p in [doe_path, final_path]:
        if not p.exists():
            print(f"ERROR: {p} が見つかりません"); sys.exit(1)

    doe_art   = json.loads(doe_path.read_text())
    final_art = json.loads(final_path.read_text())
    T_min       = doe_art["T_min"]
    t_holdout   = doe_art["t_holdout"]
    doe_effects = doe_art["doe_effects"]
    doe_ranked  = doe_art["doe_ranked"]

    splits     = _wf_splits(T_min, t_holdout)
    n_wf_folds = len(splits)

    wf_folds: list[dict] = []
    fold_weights_list: list[np.ndarray] = []
    for fold_i in range(n_wf_folds):
        wf_path = ARTIFACTS_DIR / f"wf_{mode}_fold_{fold_i}.json"
        if not wf_path.exists():
            print(f"ERROR: {wf_path} が見つかりません"); sys.exit(1)
        wa = json.loads(wf_path.read_text())
        wf_folds.append({
            "fold":           fold_i,
            "train_bars":     wa["t_train_f"],
            "oos_bars":       wa["t_oos_end_f"] - wa["t_train_f"],
            "best_sell":      wa["best_sell"],
            "best_threshold": wa["best_threshold"],
            "train_sharpe":   wa["train_sharpe"],
            "oos_sharpe":     wa["oos_sharpe"],
            "oos_n_trades":   wa["oos_n_trades"],
            "oos_win_rate":   wa["oos_win_rate"],
            "oos_avg_return": wa["oos_avg_return"],
            "oos_max_dd":     wa["oos_max_dd"],
            "all_sell_oos":   wa.get("all_sell_oos", {}),
        })
        fold_weights_list.append(np.array(wa["best_weights"]))

    avg_oos_sharpe = float(np.mean([f["oos_sharpe"] for f in wf_folds]))
    oos_arr = np.array([f["oos_sharpe"] for f in wf_folds])
    wf_stability = float(1. - np.std(oos_arr) / (abs(np.mean(oos_arr)) + 1e-6))

    best_sell   = final_art["best_sell"]
    best_thresh = final_art["best_threshold"]
    best_w      = np.array(final_art["best_weights"])
    train_shp   = final_art["train_sharpe"]
    convergence = final_art["convergence"]
    all_ga_results_raw = final_art["all_ga_results"]
    best_weights = {nm: round(float(best_w[i]), 4) for i, nm in enumerate(ind_names)}

    # ウェイト信頼区間 (WF fold 最良重み + final 最良重み)
    fold_weights_list.append(best_w)
    fold_weights_arr = np.stack(fold_weights_list, axis=0)
    weight_ci: dict = {}
    for i, nm in enumerate(ind_names):
        col  = fold_weights_arr[:, i]
        mean = float(np.mean(col)); std = float(np.std(col))
        cv   = std / (abs(mean) + 1e-6)
        weight_ci[nm] = {"mean": round(mean, 4), "std": round(std, 4), "cv": round(cv, 4)}

    print("  データ読み込み・指標計算中...")
    ticker_data, _, _ = _load_ticker_data(mode)

    hb      = sell_hold[best_sell]
    wm_best = best_w.reshape(1, 8)

    print("  MC最終評価 (5000) ...")
    np.random.seed(99)
    alpha = _make_alpha(doe_effects, ind_names)
    mc_w  = np.random.dirichlet(alpha, 5000) * 4.0
    mc_w  = np.vstack([mc_w, best_w.reshape(1, 8)])

    assemble_thresholds = SWING_BUY_THRESHOLDS if mode == "swing" else BUY_THRESHOLDS
    all_results: list[dict] = []
    shp_mat = batch_sharpe(ticker_data, mc_w, best_sell, assemble_thresholds, hb, 0, t_holdout)
    for wi in range(len(mc_w)):
        for ti, thr in enumerate(assemble_thresholds):
            sv = shp_mat[wi, ti]
            if np.isnan(sv): continue
            all_results.append({"weights": mc_w[wi], "thr": thr, "shp": float(sv)})
    all_results.sort(key=lambda x: x["shp"], reverse=True)
    top100 = all_results[:100]

    if len(all_results) >= 10:
        mc_slice    = all_results[:min(2000, len(all_results))]
        weights_arr = np.array([r["weights"] for r in mc_slice])
        sharpes_arr = np.array([r["shp"]     for r in mc_slice])
        ind_corr = {nm: round(float(np.corrcoef(weights_arr[:, i], sharpes_arr)[0, 1])
                               if np.isfinite(np.corrcoef(weights_arr[:, i], sharpes_arr)[0, 1]) else 0., 4)
                    for i, nm in enumerate(ind_names)}
    else:
        ind_corr = {nm: 0. for nm in ind_names}

    if top100:
        tw_arr = np.array([r["weights"] for r in top100])
        med_w  = {nm: round(float(np.median(tw_arr[:, i])), 3) for i, nm in enumerate(ind_names)}
    else:
        med_w = {nm: round(float(best_w[i]), 3) for i, nm in enumerate(ind_names)}
    weight_ranking = sorted(ind_names, key=lambda n: med_w[n], reverse=True)

    print("  ホールドアウト評価...")
    test_shp_m = batch_sharpe(ticker_data, wm_best, best_sell, [best_thresh], hb, t_holdout, T_min,
                              min_trades=MIN_TRADES_OOS)
    test_shp   = float(test_shp_m[0, 0]) if not np.isnan(test_shp_m[0, 0]) else 0.
    test_stats = detailed_eval_single(ticker_data, best_w, best_sell, best_thresh, hb, t_holdout, T_min)
    overfit_ratio = round(test_shp / train_shp, 4) if abs(train_shp) > 1e-6 else 0.

    print("  売りルールランキング評価中...")
    sell_stats: list[dict] = []
    for sname in sell_rules:
        hb_ = sell_hold[sname]
        sm  = batch_sharpe(ticker_data, wm_best, sname, assemble_thresholds, hb_, 0, t_holdout)
        if not np.all(np.isnan(sm)):
            best_ti    = int(np.nanargmax(sm[0]))
            best_thr_s = assemble_thresholds[best_ti]
            best_shp_s = float(sm[0, best_ti])
            avg_shp    = float(np.nanmean(sm[0]))
            dstats     = detailed_eval_single(ticker_data, best_w, sname, best_thr_s, hb_, 0, t_holdout)
        else:
            best_thr_s = assemble_thresholds[0]; best_shp_s = 0.; avg_shp = 0.
            dstats = {"win_rate": 0., "n_trades": 0, "avg_return": 0., "max_dd": 0.}
        sell_stats.append({
            "sell_rule": sname, "sell_rule_ja": sell_ja[sname],
            "best_sharpe": round(best_shp_s, 4), "avg_sharpe": round(avg_shp, 4),
            "best_win_rate": dstats["win_rate"], "best_n_trades": dstats["n_trades"],
            "best_avg_return": dstats["avg_return"], "best_weights": best_weights,
            "best_threshold": best_thr_s,
        })
    sell_stats.sort(key=lambda x: x["best_sharpe"], reverse=True)

    print("  top100 詳細評価中...")
    top100_out: list[dict] = []
    for r in top100:
        ds = detailed_eval_single(ticker_data, r["weights"], best_sell, r["thr"], hb, 0, t_holdout)
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

    ga_all_sells_summary = {}
    for sname, gr in all_ga_results_raw.items():
        w = gr["weights"] if isinstance(gr["weights"], list) else gr["weights"].tolist()
        ga_all_sells_summary[sname] = {
            "sharpe":    round(float(gr["sharpe"]), 4),
            "threshold": gr["threshold"],
            "weights":   {nm: round(float(w[i]), 4) for i, nm in enumerate(ind_names)},
        }

    # IC+Lift詳細 (swing のみ)
    ic_lift_results = None
    if mode == "swing":
        fold_ic_details = []
        for wa in [json.loads((ARTIFACTS_DIR / f"wf_{mode}_fold_{fi}.json").read_text())
                   for fi in range(n_wf_folds)]:
            if wa.get("ic_lift_details"):
                fold_ic_details.append(wa["ic_lift_details"])
        final_ic = final_art.get("ic_lift_details")
        if fold_ic_details or final_ic:
            ic_lift_results = {
                "method":       "ic_lift",
                "fold_details": fold_ic_details,
                "final":        final_ic,
            }

    result = {
        "version":       7,
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
        "weight_confidence_intervals": weight_ci,
        "summary": {
            "best_sharpe":       round(train_shp, 4),
            "best_sharpe_train": round(train_shp, 4),
            "avg_oos_sharpe":    round(avg_oos_sharpe, 4),
            "wf_stability":      round(wf_stability, 4),
            "overfit_ratio":     overfit_ratio,
            "best_sell_rule":    best_sell,
            "best_weights":      best_weights,
            "best_threshold":    best_thresh,
        },
        "doe_results": {
            "n_experiments":     18,
            "indicator_effects": {k: round(v, 4) for k, v in doe_effects.items()},
            "ranking":           doe_ranked,
        },
        "ga_results": {
            "population":             GA_POP if mode == "day" else 0,
            "generations":            GA_GENS if mode == "day" else 0,
            "method":                 "ga" if mode == "day" else "ic_lift",
            "n_sell_rules_optimized": len(sell_rules),
            "all_sell_rules":         ga_all_sells_summary,
            "best_weights":           best_weights,
            "best_sharpe_train":      round(train_shp, 4),
            "best_sell_rule":         best_sell,
            "best_threshold":         best_thresh,
            "convergence":            [round(v, 4) for v in convergence],
        },
        "ic_lift_results": ic_lift_results,
        "walk_forward": {
            "n_folds":         n_wf_folds,
            "folds":           wf_folds,
            "avg_oos_sharpe":  round(avg_oos_sharpe, 4),
            "wf_stability":    round(wf_stability, 4),
            "overfit_ratio":   overfit_ratio,
            "train_ratio":     0.80,
            "train_bars":      t_holdout,
            "test_bars":       T_min - t_holdout,
            "train_sharpe":    round(train_shp, 4),
            "test_sharpe":     round(test_shp, 4),
            "test_win_rate":   round(test_stats["win_rate"], 4),
            "test_n_trades":   test_stats["n_trades"],
            "test_avg_return": round(test_stats["avg_return"], 3),
            "test_max_dd":     round(test_stats["max_dd"], 3),
        },
    }
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[PHASE: ASSEMBLE] 保存完了: {out_file}  ({out_file.stat().st_size // 1024} KB)")
    print(f"  avg_oos_sharpe={avg_oos_sharpe:.4f}  wf_stability={wf_stability:.4f}  "
          f"overfit_ratio={overfit_ratio:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. 全体評価パイプライン
# ══════════════════════════════════════════════════════════════════════════════

def full_evaluation(ticker_data: dict, mode: str) -> dict:
    sell_rules = SWING_SELL_RULES if mode == "swing" else DAY_SELL_RULES
    sell_hold  = SWING_SELL_HOLD  if mode == "swing" else DAY_SELL_HOLD
    sell_ja    = SWING_SELL_JA    if mode == "swing" else DAY_SELL_JA
    ind_names  = IND_NAMES_SWING  if mode == "swing" else IND_NAMES_DAY

    # 全銘柄の最小バー数でホールドアウト分割
    T_min     = min(len(td["ind_scores"][0]) for td in ticker_data.values())
    t_holdout = int(T_min * 0.80)  # 後ろ20%を真のホールドアウトとして確保
    print(f"  総バー数(最小): {T_min},  ホールドアウト: {t_holdout}〜{T_min} ({T_min - t_holdout}バー)")

    # ── Step 1: DOE (訓練データ全体 0〜t_holdout) ─────────────────────────────
    print("\n  ===== STEP 1: DOE (Taguchi L18) =====")
    doe_effects, doe_ranked = run_doe(ticker_data, mode, t_holdout)

    alpha = _make_alpha(doe_effects, ind_names)

    # ── Step 2: Walk-Forward バリデーション (4 folds, 拡大窓, 各Fold全売りルール×GA) ─
    print("\n  ===== STEP 2: Walk-Forward バリデーション (4 folds × 全売りルール) =====")
    # 各Foldで全売りルールのGAを走らせる → (4+1)×n_sells GArun = 60回/mode ≈ 4時間/mode
    wf_splits = [
        (int(T_min * 0.30), int(T_min * 0.43)),
        (int(T_min * 0.43), int(T_min * 0.56)),
        (int(T_min * 0.56), int(T_min * 0.69)),
        (int(T_min * 0.69), t_holdout),
    ]
    n_wf_folds = len(wf_splits)

    wf_folds = []
    for fold_i, (t_train_f, t_oos_end_f) in enumerate(wf_splits):
        print(f"\n  --- WF Fold {fold_i+1}/{n_wf_folds}: train=[0,{t_train_f}] oos=[{t_train_f},{t_oos_end_f}] ---")

        # 全売りルールでGA → 最良を選択
        best_fold_overall, _, _ = run_ga_all_sells(ticker_data, mode, doe_effects, t_train_f)

        # OOS評価
        hb_f = sell_hold[best_fold_overall["sell"]]
        wm_f = best_fold_overall["weights"].reshape(1, 8)
        oos_shp_mat = batch_sharpe(ticker_data, wm_f, best_fold_overall["sell"],
                                    [best_fold_overall["threshold"]], hb_f, t_train_f, t_oos_end_f,
                                    min_trades=MIN_TRADES_OOS)
        oos_shp_f = float(oos_shp_mat[0, 0]) if not np.isnan(oos_shp_mat[0, 0]) else 0.

        oos_stats = detailed_eval_single(ticker_data, best_fold_overall["weights"],
                                          best_fold_overall["sell"],
                                          best_fold_overall["threshold"], hb_f, t_train_f, t_oos_end_f)

        print(f"  訓練Sharpe: {best_fold_overall['sharpe']:.4f}  OOS Sharpe: {oos_shp_f:.4f}  "
              f"OOS取引数: {oos_stats['n_trades']}  OOS勝率: {oos_stats['win_rate']*100:.1f}%")

        wf_folds.append({
            "fold":           fold_i,
            "train_bars":     t_train_f,
            "oos_bars":       t_oos_end_f - t_train_f,
            "best_sell":      best_fold_overall["sell"],
            "best_threshold": best_fold_overall["threshold"],
            "train_sharpe":   round(best_fold_overall["sharpe"], 4),
            "oos_sharpe":     round(oos_shp_f, 4),
            "oos_n_trades":   oos_stats["n_trades"],
            "oos_win_rate":   oos_stats["win_rate"],
            "oos_avg_return": oos_stats["avg_return"],
            "oos_max_dd":     oos_stats["max_dd"],
        })

    avg_oos_sharpe = float(np.mean([f["oos_sharpe"] for f in wf_folds]))
    print(f"\n  WF平均OOSシャープ: {avg_oos_sharpe:.4f}")

    # ── Step 3: 最終最適化 (全売りルール × GA, 訓練データ 0〜t_holdout) ───────
    print("\n  ===== STEP 3: 最終GA最適化 (全売りルール × GA) =====")
    best_overall, all_ga_results, _ = run_ga_all_sells(ticker_data, mode, doe_effects, t_holdout)

    best_sell   = best_overall["sell"]
    best_thresh = best_overall["threshold"]
    best_w      = best_overall["weights"]
    train_shp   = best_overall["sharpe"]
    convergence = best_overall["convergence"]

    best_weights = {nm: round(float(best_w[i]), 4) for i, nm in enumerate(ind_names)}
    print(f"  [最終] sell={best_sell}  thresh={best_thresh}  train_sharpe={train_shp:.4f}")
    print(f"  最適ウェイト: {best_weights}")

    # ── Step 4: MC最終評価 (5000サンプル, 訓練データ) ─────────────────────────
    print("\n  ===== STEP 4: MC最終評価 (5000) =====")
    np.random.seed(99)
    mc_w = np.random.dirichlet(np.ones(8), 5000) * 4.0
    mc_w = np.vstack([mc_w, best_w.reshape(1, 8)])

    fe_thresholds = SWING_BUY_THRESHOLDS if mode == "swing" else BUY_THRESHOLDS
    all_results = []
    hb = sell_hold[best_sell]
    shp_mat = batch_sharpe(ticker_data, mc_w, best_sell, fe_thresholds, hb, 0, t_holdout)
    for wi in range(len(mc_w)):
        for ti, thr in enumerate(fe_thresholds):
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

    if top100:
        tw_arr = np.array([r["weights"] for r in top100])
        med_w  = {nm: round(float(np.median(tw_arr[:, i])), 3) for i, nm in enumerate(ind_names)}
    else:
        med_w = {nm: round(float(best_w[i]), 3) for i, nm in enumerate(ind_names)}
    weight_ranking = sorted(ind_names, key=lambda n: med_w[n], reverse=True)

    # ── Step 5: ホールドアウト評価 (80〜100%) ─────────────────────────────────
    print("\n  ===== STEP 5: ホールドアウト評価 (80〜100%) =====")
    wm_best = best_w.reshape(1, 8)
    test_shp_mat = batch_sharpe(ticker_data, wm_best, best_sell,
                                [best_thresh], hb, t_holdout, T_min,
                                min_trades=MIN_TRADES_OOS)
    test_shp = float(test_shp_mat[0, 0]) if not np.isnan(test_shp_mat[0, 0]) else 0.

    test_stats = detailed_eval_single(ticker_data, best_w, best_sell,
                                       best_thresh, hb, t_holdout, T_min)

    print(f"  テスト: N={test_stats['n_trades']}  Sharpe={test_shp:.4f}  "
          f"勝率={test_stats['win_rate']*100:.1f}%  avg={test_stats['avg_return']:+.2f}%")

    # ── 売りルールランキング (訓練データ, GA最良重みで評価) ───────────────────
    print("  売りルールランキング評価中...")
    sell_stats = []
    for sname in sell_rules:
        hb_ = sell_hold[sname]
        sm  = batch_sharpe(ticker_data, wm_best, sname, fe_thresholds, hb_, 0, t_holdout)
        if not np.all(np.isnan(sm)):
            best_ti    = int(np.nanargmax(sm[0]))
            best_thr_s = fe_thresholds[best_ti]
            best_shp_s = float(sm[0, best_ti])
            avg_shp    = float(np.nanmean(sm[0]))
            dstats     = detailed_eval_single(ticker_data, best_w, sname,
                                               best_thr_s, hb_, 0, t_holdout)
        else:
            best_thr_s = fe_thresholds[0]; best_shp_s = 0.; avg_shp = 0.
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

    # ── top100 詳細評価 ────────────────────────────────────────────────────────
    print("  top100 詳細評価中...")
    top100_out = []
    for r in top100:
        ds = detailed_eval_single(ticker_data, r["weights"], best_sell,
                                   r["thr"], hb, 0, t_holdout)
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

    # GA全売りルール結果サマリ
    ga_all_sells_summary = {}
    for sname, gr in all_ga_results.items():
        ga_all_sells_summary[sname] = {
            "sharpe":    round(float(gr["sharpe"]), 4),
            "threshold": gr["threshold"],
            "weights":   {nm: round(float(gr["weights"][i]), 4) for i, nm in enumerate(ind_names)},
        }

    return {
        "version":       6,
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
            "avg_oos_sharpe":    round(avg_oos_sharpe, 4),
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
            "population":             GA_POP,
            "generations":            GA_GENS,
            "n_sell_rules_optimized": len(sell_rules),
            "all_sell_rules":         ga_all_sells_summary,
            "best_weights":           best_weights,
            "best_sharpe_train":      round(train_shp, 4),
            "best_sell_rule":         best_sell,
            "best_threshold":         best_thresh,
            "convergence":            [round(v, 4) for v in convergence],
        },
        "walk_forward": {
            "n_folds":          n_wf_folds,
            "folds":            wf_folds,
            "avg_oos_sharpe":   round(avg_oos_sharpe, 4),
            "train_ratio":      0.80,
            "train_bars":       t_holdout,
            "test_bars":        T_min - t_holdout,
            "train_sharpe":     round(train_shp, 4),
            "test_sharpe":      round(test_shp, 4),
            "test_win_rate":    round(test_stats["win_rate"], 4),
            "test_n_trades":    test_stats["n_trades"],
            "test_avg_return":  round(test_stats["avg_return"], 3),
            "test_max_dd":      round(test_stats["max_dd"], 3),
        },
    }

# ══════════════════════════════════════════════════════════════════════════════
# 13. モード実行
# ══════════════════════════════════════════════════════════════════════════════

def run_mode(mode: str):
    sell_rules = SWING_SELL_RULES if mode == "swing" else DAY_SELL_RULES
    max_hold   = MAX_HOLD_BARS_SWING if mode == "swing" else MAX_HOLD_BARS_DAY
    out_file   = RESULTS_SWING if mode == "swing" else RESULTS_DAY
    mode_ja    = "スイングトレード" if mode == "swing" else "デイトレード"

    t0 = time.time()
    print("=" * 65)
    print(f"  バックテスト v3 — {mode_ja} ({mode}) mode")
    print(f"  DOE (L18) → WF 4fold×全売り × GA (pop={GA_POP}, gen={GA_GENS}) → ホールドアウト20%")
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

        ind_scores = compute_ind_scores(td, mode)
        vol_ok     = vol_ok_mask(td["volumes"])

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

    print("\n[3/3] DOE → WF 3fold → GA (全売りルール) → ホールドアウト評価...")
    result = full_evaluation(ticker_data, mode)

    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    elapsed = time.time() - t0
    print(f"\n結果保存: {out_file}  ({out_file.stat().st_size // 1024} KB)")
    print(f"総実行時間: {elapsed:.0f}秒 ({elapsed/60:.1f}分)\n")

    wf = result["walk_forward"]
    ga = result["ga_results"]
    doe = result["doe_results"]
    print("=" * 65)
    print(f"  【{mode_ja} 最終結果】")
    print("=" * 65)
    print(f"  WF平均OOS Sharpe: {wf['avg_oos_sharpe']:.4f}  (3fold平均, 信頼性指標)")
    print(f"  訓練Sharpe: {wf['train_sharpe']:.4f}  テストSharpe: {wf['test_sharpe']:.4f}")
    print(f"  テスト勝率: {wf['test_win_rate']*100:.1f}%  "
          f"平均リターン: {wf['test_avg_return']:+.3f}%  "
          f"取引数: {wf['test_n_trades']}")
    print(f"  売りルール: {ga['best_sell_rule']}  閾値: {ga['best_threshold']}")
    print(f"\n  指標重要度 (DOE): {doe['ranking']}")
    for nm in (IND_NAMES_SWING if mode == "swing" else IND_NAMES_DAY):
        eff = doe["indicator_effects"].get(nm, 0.)
        w   = ga["best_weights"].get(nm, 0.)
        print(f"    {nm:<8s} ウェイト={w:.4f}  主効果={eff:.4f}")
    print(f"\n  WF Fold別OOS Sharpe:")
    for f in wf["folds"]:
        print(f"    Fold {f['fold']+1}: train={f['train_bars']}bars  "
              f"OOS_sharpe={f['oos_sharpe']:.4f}  "
              f"N={f['oos_n_trades']}  勝率={f['oos_win_rate']*100:.1f}%")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# 14. エントリーポイント
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="バックテスト (DOE→WF4fold×GA全売り→ホールドアウト)")
    # レガシーモード (後方互換)
    parser.add_argument("--swing", action="store_true", help="スイングトレードモード (レガシー)")
    parser.add_argument("--day",   action="store_true", help="デイトレードモード (レガシー)")
    parser.add_argument("--both",  action="store_true", help="両方実行 (レガシー)")
    # フェーズモード (16ジョブ並列用)
    parser.add_argument("--phase", type=str,
                        choices=["doe", "wf-fold", "final", "assemble"],
                        help="実行フェーズ: doe / wf-fold / final / assemble")
    parser.add_argument("--mode",  type=str, choices=["swing", "day"],
                        help="トレードモード (--phase と一緒に使用)")
    parser.add_argument("--fold",  type=int, default=0,
                        help="WF Fold番号 0〜3 (--phase wf-fold と一緒に使用)")
    parser.add_argument("--time-limit", type=int, default=0, metavar="MINUTES",
                        help="GA実行タイムリミット (分, 0=無制限)。超過時はチェックポイント保存して終了 (exit 1)。")
    args = parser.parse_args()

    time_limit_s = args.time_limit * 60 if args.time_limit > 0 else None

    if args.phase:
        if not args.mode:
            print("--phase 使用時は --mode が必要です"); sys.exit(1)
        if args.phase == "doe":
            run_phase_doe(args.mode)
        elif args.phase == "wf-fold":
            run_phase_wf_fold(args.mode, args.fold, time_limit_s=time_limit_s)
        elif args.phase == "final":
            run_phase_final(args.mode, time_limit_s=time_limit_s)
        elif args.phase == "assemble":
            run_phase_assemble(args.mode)
    elif args.both:
        run_mode("swing"); run_mode("day")
    elif args.swing:
        run_mode("swing")
    elif args.day:
        run_mode("day")
    else:
        print("引数が必要です: --swing / --day / --both  または  --phase <phase> --mode <mode>")
        parser.print_help(); sys.exit(1)


if __name__ == "__main__":
    main()
