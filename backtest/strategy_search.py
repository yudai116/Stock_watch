#!/usr/bin/env python3
"""
strategy_search.py v2 — 買い指標 × 売り指標 網羅的最適化

【先読みバイアスの排除】
  - スコア計算: bar[t] の終値時点のデータのみ使用
  - エントリー価格: bar[t+1] の始値（翌日のオープン）
  - 終値シグナル → 翌日始値エントリー → 現実的な執行をシミュレート

【指標 (8種類)】
  1. RSI(14)       — 売られすぎ逆張り
  2. MACD(12,26,9) — トレンドフォロー
  3. BB(20)        — バンド下限逆張り
  4. MA(20/50)     — 移動平均ゴールデンクロス
  5. Aroon(25)     — 時間ベーストレンド方向
  6. Stoch(14,3)   — ストキャスティクス逆張り
  7. CCI(20)       — 典型価格の統計偏差
  8. ROC(10)       — 純粋モメンタム

【売りルール (14種)】
  固定保有 3/5/7/10/15/20日
  利確+ストップ: +5/-3%, +10/-5%, +15/-5%, +20/-5%, +15/-7%, +20/-7%
  トレーリングストップ: 5%, 10%

【使い方】
  # 日足スイング (2015-2025, 15銘柄)
  python3 backtest/strategy_search.py

  # 1時間足デイトレ (直近2年)
  python3 backtest/strategy_search.py --intraday
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE         = Path(__file__).parent
DATA_FILE    = HERE / "price_data.json"
INTRADAY_FILE= HERE / "price_data_intraday.json"
RESULTS_FILE = HERE / "strategy_results.json"

# ── パラメータ ─────────────────────────────────────────────────────────────
N_WEIGHT_SAMPLES     = 8_000    # モンテカルロ重みサンプル数
MIN_TRADES           = 20
MAX_HOLD_DAYS        = 60
TRADING_DAYS_PER_YEAR = 252
BUY_THRESHOLDS       = [60, 65, 70, 75]

IND_NAMES = ["RSI", "MACD", "BB", "MA", "Aroon", "Stoch", "CCI", "ROC"]

SELL_RULES: dict[str, dict] = {
    "hold_3d":        {"type": "hold",        "days": 3},
    "hold_5d":        {"type": "hold",        "days": 5},
    "hold_7d":        {"type": "hold",        "days": 7},
    "hold_10d":       {"type": "hold",        "days": 10},
    "hold_15d":       {"type": "hold",        "days": 15},
    "hold_20d":       {"type": "hold",        "days": 20},
    "target5_stop3":  {"type": "target_stop", "target": 5,  "stop": 3},
    "target10_stop5": {"type": "target_stop", "target": 10, "stop": 5},
    "target15_stop5": {"type": "target_stop", "target": 15, "stop": 5},
    "target20_stop5": {"type": "target_stop", "target": 20, "stop": 5},
    "target15_stop7": {"type": "target_stop", "target": 15, "stop": 7},
    "target20_stop7": {"type": "target_stop", "target": 20, "stop": 7},
    "trail_5pct":     {"type": "trailing",    "trail": 5},
    "trail_10pct":    {"type": "trailing",    "trail": 10},
}

SELL_HOLD_DAYS = {
    "hold_3d": 3,  "hold_5d": 5,  "hold_7d": 7,
    "hold_10d": 10, "hold_15d": 15, "hold_20d": 20,
    "target5_stop3": 5,   "target10_stop5": 8,
    "target15_stop5": 10, "target20_stop5": 12,
    "target15_stop7": 10, "target20_stop7": 12,
    "trail_5pct": 8, "trail_10pct": 10,
}

SELL_RULE_JA = {
    "hold_3d":        "固定保有 3日",
    "hold_5d":        "固定保有 5日",
    "hold_7d":        "固定保有 7日",
    "hold_10d":       "固定保有 10日",
    "hold_15d":       "固定保有 15日",
    "hold_20d":       "固定保有 20日",
    "target5_stop3":  "利確+5% / ストップ−3%",
    "target10_stop5": "利確+10% / ストップ−5%",
    "target15_stop5": "利確+15% / ストップ−5%",
    "target20_stop5": "利確+20% / ストップ−5%",
    "target15_stop7": "利確+15% / ストップ−7%",
    "target20_stop7": "利確+20% / ストップ−7%",
    "trail_5pct":     "トレーリングストップ 5%",
    "trail_10pct":    "トレーリングストップ 10%",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. データ読み込み
# ══════════════════════════════════════════════════════════════════════════════

def load_data(intraday: bool = False) -> dict[str, dict]:
    f = INTRADAY_FILE if intraday else DATA_FILE
    if not f.exists() or f.stat().st_size < 100:
        mode = "intraday" if intraday else "daily"
        print(f"ERROR: {f.name} が見つかりません。")
        if intraday:
            print("  node backtest/fetch_data.mjs --intraday  を実行してください")
        else:
            print("  node backtest/fetch_data.mjs  を実行してください")
        sys.exit(1)
    raw = json.loads(f.read_text())
    result = {}
    for ticker, rows in raw.items():
        if len(rows) < 200:
            print(f"  SKIP {ticker}: {len(rows)} bars < 200")
            continue
        result[ticker] = {
            "closes": np.array([r["close"] for r in rows], dtype=np.float64),
            "opens":  np.array([r["open"]  for r in rows], dtype=np.float64),
            "highs":  np.array([r["high"]  for r in rows], dtype=np.float64),
            "lows":   np.array([r["low"]   for r in rows], dtype=np.float64),
            "dates":  [r["date"] for r in rows],
        }
    return result

# ══════════════════════════════════════════════════════════════════════════════
# 2. テクニカル指標計算
# ══════════════════════════════════════════════════════════════════════════════

def _ema(x: np.ndarray, k: int) -> np.ndarray:
    T   = len(x)
    out = np.full(T, np.nan)
    fv  = np.where(~np.isnan(x))[0]
    if len(fv) < k: return out
    sv = fv[0]
    if sv + k > T: return out
    out[sv + k - 1] = np.nanmean(x[sv:sv + k])
    a = 2.0 / (k + 1)
    for t in range(sv + k, T):
        out[t] = (x[t] if not np.isnan(x[t]) else out[t-1]) * a + out[t-1] * (1-a)
    return out

def _sma(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(k-1, len(x)):
        out[i] = x[i-k+1:i+1].mean()
    return out

def calc_rsi(c: np.ndarray, p: int = 14) -> np.ndarray:
    T = len(c); out = np.full(T, np.nan)
    if T < p+1: return out
    d = np.diff(c)
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ag, al = up[:p].mean(), dn[:p].mean()
    for i in range(p, T-1):
        ag = (ag*(p-1)+up[i])/p; al = (al*(p-1)+dn[i])/p
        out[i+1] = 100. if al==0 else 100.-100./(1.+ag/al)
    return out

def calc_macd(c: np.ndarray):
    ef = _ema(c, 12); es = _ema(c, 26)
    ml = ef - es; sl = _ema(ml, 9)
    return ml, sl, ml - sl

def calc_bb(c: np.ndarray, p: int = 20):
    T = len(c); pb = np.full(T, np.nan)
    for i in range(p-1, T):
        w = c[i-p+1:i+1]; m = w.mean(); s = w.std(ddof=1)
        u = m + 2*s; lo = m - 2*s
        pb[i] = (c[i]-lo)/(u-lo) if (u-lo) > 1e-10 else 0.5
    return pb

def calc_aroon(h: np.ndarray, lo: np.ndarray, p: int = 25):
    T = len(h); up = np.full(T, np.nan); dn = np.full(T, np.nan)
    for i in range(p, T):
        wh = h[i-p:i+1]; wl = lo[i-p:i+1]
        up[i] = np.argmax(wh) / p * 100
        dn[i] = np.argmin(wl) / p * 100
    return up, dn

def calc_stochastic(c: np.ndarray, h: np.ndarray, lo: np.ndarray,
                    k_period: int = 14, d_period: int = 3):
    T = len(c); k = np.full(T, np.nan)
    for i in range(k_period-1, T):
        hi = h[i-k_period+1:i+1].max(); li = lo[i-k_period+1:i+1].min()
        k[i] = (c[i]-li)/(hi-li)*100 if (hi-li) > 1e-10 else 50.
    d = _sma(k, d_period)
    return k, d

def calc_cci(c: np.ndarray, h: np.ndarray, lo: np.ndarray, p: int = 20):
    T = len(c); out = np.full(T, np.nan)
    tp = (c + h + lo) / 3.0
    for i in range(p-1, T):
        w = tp[i-p+1:i+1]; m = w.mean()
        md = np.abs(w - m).mean()
        out[i] = (tp[i]-m) / (0.015*md) if md > 1e-10 else 0.
    return out

def calc_roc(c: np.ndarray, p: int = 10) -> np.ndarray:
    out = np.full(len(c), np.nan)
    for i in range(p, len(c)):
        if c[i-p] > 0:
            out[i] = (c[i]/c[i-p] - 1) * 100
    return out

# ══════════════════════════════════════════════════════════════════════════════
# 3. 生スコア計算 (各 0-25 点)
# ══════════════════════════════════════════════════════════════════════════════

def score_rsi_v(r: np.ndarray) -> np.ndarray:
    s = np.where(r < 25,                          23.0,
        np.where(r < 35, 19-(r-25)*0.4,
        np.where(r < 45, 12-(r-35)*0.2,
        np.where(r < 55, 10-(r-45)*0.2,
        np.where(r < 65,  7-(r-55)*0.2,
                          np.maximum(0., 4-(r-65)*0.1))))))
    return np.clip(np.where(np.isnan(r), 0., s), 0., 25.)

def score_macd_v(ml: np.ndarray, sl: np.ndarray, hl: np.ndarray) -> np.ndarray:
    T = len(ml); v = ~np.isnan(ml) & ~np.isnan(sl)
    pml = np.roll(ml,1); psl = np.roll(sl,1); phl = np.roll(hl,1)
    gc   = v & (ml>sl)  & (pml<=psl); gc[0]  = False
    dc   = v & (ml<=sl) & (pml>psl);  dc[0]  = False
    above= v & (ml>sl)  & ~gc
    exp  = above & ~np.isnan(hl) & ~np.isnan(phl) & (hl>phl) & (hl>0); exp[0]=False
    s = np.where(gc, 24., np.where(exp, 15., np.where(above, 10., np.where(dc, 2., 0.))))
    return np.where(v, s, 0.)

def score_bb_v(pb: np.ndarray) -> np.ndarray:
    s = np.where(pb < 0.,                         20.,
        np.where(pb < 0.1, 15+(0.1-pb)/0.1*5,
        np.where(pb < 0.3,  9+(0.3-pb)/0.2*6,
        np.where(pb < 0.5,  6+(0.5-pb)/0.2*3,
        np.where(pb < 0.7,  3+(0.7-pb)/0.2*3,
        np.where(pb < 0.9,  1+(0.9-pb)/0.2*2, 0.))))))
    return np.clip(np.where(np.isnan(pb), 0., s), 0., 25.)

def score_ma_v(c: np.ndarray, ef: np.ndarray, es: np.ndarray) -> np.ndarray:
    v  = ~np.isnan(ef) & ~np.isnan(es)
    rt = np.where(v, c / np.where(ef!=0, ef, 1.), 1.)
    gc = v & (ef > es)
    pc = np.roll(c,1); pe = np.roll(ef,1)
    cr = v & (c>ef) & (pc<=pe); cr[0]=False
    base = np.where(cr, 15.,
           np.where(rt > 1., np.minimum(12., 4.+(rt-1.)*100.),
                    np.maximum(0., 3.-(1.-rt)*50.)))
    return np.clip(np.where(v, base+np.where(gc,8.,0.), 0.), 0., 25.)

def score_aroon_v(aup: np.ndarray, adn: np.ndarray) -> np.ndarray:
    """Aroon: トレンドフォロー (上昇トレンドで高得点)"""
    diff = aup - adn  # range: -100 to +100
    s = np.where(diff > 70,  20+(diff-70)/30*5,   # strong uptrend: 20-25
        np.where(diff > 30,  13+(diff-30)/40*7,   # uptrend: 13-20
        np.where(diff > 0,    8+(diff/30)*5,       # slight up: 8-13
        np.where(diff > -30,  4+(diff+30)/30*4,   # slight down: 4-8
        np.where(diff > -70,  2+(diff+70)/40*2,   # downtrend: 2-4
                              0.)))))              # strong downtrend: 0-2
    return np.clip(np.where(np.isnan(aup)|np.isnan(adn), 0., s), 0., 25.)

def score_stoch_v(k: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Stochastic: 売られすぎ逆張り (RSI と補完関係)"""
    s = np.where(k < 20,  20+(20-k)/20*5,    # oversold: 20-25
        np.where(k < 35,  13+(35-k)/15*7,    # leaving oversold: 13-20
        np.where(k < 50,   7+(50-k)/15*6,    # neutral-low: 7-13
        np.where(k < 70,   3+(70-k)/20*4,    # neutral-high: 3-7
        np.where(k < 85,   1+(85-k)/15*2,    # overbought approach: 1-3
                           0.)))))            # overbought: 0-1
    # ゴールデンクロス bonus (+3 points)
    pk = np.roll(k,1); pd = np.roll(d,1)
    gc = ~np.isnan(k) & ~np.isnan(d) & (k>d) & (pk<=pd); gc[0]=False
    s  = np.where(gc, s+3., s)
    return np.clip(np.where(np.isnan(k), 0., s), 0., 25.)

def score_cci_v(cci: np.ndarray) -> np.ndarray:
    """CCI: 売られすぎ逆張り"""
    s = np.where(cci < -200,  22+(np.minimum(cci+300,100)/100)*3,  # extreme: 22-25
        np.where(cci < -100,  14+(cci+200)/100*8,    # oversold: 14-22
        np.where(cci < 0,      8+(cci+100)/100*6,    # weak: 8-14
        np.where(cci < 100,    3+(100-cci)/100*5,    # neutral: 3-8
        np.where(cci < 200,    1+(200-cci)/100*2,    # overbought: 1-3
                               0.)))))               # extreme overbought: 0
    return np.clip(np.where(np.isnan(cci), 0., s), 0., 25.)

def score_roc_v(roc: np.ndarray) -> np.ndarray:
    """ROC: 純粋モメンタム (上昇モメンタムで高得点)"""
    s = np.where(roc > 30,   18+(np.minimum(roc-30,20)/20)*7,  # strong: 18-25
        np.where(roc > 15,   12+(roc-15)/15*6,    # good momentum: 12-18
        np.where(roc > 5,     7+(roc-5)/10*5,     # mild positive: 7-12
        np.where(roc > 0,     4+roc/5*3,          # slight positive: 4-7
        np.where(roc > -10,   1+(roc+10)/10*3,    # slight negative: 1-4
        np.where(roc > -25,   0.,                 # negative: 0
                              0.))))))
    return np.clip(np.where(np.isnan(roc), 0., s), 0., 25.)

# ══════════════════════════════════════════════════════════════════════════════
# 4. 売りルール事前計算 ★先読みバイアスなし★
#    シグナル: bar[t] の終値ベース
#    エントリー: bar[t+1] の始値 (opens[t+1])
#    エグジット: ルールに応じて bar[t+1] 以降の高値/安値/終値を参照
# ══════════════════════════════════════════════════════════════════════════════

def precompute_sell_outcomes(closes, opens, highs, lows) -> dict[str, np.ndarray]:
    T = len(closes)
    outcomes = {name: np.full(T, np.nan) for name in SELL_RULES}

    # ── 固定保有ルール: vectorized ──────────────────────────────────────────
    # signal at bar t → entry at opens[t+1] → exit at closes[t+N]
    for name, rule in SELL_RULES.items():
        if rule["type"] != "hold": continue
        N = rule["days"]
        ret = np.full(T, np.nan)
        # t ranges 0 .. T-N-2  (need opens[t+1] and closes[t+N], both in bounds)
        valid = T - N - 1
        if valid > 0:
            entry = opens[1:1+valid]         # opens[1], opens[2], ..., opens[T-N-1]
            exit_ = closes[N:N+valid]        # closes[N], closes[N+1], ..., closes[T-2]
            valid_mask = (entry > 0) & ~np.isnan(entry) & (exit_ > 0) & ~np.isnan(exit_)
            ret[:valid] = np.where(valid_mask, exit_ / entry - 1., np.nan)
        outcomes[name] = ret

    # ── 利確+ストップ / トレーリング: Python ループ ─────────────────────────
    for name, rule in SELL_RULES.items():
        if rule["type"] not in ("target_stop", "trailing"): continue
        ret = np.full(T, np.nan)
        for t in range(T - 2):  # need opens[t+1] to exist
            entry = opens[t + 1]
            if entry <= 0 or np.isnan(entry): continue

            if rule["type"] == "target_stop":
                P  = rule["target"] / 100.0
                S  = rule["stop"]   / 100.0
                tp = entry * (1 + P)
                sl = entry * (1 - S)
                result = None
                # bar t+1 以降をスキャン (エントリーバー含む)
                for i in range(t+1, min(t+1+MAX_HOLD_DAYS, T)):
                    if highs[i] >= tp: result = P;  break
                    if lows[i]  <= sl: result = -S; break
                if result is None:
                    # 時間切れ: MAX_HOLD 後の終値で手仕舞い
                    xt = t + 1 + MAX_HOLD_DAYS
                    if xt < T:
                        result = closes[xt] / entry - 1.
                ret[t] = result if result is not None else np.nan

            else:  # trailing
                trail  = rule["trail"] / 100.0
                peak   = entry
                result = None
                for i in range(t+1, min(t+1+MAX_HOLD_DAYS, T)):
                    peak = max(peak, highs[i])
                    if lows[i] <= peak * (1 - trail):
                        result = peak * (1 - trail) / entry - 1.; break
                if result is None:
                    xt = t + 1 + MAX_HOLD_DAYS
                    if xt < T:
                        result = closes[xt] / entry - 1.
                ret[t] = result if result is not None else np.nan

        outcomes[name] = ret
    return outcomes

# ══════════════════════════════════════════════════════════════════════════════
# 5. 戦略評価 (numpy ベクトル化)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_strategies(ticker_data: dict, weight_matrix: np.ndarray) -> list[dict]:
    N = len(weight_matrix)
    results = []
    total = len(SELL_RULES)
    done  = 0
    t0    = time.time()

    for sell_name in SELL_RULES:
        hold_d = SELL_HOLD_DAYS[sell_name]
        n_thr  = len(BUY_THRESHOLDS)

        acc_n    = np.zeros((N, n_thr))
        acc_sum  = np.zeros((N, n_thr))
        acc_sq   = np.zeros((N, n_thr))
        acc_wins = np.zeros((N, n_thr))
        acc_min  = np.full((N, n_thr), np.inf)

        for td in ticker_data.values():
            ind   = td["ind_scores"]          # (8, T)
            sout  = td["sell_outcomes"][sell_name]  # (T,)
            valid = ~np.isnan(sout)

            comp = (weight_matrix @ ind).astype(np.float32)  # (N, T)

            for ti, thr in enumerate(BUY_THRESHOLDS):
                mask = (comp >= thr) & valid     # (N, T)
                tr   = np.where(mask, sout.astype(np.float32), np.float32(np.nan))

                is_t = ~np.isnan(tr)
                safe = np.nan_to_num(tr, nan=0.)
                acc_n[:,  ti] += is_t.sum(1)
                acc_sum[:, ti] += (safe * is_t).sum(1)
                acc_sq[:,  ti] += (safe**2 * is_t).sum(1)
                acc_wins[:, ti] += (tr > 0).sum(1)
                rm = np.where(mask, sout.astype(np.float32), np.float32(np.inf)).min(1)
                acc_min[:, ti] = np.minimum(acc_min[:, ti], rm)

        for ti, thr in enumerate(BUY_THRESHOLDS):
            n    = acc_n[:, ti]
            s    = acc_sum[:, ti]
            sq   = acc_sq[:, ti]
            wins = acc_wins[:, ti]
            mn   = acc_min[:, ti]

            ok  = n >= MIN_TRADES
            avg = np.where(ok, s / np.where(n>0, n, 1), np.nan)
            var = np.where(ok & (n>1), sq/np.where(n>0,n,1) - avg**2, np.nan)
            std = np.sqrt(np.maximum(var, 0.))
            shp = np.where(ok & (std>1e-10),
                           avg/std * np.sqrt(TRADING_DAYS_PER_YEAR/hold_d), np.nan)
            wr  = np.where(ok, wins/n, np.nan)

            for w in range(N):
                if np.isnan(shp[w]): continue
                wv = weight_matrix[w]
                results.append({
                    "buy_weights": {nm: round(float(wv[i]),3) for i,nm in enumerate(IND_NAMES)},
                    "buy_threshold": thr,
                    "sell_rule":    sell_name,
                    "sharpe":       round(float(shp[w]),4),
                    "n_trades":     int(n[w]),
                    "win_rate":     round(float(wr[w]),4),
                    "avg_return":   round(float(avg[w]*100),3),
                    "max_dd":       round(float(mn[w]*100),3),
                })

        done += 1
        el  = time.time()-t0
        eta = el/done*(total-done)
        print(f"  [{done:2d}/{total}] {sell_name:<20s} 経過={el:.0f}s  残り≈{eta:.0f}s")

    return results

# ══════════════════════════════════════════════════════════════════════════════
# 6. メイン
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intraday", action="store_true",
                        help="1時間足モード (日足がデフォルト)")
    args = parser.parse_args()

    t0 = time.time()
    mode = "intraday (1h)" if args.intraday else "daily (1d)"
    print("=" * 65)
    print(f"  買い×売り 戦略最適化エンジン v2 — {mode}")
    print(f"  先読みバイアス修正: エントリー = 翌バーの始値 (opens[t+1])")
    print(f"  指標: {', '.join(IND_NAMES)}")
    print("=" * 65)

    raw = load_data(args.intraday)
    tickers = list(raw.keys())
    print(f"\n[1/4] {len(tickers)} 銘柄読み込み完了")

    print("\n[2/4] 指標計算中...")
    ticker_data: dict[str, dict] = {}

    for tkr, td in raw.items():
        c, o = td["closes"], td["opens"]
        h, lo = td["highs"], td["lows"]

        rsi_v            = calc_rsi(c)
        ml, sl, hl       = calc_macd(c)
        pb               = calc_bb(c)
        ef               = _ema(c, 20); es = _ema(c, 50)
        aup, adn         = calc_aroon(h, lo)
        sk, sd           = calc_stochastic(c, h, lo)
        cci              = calc_cci(c, h, lo)
        roc              = calc_roc(c)

        ind_scores = np.stack([
            score_rsi_v(rsi_v),
            score_macd_v(ml, sl, hl),
            score_bb_v(pb),
            score_ma_v(c, ef, es),
            score_aroon_v(aup, adn),
            score_stoch_v(sk, sd),
            score_cci_v(cci),
            score_roc_v(roc),
        ], axis=0)  # (8, T)

        print(f"  {tkr}: 売りリターン事前計算中 (エントリー=翌始値)...", end="", flush=True)
        sell_outcomes = precompute_sell_outcomes(c, o, h, lo)
        print(" done")

        ticker_data[tkr] = {"ind_scores": ind_scores, "sell_outcomes": sell_outcomes}

    print(f"\n[3/4] {N_WEIGHT_SAMPLES:,} 重みベクトルサンプリング (8指標)...")
    np.random.seed(42)
    raw_w = np.random.dirichlet([1]*8, N_WEIGHT_SAMPLES)
    wm    = raw_w * 4.0  # (N, 8), 合計=4 → 最大スコア=100

    # 境界ケース: 単一指標・等重み
    extra = np.vstack([
        np.eye(8) * 4.0,                         # 各指標単独 (8通り)
        np.ones((1,8)) * 0.5,                    # 等重み
        np.array([[1.5,1.5,0.5,0.5,0,0,0,0]]),  # RSI+MACD重視
        np.array([[0,0,1.5,1.5,0,0,0.5,0.5]]),  # BB+MA+CCI+ROC
        np.array([[0.8]*8]),                     # 全指標均等
    ])
    wm = np.vstack([wm, extra])
    print(f"  合計 {len(wm):,} 重みベクトル")

    print(f"\n[4/4] 戦略評価中 (合計 ≈ {len(wm)*len(SELL_RULES)*len(BUY_THRESHOLDS):,} 組み合わせ)...")
    results = evaluate_strategies(ticker_data, wm)
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    top100 = results[:100]

    sell_stats: list[dict] = []
    for name in SELL_RULES:
        sub = [r for r in results if r["sell_rule"]==name]
        if not sub: continue
        best = max(sub, key=lambda x: x["sharpe"])
        sell_stats.append({
            "sell_rule":        name,
            "sell_rule_ja":     SELL_RULE_JA[name],
            "best_sharpe":      round(best["sharpe"],4),
            "avg_sharpe":       round(float(np.mean([r["sharpe"] for r in sub])),4),
            "best_win_rate":    round(best["win_rate"],4),
            "best_n_trades":    best["n_trades"],
            "best_avg_return":  best["avg_return"],
            "best_weights":     best["buy_weights"],
            "best_threshold":   best["buy_threshold"],
        })
    sell_stats.sort(key=lambda x: x["best_sharpe"], reverse=True)

    sharpes_top = np.array([r["sharpe"] for r in results[:2000]])
    ind_corr = {}
    for nm in IND_NAMES:
        w_arr = np.array([r["buy_weights"][nm] for r in results[:2000]])
        ind_corr[nm] = round(float(np.corrcoef(w_arr, sharpes_top)[0,1]),4)

    med_w  = {nm: round(float(np.median([r["buy_weights"][nm] for r in top100])),3)
              for nm in IND_NAMES}
    ranked = sorted(IND_NAMES, key=lambda n: med_w[n], reverse=True)

    output = {
        "version":          4,
        "generated_at":     time.strftime("%Y-%m-%d %H:%M"),
        "mode":             mode,
        "n_tickers":        len(tickers),
        "tickers":          tickers,
        "n_evaluated":      len(results),
        "top100":           top100,
        "sell_rule_ranking":sell_stats,
        "indicator_weight_corr_with_sharpe": ind_corr,
        "top100_median_weights":   med_w,
        "top100_weight_ranking":   ranked,
        "summary": {
            "best_sharpe":    top100[0]["sharpe"] if top100 else None,
            "best_sell_rule": top100[0]["sell_rule"] if top100 else None,
            "best_weights":   top100[0]["buy_weights"] if top100 else None,
            "best_threshold": top100[0]["buy_threshold"] if top100 else None,
        },
        "lookahead_bias_fixed": True,
        "entry_price": "open of bar t+1 (signal at close of bar t)",
    }

    RESULTS_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n結果保存: {RESULTS_FILE}  ({RESULTS_FILE.stat().st_size//1024} KB)")
    print(f"総実行時間: {time.time()-t0:.0f}秒\n")

    print("=" * 65)
    print("  【戦略 TOP 10 (先読みバイアスなし)】")
    print("=" * 65)
    for i, r in enumerate(top100[:10]):
        w = r["buy_weights"]
        rk = sorted(IND_NAMES, key=lambda n: w[n], reverse=True)
        print(f"\n#{i+1:2d} シャープ={r['sharpe']:.3f}  勝率={r['win_rate']*100:.1f}%  "
              f"平均={r['avg_return']:+.2f}%  N={r['n_trades']}")
        print(f"    買い: {' + '.join(f'{n}×{w[n]:.2f}' for n in rk[:4])}")
        print(f"    閾値≥{r['buy_threshold']}  売り: {SELL_RULE_JA[r['sell_rule']]}")

    print("\n" + "=" * 65)
    print("  【売りルール ランキング】")
    print("=" * 65)
    for i, sr in enumerate(sell_stats[:10]):
        print(f"  {i+1:2d}位  {sr['sell_rule_ja']:<26s} "
              f"シャープ={sr['best_sharpe']:.3f}  勝率={sr['best_win_rate']*100:.1f}%")

    print("\n" + "=" * 65)
    print("  【買い指標 重要度 (上位100戦略)】")
    print("=" * 65)
    for i, nm in enumerate(ranked):
        print(f"  {i+1}位 {nm:<7s} 重み中央値={med_w[nm]:.3f}  "
              f"シャープ相関={ind_corr[nm]:+.3f}")


if __name__ == "__main__":
    main()
