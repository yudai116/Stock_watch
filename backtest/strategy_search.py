#!/usr/bin/env python3
"""
strategy_search.py — 買い指標 × 売り指標 網羅的最適化

【使い方】
  Step 1: node backtest/fetch_data.mjs          # 実際の株価データ取得
  Step 2: python3 backtest/strategy_search.py   # 最適化実行（約5〜15分）
  Step 3: 結果は backtest/strategy_results.json に保存される

【検証内容】
  買いシグナル: RSI/MACD/BB/MA の重み組み合わせ 10,000通り × 閾値4種
  売りルール: 固定保有・利確+ストップ・トレーリングストップ など14種
  合計: 約 560,000 通りの戦略組み合わせ
  評価指標: アニュアライズド・シャープレシオ (リスク調整後収益)

【依存】
  pip install numpy (標準的な科学計算環境に含まれる)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── パス設定 ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
DATA_FILE    = HERE / "price_data.json"
RESULTS_FILE = HERE / "strategy_results.json"

# ── パラメータ ───────────────────────────────────────────────────────────────────
N_WEIGHT_SAMPLES     = 10_000   # Monte Carlo 重みサンプル数
MIN_TRADES           = 20       # 有効とみなす最小トレード数
MAX_HOLD_DAYS        = 60       # 売りルール探索の最大保有日数
TRADING_DAYS_PER_YEAR = 252
BUY_THRESHOLDS       = [60, 65, 70, 75]  # 買い閾値（スコアがこれ以上で買い）
IND_NAMES            = ["RSI", "MACD", "BB", "MA"]

# ── 売りルール定義 ───────────────────────────────────────────────────────────────
SELL_RULES: dict[str, dict] = {
    "hold_3d":         {"type": "hold",        "days": 3},
    "hold_5d":         {"type": "hold",        "days": 5},
    "hold_7d":         {"type": "hold",        "days": 7},
    "hold_10d":        {"type": "hold",        "days": 10},
    "hold_15d":        {"type": "hold",        "days": 15},
    "hold_20d":        {"type": "hold",        "days": 20},
    "target5_stop3":   {"type": "target_stop", "target": 5,  "stop": 3},
    "target10_stop5":  {"type": "target_stop", "target": 10, "stop": 5},
    "target15_stop5":  {"type": "target_stop", "target": 15, "stop": 5},
    "target20_stop5":  {"type": "target_stop", "target": 20, "stop": 5},
    "target15_stop7":  {"type": "target_stop", "target": 15, "stop": 7},
    "target20_stop7":  {"type": "target_stop", "target": 20, "stop": 7},
    "trail_5pct":      {"type": "trailing",    "trail": 5},
    "trail_10pct":     {"type": "trailing",    "trail": 10},
}

# 平均保有日数（シャープ年率換算用）
SELL_HOLD_DAYS: dict[str, float] = {
    "hold_3d": 3, "hold_5d": 5, "hold_7d": 7,
    "hold_10d": 10, "hold_15d": 15, "hold_20d": 20,
    "target5_stop3": 5,   "target10_stop5": 8,
    "target15_stop5": 10, "target20_stop5": 12,
    "target15_stop7": 10, "target20_stop7": 12,
    "trail_5pct": 8, "trail_10pct": 10,
}

SELL_RULE_JA: dict[str, str] = {
    "hold_3d":         "固定保有 3日",
    "hold_5d":         "固定保有 5日",
    "hold_7d":         "固定保有 7日",
    "hold_10d":        "固定保有 10日",
    "hold_15d":        "固定保有 15日",
    "hold_20d":        "固定保有 20日",
    "target5_stop3":   "利確+5% / ストップ−3%",
    "target10_stop5":  "利確+10% / ストップ−5%",
    "target15_stop5":  "利確+15% / ストップ−5%",
    "target20_stop5":  "利確+20% / ストップ−5%",
    "target15_stop7":  "利確+15% / ストップ−7%",
    "target20_stop7":  "利確+20% / ストップ−7%",
    "trail_5pct":      "トレーリングストップ 5%",
    "trail_10pct":     "トレーリングストップ 10%",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. データ読み込み
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> dict[str, dict]:
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size < 10:
        print("=" * 60)
        print("ERROR: price_data.json が存在しないか空です。")
        print()
        print("まず以下を実行してください:")
        print("  cd /path/to/Stock_watch")
        print("  node backtest/fetch_data.mjs")
        print("=" * 60)
        sys.exit(1)

    raw = json.loads(DATA_FILE.read_text())
    result: dict[str, dict] = {}
    for ticker, rows in raw.items():
        if len(rows) < 200:
            print(f"  SKIP {ticker}: データ不足 ({len(rows)} bars < 200)")
            continue
        result[ticker] = {
            "closes": np.array([r["close"] for r in rows], dtype=np.float64),
            "highs":  np.array([r["high"]  for r in rows], dtype=np.float64),
            "lows":   np.array([r["low"]   for r in rows], dtype=np.float64),
            "dates":  [r["date"] for r in rows],
        }
    return result

# ══════════════════════════════════════════════════════════════════════════════
# 2. テクニカル指標計算
# ══════════════════════════════════════════════════════════════════════════════

def _ema(x: np.ndarray, k: int) -> np.ndarray:
    T = len(x)
    out = np.full(T, np.nan)
    if T < k:
        return out
    first = np.where(~np.isnan(x))[0]
    if len(first) < k:
        return out
    sv = first[0]
    if sv + k > T:
        return out
    out[sv + k - 1] = np.nanmean(x[sv : sv + k])
    a = 2.0 / (k + 1)
    for t in range(sv + k, T):
        prev = out[t - 1]
        if np.isnan(x[t]):
            out[t] = prev
        else:
            out[t] = x[t] * a + prev * (1.0 - a)
    return out


def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    T = len(closes)
    out = np.full(T, np.nan)
    if T < period + 1:
        return out
    d = np.diff(closes)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ag = up[:period].mean()
    al = dn[:period].mean()
    for i in range(period, T - 1):
        ag = (ag * (period - 1) + up[i]) / period
        al = (al * (period - 1) + dn[i]) / period
        out[i + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def calc_macd(closes: np.ndarray, fast=12, slow=26, signal=9):
    ef = _ema(closes, fast)
    es = _ema(closes, slow)
    line = ef - es
    sig  = _ema(line, signal)
    hist = line - sig
    return line, sig, hist


def calc_bb(closes: np.ndarray, period: int = 20, n_std: float = 2.0):
    T = len(closes)
    pct_b = np.full(T, np.nan)
    upper = np.full(T, np.nan)
    lower = np.full(T, np.nan)
    for i in range(period - 1, T):
        w   = closes[i - period + 1 : i + 1]
        mid = w.mean()
        std = w.std(ddof=1)
        up  = mid + n_std * std
        lo  = mid - n_std * std
        upper[i] = up
        lower[i] = lo
        pct_b[i] = (closes[i] - lo) / (up - lo) if (up - lo) > 1e-10 else 0.5
    return pct_b, upper, lower


def calc_aroon(highs: np.ndarray, lows: np.ndarray, period: int = 25):
    T = len(highs)
    up = np.full(T, np.nan)
    dn = np.full(T, np.nan)
    for i in range(period, T):
        wh = highs[i - period : i + 1]
        wl = lows[i - period : i + 1]
        up[i] = np.argmax(wh) / period * 100.0
        dn[i] = np.argmin(wl) / period * 100.0
    return up, dn

# ══════════════════════════════════════════════════════════════════════════════
# 3. 生スコア計算 (0-25点 各指標)
# ══════════════════════════════════════════════════════════════════════════════

def score_rsi_v(r: np.ndarray) -> np.ndarray:
    """RSI → 0〜25点 (逆張り: 売られすぎで高得点)"""
    s = np.zeros(len(r))
    s = np.where(r < 25,                            23.0,                             s)
    s = np.where((r >= 25) & (r < 35), 19.0 - (r - 25) * 0.4,                      s)
    s = np.where((r >= 35) & (r < 45), 12.0 - (r - 35) * 0.2,                      s)
    s = np.where((r >= 45) & (r < 55), 10.0 - (r - 45) * 0.2,                      s)
    s = np.where((r >= 55) & (r < 65),  7.0 - (r - 55) * 0.2,                      s)
    s = np.where(r >= 65,               np.maximum(0.0, 4.0 - (r - 65) * 0.1),     s)
    s = np.where(np.isnan(r), 0.0, s)
    return np.clip(s, 0.0, 25.0)


def score_macd_v(line: np.ndarray, sig: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """MACD → 0〜25点 (ゴールデンクロス・ヒスト拡大で高得点)"""
    T = len(line)
    valid = (~np.isnan(line)) & (~np.isnan(sig))
    prev_line = np.roll(line, 1)
    prev_sig  = np.roll(sig,  1)
    prev_hist = np.roll(hist, 1)

    gc = valid & (line > sig)  & (prev_line <= prev_sig);  gc[0]  = False
    dc = valid & (line <= sig) & (prev_line > prev_sig);   dc[0]  = False
    above = valid & (line > sig) & ~gc
    expand = above & (~np.isnan(hist)) & (~np.isnan(prev_hist)) & (hist > prev_hist) & (hist > 0)
    expand[0] = False

    s = np.zeros(T)
    s = np.where(gc,              24.0, s)
    s = np.where(expand,          15.0, s)
    s = np.where(above & ~expand, 10.0, s)
    s = np.where(dc,               2.0, s)
    return s


def score_bb_v(pct_b: np.ndarray) -> np.ndarray:
    """BB %B → 0〜25点 (下限付近で高得点)"""
    p = pct_b
    s = np.zeros(len(p))
    s = np.where(p < 0.0,                          20.0,                                    s)
    s = np.where((p >= 0.0) & (p < 0.1),   15.0 + (0.1 - p) / 0.1 * 5.0,                 s)
    s = np.where((p >= 0.1) & (p < 0.3),    9.0 + (0.3 - p) / 0.2 * 6.0,                 s)
    s = np.where((p >= 0.3) & (p < 0.5),    6.0 + (0.5 - p) / 0.2 * 3.0,                 s)
    s = np.where((p >= 0.5) & (p < 0.7),    3.0 + (0.7 - p) / 0.2 * 3.0,                 s)
    s = np.where((p >= 0.7) & (p < 0.9),    1.0 + (0.9 - p) / 0.2 * 2.0,                 s)
    s = np.where(p >= 0.9,                   0.0,                                           s)
    s = np.where(np.isnan(p),                0.0,                                           s)
    return np.clip(s, 0.0, 25.0)


def score_ma_v(closes: np.ndarray, ema_fast: np.ndarray, ema_slow: np.ndarray) -> np.ndarray:
    """MA → 0〜25点 (EMA上抜け・GC帯で高得点)"""
    valid  = (~np.isnan(ema_fast)) & (~np.isnan(ema_slow))
    ratio  = np.where(valid, closes / np.where(ema_fast != 0, ema_fast, 1.0), 1.0)
    gc     = valid & (ema_fast > ema_slow)
    prev_c = np.roll(closes,   1)
    prev_e = np.roll(ema_fast, 1)
    cross  = valid & (closes > ema_fast) & (prev_c <= prev_e)
    cross[0] = False

    base = np.where(cross, 15.0,
           np.where(ratio > 1.0, np.minimum(12.0, 4.0 + (ratio - 1.0) * 100.0),
                    np.maximum(0.0, 3.0 - (1.0 - ratio) * 50.0)))
    base  = np.where(valid, base, 0.0)
    bonus = np.where(gc, 8.0, 0.0)
    return np.clip(base + bonus, 0.0, 25.0)

# ══════════════════════════════════════════════════════════════════════════════
# 4. 売りルール: 日別リターン事前計算
# ══════════════════════════════════════════════════════════════════════════════

def precompute_sell_outcomes(
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
) -> dict[str, np.ndarray]:
    """各日tにエントリーした場合の、各売りルール下でのリターンを事前計算"""
    T = len(closes)
    outcomes: dict[str, np.ndarray] = {
        name: np.full(T, np.nan) for name in SELL_RULES
    }

    # 固定保有: numpy シフトで高速化
    for name, rule in SELL_RULES.items():
        if rule["type"] == "hold":
            n = rule["days"]
            if n < T:
                # outcome[t] = closes[t+n] / closes[t] - 1
                future = np.full(T, np.nan)
                future[: T - n] = closes[n:]
                outcomes[name] = future / closes - 1.0

    # 利確+ストップ / トレーリング: Python ループ (事前計算なので1回限り)
    for name, rule in SELL_RULES.items():
        if rule["type"] not in ("target_stop", "trailing"):
            continue

        ret = np.full(T, np.nan)
        for t in range(T - 1):
            entry = closes[t]
            if entry <= 0.0 or np.isnan(entry):
                continue

            if rule["type"] == "target_stop":
                P = rule["target"] / 100.0
                S = rule["stop"]   / 100.0
                tp = entry * (1.0 + P)
                sl = entry * (1.0 - S)
                result = None
                for i in range(t + 1, min(t + MAX_HOLD_DAYS + 1, T)):
                    if highs[i] >= tp:
                        result = P;  break
                    if lows[i]  <= sl:
                        result = -S; break
                if result is None and t + MAX_HOLD_DAYS < T:
                    result = closes[t + MAX_HOLD_DAYS] / entry - 1.0
                if result is not None:
                    ret[t] = result

            else:  # trailing
                trail = rule["trail"] / 100.0
                peak  = entry
                result = None
                for i in range(t + 1, min(t + MAX_HOLD_DAYS + 1, T)):
                    peak = max(peak, highs[i])
                    if lows[i] <= peak * (1.0 - trail):
                        result = peak * (1.0 - trail) / entry - 1.0
                        break
                if result is None and t + MAX_HOLD_DAYS < T:
                    result = closes[t + MAX_HOLD_DAYS] / entry - 1.0
                if result is not None:
                    ret[t] = result

        outcomes[name] = ret

    return outcomes

# ══════════════════════════════════════════════════════════════════════════════
# 5. 戦略評価エンジン (numpy ベクトル化)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_strategies(
    ticker_data: dict[str, dict],
    weight_matrix: np.ndarray,  # (N, 4)  — 各行が一つの重みベクトル
) -> list[dict]:
    """
    全重みベクトル × 全売りルール × 全閾値 の戦略を評価し、
    有効な結果リストを返す。
    """
    N = len(weight_matrix)
    results: list[dict] = []

    total_combos = len(SELL_RULES) * len(BUY_THRESHOLDS)
    done = 0
    t_start = time.time()

    for sell_name, sell_rule in SELL_RULES.items():
        hold_days = SELL_HOLD_DAYS[sell_name]

        # 全閾値をまとめて処理: ticker ループは1回で済む
        # accumulate statistics per (weight_idx, threshold_idx)
        n_thresh = len(BUY_THRESHOLDS)

        # Accumulators: shape (N, n_thresh)
        acc_n    = np.zeros((N, n_thresh), dtype=np.float64)
        acc_sum  = np.zeros((N, n_thresh), dtype=np.float64)
        acc_sq   = np.zeros((N, n_thresh), dtype=np.float64)
        acc_wins = np.zeros((N, n_thresh), dtype=np.float64)
        acc_min  = np.full((N, n_thresh),  np.inf, dtype=np.float64)

        for ticker, td in ticker_data.items():
            ind_scores  = td["ind_scores"]          # (4, T)
            sell_outs   = td["sell_outcomes"][sell_name]  # (T,)
            valid_sell  = ~np.isnan(sell_outs)      # (T,)

            # composite scores: (N, T)  [使用メモリ: N*T*8 byte ≈ 100 MB]
            composite = (weight_matrix @ ind_scores).astype(np.float32)

            for ti, threshold in enumerate(BUY_THRESHOLDS):
                buy_mask = (composite >= threshold) & valid_sell  # (N, T)

                # trade matrix: returns for trade days, NaN otherwise
                # shape (N, T), dtype float32
                trade_rets = np.where(
                    buy_mask,
                    sell_outs.astype(np.float32),
                    np.float32(np.nan),
                )

                is_trade = ~np.isnan(trade_rets)                        # (N, T)
                safe     = np.nan_to_num(trade_rets, nan=0.0)

                acc_n[:,  ti] += is_trade.sum(axis=1)
                acc_sum[:, ti] += (safe * is_trade).sum(axis=1)
                acc_sq[:, ti]  += (safe ** 2 * is_trade).sum(axis=1)
                acc_wins[:, ti] += (trade_rets > 0).sum(axis=1)

                # min return per weight vector
                row_min = np.where(buy_mask, sell_outs.astype(np.float32), np.float32(np.inf)).min(axis=1)
                acc_min[:, ti] = np.minimum(acc_min[:, ti], row_min)

        # Compute metrics for all (weight, threshold) pairs
        for ti, threshold in enumerate(BUY_THRESHOLDS):
            n    = acc_n[:, ti]
            s    = acc_sum[:, ti]
            sq   = acc_sq[:, ti]
            wins = acc_wins[:, ti]
            mn   = acc_min[:, ti]

            valid_mask = n >= MIN_TRADES
            avg = np.where(valid_mask, s / np.where(n > 0, n, 1), np.nan)
            var = np.where(valid_mask & (n > 1),
                           sq / np.where(n > 0, n, 1) - avg ** 2,
                           np.nan)
            std = np.sqrt(np.maximum(var, 0.0))
            # Annualized Sharpe: scale by sqrt(trades_per_year)
            trades_per_year = TRADING_DAYS_PER_YEAR / hold_days
            sharpe = np.where(
                valid_mask & (std > 1e-10),
                avg / std * np.sqrt(trades_per_year),
                np.nan,
            )
            win_rate = np.where(valid_mask, wins / n, np.nan)

            for w in range(N):
                if np.isnan(sharpe[w]):
                    continue
                wv = weight_matrix[w]
                results.append({
                    "buy_weights": {
                        "RSI":  round(float(wv[0]), 3),
                        "MACD": round(float(wv[1]), 3),
                        "BB":   round(float(wv[2]), 3),
                        "MA":   round(float(wv[3]), 3),
                    },
                    "buy_threshold": threshold,
                    "sell_rule":     sell_name,
                    "sharpe":        round(float(sharpe[w]), 4),
                    "n_trades":      int(n[w]),
                    "win_rate":      round(float(win_rate[w]), 4),
                    "avg_return":    round(float(avg[w] * 100), 3),
                    "max_dd":        round(float(mn[w] * 100), 3),
                })

        done += 1
        elapsed = time.time() - t_start
        eta = elapsed / done * (total_combos - done)
        print(f"  [{done:2d}/{total_combos}] {sell_name}  "
              f"経過={elapsed:.0f}s  残り≈{eta:.0f}s")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. メイン
# ══════════════════════════════════════════════════════════════════════════════

def _top_weights_for(results: list[dict], sell_name: str) -> dict | None:
    sub = [r for r in results if r["sell_rule"] == sell_name]
    if not sub:
        return None
    return max(sub, key=lambda x: x["sharpe"])


def main() -> None:
    t0 = time.time()
    print("=" * 65)
    print("  買い×売り 戦略最適化エンジン — 実データ版")
    print("=" * 65)

    # ── データ読み込み ───────────────────────────────────────────────────────
    print("\n[1/4] 株価データ読み込み中...")
    raw_data = load_data()
    tickers = list(raw_data.keys())
    print(f"  {len(tickers)} 銘柄: {', '.join(tickers)}")
    for tkr, td in raw_data.items():
        T = len(td["closes"])
        print(f"  {tkr}: {T} 営業日 ({td['dates'][0]} → {td['dates'][-1]})")

    # ── 指標計算 ─────────────────────────────────────────────────────────────
    print("\n[2/4] テクニカル指標計算中...")
    ticker_data: dict[str, dict] = {}

    for ticker, td in raw_data.items():
        closes = td["closes"]
        highs  = td["highs"]
        lows   = td["lows"]

        rsi_vals          = calc_rsi(closes)
        ml, sl, hl        = calc_macd(closes)
        pct_b, bb_up, _   = calc_bb(closes)
        ema20             = _ema(closes, 20)
        ema50             = _ema(closes, 50)

        ind_scores = np.stack([
            score_rsi_v(rsi_vals),
            score_macd_v(ml, sl, hl),
            score_bb_v(pct_b),
            score_ma_v(closes, ema20, ema50),
        ], axis=0)  # (4, T)

        print(f"  {ticker}: 売りリターン事前計算中...", end="", flush=True)
        sell_outcomes = precompute_sell_outcomes(closes, highs, lows)
        print(" done")

        ticker_data[ticker] = {
            "ind_scores":    ind_scores,
            "sell_outcomes": sell_outcomes,
            "T":             len(closes),
        }

    # ── 重みサンプリング ─────────────────────────────────────────────────────
    print(f"\n[3/4] {N_WEIGHT_SAMPLES:,} 通りの重みベクトルをサンプリング中...")
    np.random.seed(42)

    # Dirichlet(1,1,1,1): 全指標に均等な事前分布、合計=1
    # × 4 でスケール → 各重みの範囲 ≈ 0〜2.5、合計=4
    raw_w = np.random.dirichlet([1, 1, 1, 1], size=N_WEIGHT_SAMPLES)
    weight_matrix = raw_w * 4.0  # (N, 4)

    # 境界ケースを追加: 単一指標、等重み
    extra = np.array([
        [4.0, 0.0, 0.0, 0.0],  # RSI のみ
        [0.0, 4.0, 0.0, 0.0],  # MACD のみ
        [0.0, 0.0, 4.0, 0.0],  # BB のみ
        [0.0, 0.0, 0.0, 4.0],  # MA のみ
        [1.0, 1.0, 1.0, 1.0],  # 等重み
        [2.0, 1.0, 0.5, 0.5],  # MACD 重視
        [0.5, 0.5, 1.0, 2.0],  # MA 重視
        [1.5, 1.5, 0.5, 0.5],  # MACD+RSI 重視
    ], dtype=np.float64)
    weight_matrix = np.vstack([weight_matrix, extra])

    N_total = len(weight_matrix)
    print(f"  合計 {N_total:,} 通りの重みベクトル")

    # ── 戦略評価 ─────────────────────────────────────────────────────────────
    print(f"\n[4/4] 戦略評価中...")
    print(f"  売りルール数={len(SELL_RULES)}, 閾値={BUY_THRESHOLDS}")
    print(f"  検証組み合わせ数 ≈ {N_total * len(SELL_RULES) * len(BUY_THRESHOLDS):,}")

    results = evaluate_strategies(ticker_data, weight_matrix)

    # ── 集計・ランキング ──────────────────────────────────────────────────────
    print(f"\n有効戦略数: {len(results):,}")
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    top100 = results[:100]

    # 売りルール別ベスト
    best_by_sell = {
        name: _top_weights_for(results, name)
        for name in SELL_RULES
    }

    # 売りルール別ランキング (Sharpe平均で比較)
    sell_rule_stats: list[dict] = []
    for name in SELL_RULES:
        sub = [r for r in results if r["sell_rule"] == name]
        if not sub:
            continue
        sharpes = [r["sharpe"] for r in sub]
        best = max(sub, key=lambda x: x["sharpe"])
        sell_rule_stats.append({
            "sell_rule":     name,
            "sell_rule_ja":  SELL_RULE_JA[name],
            "best_sharpe":   round(best["sharpe"], 4),
            "avg_sharpe":    round(float(np.mean(sharpes)), 4),
            "best_win_rate": round(best["win_rate"], 4),
            "best_n_trades": best["n_trades"],
            "best_avg_return": best["avg_return"],
            "best_weights":  best["buy_weights"],
            "best_threshold": best["buy_threshold"],
        })
    sell_rule_stats.sort(key=lambda x: x["best_sharpe"], reverse=True)

    # 指標重要度: 上位1000戦略でのシャープとの相関
    if len(results) >= 1000:
        top1k = results[:1000]
        sharpes_arr = np.array([r["sharpe"] for r in top1k])
        ind_corr = {}
        for name in IND_NAMES:
            w_arr = np.array([r["buy_weights"][name] for r in top1k])
            corr = float(np.corrcoef(w_arr, sharpes_arr)[0, 1])
            ind_corr[name] = round(corr, 4)
    else:
        ind_corr = {name: 0.0 for name in IND_NAMES}

    # 最適重みの中央値 (上位100戦略)
    top_weights_median = {
        name: round(float(np.median([r["buy_weights"][name] for r in top100])), 3)
        for name in IND_NAMES
    }
    # 順位付け
    top_weights_ranked = sorted(IND_NAMES, key=lambda n: top_weights_median[n], reverse=True)

    # ── 保存 ────────────────────────────────────────────────────────────────
    output = {
        "version":           3,
        "generated_at":      time.strftime("%Y-%m-%d %H:%M"),
        "n_tickers":         len(tickers),
        "tickers":           tickers,
        "n_weight_samples":  N_total,
        "n_evaluated":       len(results),
        "top100":            top100,
        "sell_rule_ranking": sell_rule_stats,
        "indicator_weight_corr_with_sharpe": ind_corr,
        "top100_median_weights":  top_weights_median,
        "top100_weight_ranking":  top_weights_ranked,
        "summary": {
            "best_sharpe":      top100[0]["sharpe"] if top100 else None,
            "best_sell_rule":   top100[0]["sell_rule"] if top100 else None,
            "best_weights":     top100[0]["buy_weights"] if top100 else None,
            "best_threshold":   top100[0]["buy_threshold"] if top100 else None,
        },
    }

    RESULTS_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n結果保存: {RESULTS_FILE}")
    print(f"総実行時間: {time.time() - t0:.0f}秒\n")

    # ── コンソール出力 ────────────────────────────────────────────────────────
    print("=" * 65)
    print("  【買い×売り 戦略 TOP 10】")
    print("=" * 65)
    for i, r in enumerate(top100[:10]):
        w = r["buy_weights"]
        print(f"\n#{i+1:2d}  シャープ={r['sharpe']:.3f}  "
              f"勝率={r['win_rate']*100:.1f}%  "
              f"平均収益={r['avg_return']:+.2f}%  "
              f"取引数={r['n_trades']}")
        ranked = sorted(IND_NAMES, key=lambda n: w[n], reverse=True)
        print(f"     買い: {' + '.join(f'{n}×{w[n]:.2f}' for n in ranked)}")
        print(f"          スコア閾値≥{r['buy_threshold']}")
        print(f"     売り: {SELL_RULE_JA[r['sell_rule']]}")
        print(f"     最大損失: {r['max_dd']:+.1f}%")

    print("\n" + "=" * 65)
    print("  【売りルール ランキング (ベストシャープ順)】")
    print("=" * 65)
    for i, sr in enumerate(sell_rule_stats[:10]):
        print(f"  {i+1:2d}位  {sr['sell_rule_ja']:<26s}  "
              f"シャープ={sr['best_sharpe']:.3f}  "
              f"勝率={sr['best_win_rate']*100:.1f}%  "
              f"平均={sr['best_avg_return']:+.2f}%")

    print("\n" + "=" * 65)
    print("  【買い指標 重要度ランキング (上位100戦略での重み中央値)】")
    print("=" * 65)
    for i, name in enumerate(top_weights_ranked):
        corr = ind_corr.get(name, 0)
        med  = top_weights_median[name]
        print(f"  {i+1}位  {name:<6s}  重み中央値={med:.3f}  "
              f"シャープとの相関={corr:+.3f}")

    print("\n✓ strategy_results.json を UI に読み込んでください")


if __name__ == "__main__":
    main()
