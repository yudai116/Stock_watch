#!/usr/bin/env python3
"""
Paper Trading Signal Generator
毎営業日 UTC 23:00 に GitHub Actions で実行。
yfinance で実価格を取得し、10アルゴ並行でシグナル計算 → positions.json / trade_log.json を更新。
US株: USD資金プール / JP株: JPY資金プール — 別管理。
"""

import json
import math
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import yfinance as yf

BASE_DIR = Path(__file__).parent.parent
TRADING_DIR = Path(__file__).parent

US_TICKERS = [
    "NVDA", "ASML", "MSFT", "AMD", "AVGO", "QCOM", "MU", "TSM", "ARM", "INTC",
    "TXN", "AMAT", "LRCX", "KLAC", "ENTG", "ACLS", "ADI", "NXPI", "MCHP", "ON",
    "MPWR", "SWKS", "MRVL", "CRWD", "PLTR", "SMCI", "META", "GOOGL", "AMZN",
    "ORCL", "ANET", "NOW", "PANW", "DDOG", "ZS", "SOUN", "IONQ",
]
JP_TICKERS = [
    "8035.T",  # 東京エレクトロン
    "6857.T",  # アドバンテスト
    "6723.T",  # ルネサスエレクトロニクス
    "4063.T",  # 信越化学
    "6963.T",  # ローム
    "6920.T",  # レーザーテック
    "6146.T",  # ディスコ
    "6981.T",  # 村田製作所
    "6762.T",  # TDK
    "6902.T",  # デンソー
]
ALL_TICKERS = US_TICKERS + JP_TICKERS

# Per-algo initial capital
INITIAL_CAPITAL_USD = 20000.0     # 各アルゴ $20,000
INITIAL_CAPITAL_JPY = 3000000.0   # 各アルゴ ¥3,000,000 (≈ $20,000 at ¥150)

# Position sizing
BASE_POSITION_USD = 2000.0    # 基本ポジションサイズ (USD)
BASE_POSITION_JPY = 300000.0  # 基本ポジションサイズ (JPY) ≈ $2,000 at ¥150
MAX_MULTIPLIER = 2.0          # スコア最大時の倍率
MIN_MULTIPLIER = 0.5          # 閾値ちょうどの最小倍率 (滑らかな入りのため)

MAX_POSITIONS = 5             # 1アルゴあたり最大ポジション数
MAX_JP_POSITIONS = 2          # うちJP株の上限（集中リスク回避）

IND_NAMES = ["RSI", "MACD", "BB", "EMA200", "MOM3M", "Stoch", "CCI", "52WK"]


def is_jp(ticker: str) -> bool:
    return ticker.endswith(".T")


def commission_for(ticker: str) -> float:
    return 0.0015 if is_jp(ticker) else 0.0016


def base_position_for(ticker: str) -> float:
    return BASE_POSITION_JPY if is_jp(ticker) else BASE_POSITION_USD


def capital_key_for(ticker: str) -> str:
    return "capital_jpy" if is_jp(ticker) else "capital_usd"


def pnl_key_for(ticker: str) -> str:
    return "closed_pnl_jpy" if is_jp(ticker) else "closed_pnl_usd"


# ── Indicator calculation (copied from backtest/strategy_search.py) ──────────

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

# ── Score functions (swing / trend mode) ─────────────────────────────────────

def score_rsi_trend(r: np.ndarray) -> np.ndarray:
    s = np.where(r < 25,  2.,
        np.where(r < 35,  5. + (r - 25.) / 10. * 10.,
        np.where(r < 42,  15. + (r - 35.) / 7. * 8.,
        np.where(r < 52,  23. + (52. - r) / 10. * 2.,
        np.where(r < 60,  16. + (60. - r) / 8. * 7.,
        np.where(r < 72,   7. + (72. - r) / 12. * 9.,
                            0.))))))
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

def score_bb_trend(pb: np.ndarray) -> np.ndarray:
    s = np.where(pb < 0.,   4.,
        np.where(pb < 0.15,  7. + (0.15 - pb) / 0.15 * 5.,
        np.where(pb < 0.28, 12. + (0.28 - pb) / 0.13 * 8.,
        np.where(pb < 0.45, 20. + (0.45 - pb) / 0.17 * 5.,
        np.where(pb < 0.55, 15.,
        np.where(pb < 0.72,  8. + (0.72 - pb) / 0.17 * 7.,
        np.where(pb < 0.87,  3. + (0.87 - pb) / 0.15 * 5.,
                              0.)))))))
    return np.clip(np.where(np.isnan(pb), 0., s), 0., 25.)

def score_ema200_trend(c: np.ndarray, ema200: np.ndarray) -> np.ndarray:
    v = ~np.isnan(ema200) & (ema200 > 0)
    dist = np.where(v, (c / np.where(ema200 > 0, ema200, 1.) - 1.) * 100., 0.)
    s = np.where(dist < 0,    0.,
        np.where(dist < 3,   22.,
        np.where(dist < 8,   18.,
        np.where(dist < 15,  12.,
        np.where(dist < 25,   6.,
        np.where(dist < 40,   2., 0.))))))
    pema10 = np.roll(ema200, 10)
    pema10[:10] = ema200[min(10, len(ema200) - 1)]
    slope_pct = np.where(v & (pema10 > 0), (ema200 - pema10) / pema10 * 100., 0.)
    slope_bonus = np.clip(slope_pct * 5., 0., 7.)
    return np.clip(np.where(v, s + slope_bonus, 0.), 0., 25.)

def score_mom3m_trend(c: np.ndarray, period: int = 63) -> np.ndarray:
    T = len(c)
    roc = np.full(T, np.nan)
    prev = c[:T - period]
    roc[period:] = np.where(prev > 0, (c[period:] / prev - 1.) * 100., np.nan)
    s = np.where(roc < -30,  1.,
        np.where(roc < -20,  7. + (-roc - 20.) / 10. * 7.,
        np.where(roc < -10, 14. + (-roc - 10.) / 10. * 6.,
        np.where(roc < 0,   20. + (-roc) / 10. * 5.,
        np.where(roc < 10,  15. - roc / 10. * 5.,
        np.where(roc < 20,   8. - (roc - 10.) / 10. * 5.,
                              2.))))))
    return np.clip(np.where(np.isnan(roc), 0., s), 0., 25.)

def score_stoch_trend(k: np.ndarray, d: np.ndarray) -> np.ndarray:
    s = np.where(k < 20,   4.,
        np.where(k < 30,   9. + (k - 20.) / 10. * 8.,
        np.where(k < 40,  17. + (k - 30.) / 10. * 5.,
        np.where(k < 55,  22. + (55. - k) / 15. * 3.,
        np.where(k < 65,  15. + (65. - k) / 10. * 7.,
        np.where(k < 78,   6. + (78. - k) / 13. * 9.,
                            0.))))))
    pk = np.roll(k, 1); pd_ = np.roll(d, 1)
    gc = ~np.isnan(k) & ~np.isnan(d) & (k > d) & (pk <= pd_)
    gc[0] = False
    s = np.where(gc, s + 4., s)
    return np.clip(np.where(np.isnan(k), 0., s), 0., 25.)

def score_cci_trend(cci: np.ndarray) -> np.ndarray:
    s = np.where(cci < -200,  2.,
        np.where(cci < -100,  7. + (cci + 200.) / 100. * 8.,
        np.where(cci < -50,  15. + (cci + 100.) / 50. * 5.,
        np.where(cci < 0,    20. + cci / 50. * 5.,
        np.where(cci < 100,  12. + (100. - cci) / 100. * 8.,
        np.where(cci < 200,   4. + (200. - cci) / 100. * 8.,
                               0.))))))
    return np.clip(np.where(np.isnan(cci), 0., s), 0., 25.)

def score_52wk_trend(c: np.ndarray, h: np.ndarray, lo: np.ndarray,
                     period: int = 252) -> np.ndarray:
    T = len(c)
    pos = np.full(T, np.nan)
    for i in range(period - 1, T):
        hi52 = h[i - period + 1:i + 1].max()
        lo52 = lo[i - period + 1:i + 1].min()
        rng  = hi52 - lo52
        pos[i] = (c[i] - lo52) / rng if rng > 1e-10 else 0.5
    s = np.where(pos < 0.15,  1.,
        np.where(pos < 0.30,  7. + (pos - 0.15) / 0.15 * 9.,
        np.where(pos < 0.45, 16. + (pos - 0.30) / 0.15 * 6.,
        np.where(pos < 0.60, 22. + (0.60 - pos) / 0.15 * 3.,
        np.where(pos < 0.75, 14. + (0.75 - pos) / 0.15 * 8.,
        np.where(pos < 0.87,  6. + (0.87 - pos) / 0.12 * 8.,
                               2.))))))
    return np.clip(np.where(np.isnan(pos), 0., s), 0., 25.)

# ── Core logic ────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], days: int = 450) -> dict:
    """days=450 calendar days ≈ 321 trading days — ensures 52WK(252) and EMA200 warmup."""
    end = datetime.today()
    start = end - timedelta(days=days)
    result = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            if df is None or len(df) < 60:
                print(f"  [SKIP] {ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
                continue
            last_date = date.fromisoformat(str(df.index[-1])[:10])
            if (date.today() - last_date).days > 4:
                print(f"  [SKIP] {ticker}: stale data (last={last_date}, likely holiday gap)")
                continue
            result[ticker] = {
                "closes":     df["Close"].values.astype(float),
                "highs":      df["High"].values.astype(float),
                "lows":       df["Low"].values.astype(float),
                "volumes":    df["Volume"].values.astype(float),
                "dates":      [str(d)[:10] for d in df.index],
                "last_close": float(df["Close"].iloc[-1]),
                "last_date":  str(df.index[-1])[:10],
            }
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")
    return result


def compute_scores(pd_: dict) -> np.ndarray:
    c, h, lo = pd_["closes"], pd_["highs"], pd_["lows"]
    rsi_v = calc_rsi(c, 14)
    ml, sl, hl = calc_macd(c, 12, 26, 9)
    pb = calc_bb(c, 20)
    ema200 = _ema(c, 200)
    sk, sd = calc_stoch(c, h, lo, 14, 3)
    cci = calc_cci(c, h, lo, 20)
    return np.stack([
        score_rsi_trend(rsi_v),
        score_macd(ml, sl, hl),
        score_bb_trend(pb),
        score_ema200_trend(c, ema200),
        score_mom3m_trend(c, 63),
        score_stoch_trend(sk, sd),
        score_cci_trend(cci),
        score_52wk_trend(c, h, lo, 252),
    ], axis=0)  # shape (8, T)


def composite_score(ind_scores: np.ndarray, weights: dict) -> float:
    w = np.array([weights.get(n, 0.0) for n in IND_NAMES])
    wsum = w.sum()
    if wsum <= 0:
        return 0.0
    last = ind_scores[:, -1]
    return float(np.dot(w, last) / wsum * 100.0 / 25.0)


def calc_position_multiplier(score: float, threshold: float) -> float:
    """MIN_MULTIPLIER at threshold → MAX_MULTIPLIER at score=100. Smooth entry."""
    if score <= threshold:
        return 0.0
    ratio = (score - threshold) / (100.0 - threshold)
    return MIN_MULTIPLIER + (MAX_MULTIPLIER - MIN_MULTIPLIER) * ratio


def open_position(algo: dict, ticker: str, price: float, today_str: str,
                  score: float, positions: dict, logs: list):
    aid = str(algo["id"])
    state = positions[aid]
    ccy = "JPY" if is_jp(ticker) else "USD"
    cap_key = capital_key_for(ticker)
    base = base_position_for(ticker)
    mult = calc_position_multiplier(score, algo["threshold"])
    position_size = base * mult
    shares = math.floor(position_size / price)
    if shares <= 0:
        return
    comm = commission_for(ticker)
    cost = shares * price * (1 + comm)
    if cost > state[cap_key]:
        return
    state[cap_key] -= cost
    state["open"][ticker] = {
        "entry_price": price,
        "shares": shares,
        "entry_date": today_str,
        "score": round(score, 2),
        "multiplier": round(mult, 2),
        "currency": ccy,
        "target_pct": algo["target_pct"],
        "stop_pct": algo["stop_pct"],
        "trail_pct": algo["trail_pct"],
        "max_price": price,
    }
    logs.append({
        "algo_id": algo["id"],
        "algo_name": algo["name"],
        "ticker": ticker,
        "action": "BUY",
        "price": round(price, 4),
        "shares": shares,
        "date": today_str,
        "score": round(score, 2),
        "multiplier": round(mult, 2),
        "currency": ccy,
        "pnl": None,
        "return_pct": None,
    })
    sym = "¥" if ccy == "JPY" else "$"
    print(f"  BUY  {ticker} @ {price:,.0f}{sym}  x{shares}  score={score:.1f}  mult={mult:.1f}x  algo={algo['name']}")


def close_position(algo: dict, ticker: str, price: float, today_str: str,
                   reason: str, positions: dict, logs: list):
    aid = str(algo["id"])
    pos = positions[aid]["open"].pop(ticker)
    ccy = pos.get("currency", "USD")
    comm = commission_for(ticker)
    cap_key = capital_key_for(ticker)
    pnl_key = pnl_key_for(ticker)
    proceeds = pos["shares"] * price * (1 - comm)
    cost_basis = pos["shares"] * pos["entry_price"] * (1 + comm)
    pnl = proceeds - cost_basis
    ret_pct = (price / pos["entry_price"] - 1) * 100
    positions[aid][cap_key] += proceeds
    positions[aid][pnl_key] += pnl
    positions[aid]["total_trades"] += 1
    if pnl > 0:
        positions[aid]["winning_trades"] += 1
    logs.append({
        "algo_id": algo["id"],
        "algo_name": algo["name"],
        "ticker": ticker,
        "action": "SELL",
        "price": round(price, 4),
        "shares": pos["shares"],
        "date": today_str,
        "score": None,
        "multiplier": pos.get("multiplier"),
        "currency": ccy,
        "pnl": round(pnl, 2),
        "return_pct": round(ret_pct, 2),
        "reason": reason,
    })
    sym = "¥" if ccy == "JPY" else "$"
    print(f"  SELL {ticker} @ {price:,.0f}{sym}  pnl={pnl:+,.0f}{sym}  ret={ret_pct:+.1f}%  [{reason}]  algo={algo['name']}")


def check_exits(algo: dict, price_data: dict, today_str: str,
                positions: dict, logs: list):
    aid = str(algo["id"])
    for ticker, pos in list(positions[aid]["open"].items()):
        if ticker not in price_data:
            continue
        current = price_data[ticker]["last_close"]
        entry = pos["entry_price"]
        ret = (current - entry) / entry
        days_held = (date.fromisoformat(today_str) - date.fromisoformat(pos["entry_date"])).days

        if pos["max_price"] < current:
            pos["max_price"] = current

        reason = None
        if pos["target_pct"] and ret >= pos["target_pct"]:
            reason = "target"
        elif pos["stop_pct"] and ret <= -pos["stop_pct"]:
            reason = "stop"
        elif pos["trail_pct"]:
            trail_drop = (pos["max_price"] - current) / pos["max_price"]
            if trail_drop >= pos["trail_pct"]:
                reason = "trail" if ret >= 0 else "trail_stop"
        if reason is None and days_held >= algo["max_hold_days"]:
            reason = "max_hold"

        if reason:
            close_position(algo, ticker, current, today_str, reason, positions, logs)


def run_daily(dry_run: bool = False):
    today_str = date.today().isoformat()
    print(f"\n=== Paper Trading Run: {today_str} (dry_run={dry_run}) ===")

    algos = json.loads((TRADING_DIR / "algorithms.json").read_text())
    positions = json.loads((TRADING_DIR / "positions.json").read_text())
    logs = json.loads((TRADING_DIR / "trade_log.json").read_text())

    print("Fetching price data (days=450 for 52WK indicator warmup)...")
    price_data = fetch_prices(ALL_TICKERS, days=450)
    print(f"Fetched {len(price_data)} tickers "
          f"(US: {sum(1 for t in price_data if not is_jp(t))}, "
          f"JP: {sum(1 for t in price_data if is_jp(t))})")

    print("Computing indicator scores...")
    ind_scores: dict[str, np.ndarray] = {}
    for ticker, pd_ in price_data.items():
        try:
            ind_scores[ticker] = compute_scores(pd_)
        except Exception as e:
            print(f"  [SCORE ERROR] {ticker}: {e}")

    for algo in algos:
        aid = str(algo["id"])
        print(f"\n[Algo {aid}] {algo['name']}")

        check_exits(algo, price_data, today_str, positions, logs)

        open_count = len(positions[aid]["open"])
        jp_open = sum(1 for t in positions[aid]["open"] if is_jp(t))

        if open_count >= MAX_POSITIONS:
            print(f"  Max positions reached ({MAX_POSITIONS}), skip entries")
            continue

        candidates = []
        for ticker in ALL_TICKERS:
            if ticker not in ind_scores:
                continue
            if ticker in positions[aid]["open"]:
                continue
            score = composite_score(ind_scores[ticker], algo["weights"])
            if score >= algo["threshold"]:
                candidates.append((score, ticker))

        candidates.sort(reverse=True)
        slots = MAX_POSITIONS - open_count
        entered = 0
        for score, ticker in candidates:
            if entered >= slots:
                break
            if is_jp(ticker) and jp_open >= MAX_JP_POSITIONS:
                continue
            if not dry_run:
                prev_open = len(positions[aid]["open"])
                open_position(algo, ticker, price_data[ticker]["last_close"],
                              today_str, score, positions, logs)
                if len(positions[aid]["open"]) > prev_open:
                    entered += 1
                    if is_jp(ticker):
                        jp_open += 1
            else:
                price = price_data[ticker]["last_close"]
                mult = calc_position_multiplier(score, algo["threshold"])
                sym = "¥" if is_jp(ticker) else "$"
                print(f"  [DRY] BUY {ticker} @ {price:,.0f}{sym}  score={score:.1f}  mult={mult:.1f}x")
                entered += 1

    print("\n=== Summary ===")
    for algo in algos:
        aid = str(algo["id"])
        st = positions[aid]
        usd_open = sum(
            price_data[t]["last_close"] * pos["shares"]
            for t, pos in st["open"].items()
            if t in price_data and not is_jp(t)
        )
        jpy_open = sum(
            price_data[t]["last_close"] * pos["shares"]
            for t, pos in st["open"].items()
            if t in price_data and is_jp(t)
        )
        usd_eq = st["capital_usd"] + usd_open
        jpy_eq = st["capital_jpy"] + jpy_open
        usd_pnl = usd_eq - INITIAL_CAPITAL_USD
        jpy_pnl = (jpy_eq - INITIAL_CAPITAL_JPY) / 1000
        print(f"  [{aid}] {algo['name']:20s}  "
              f"USD eq={usd_eq:>8,.0f}({usd_pnl:+,.0f})  "
              f"JPY eq={jpy_eq:>10,.0f}({jpy_pnl:+,.0f}k)  "
              f"open={len(st['open'])}  trades={st['total_trades']}")

    if not dry_run:
        (TRADING_DIR / "positions.json").write_text(json.dumps(positions, indent=2))
        (TRADING_DIR / "trade_log.json").write_text(json.dumps(logs, indent=2))
        print("\nSaved positions.json and trade_log.json")
        commit_results()
    else:
        print("\n[DRY RUN] No files written.")


def commit_results():
    try:
        subprocess.run(["git", "config", "user.email", "action@github.com"], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "config", "user.name", "GitHub Actions"], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "add",
                        "trading/positions.json",
                        "trading/trade_log.json"], cwd=BASE_DIR, check=True)
        result = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=BASE_DIR)
        if result.returncode != 0:
            msg = f"📈 paper trade update {date.today().isoformat()}"
            subprocess.run(["git", "commit", "-m", msg], cwd=BASE_DIR, check=True)
            subprocess.run(["git", "push"], cwd=BASE_DIR, check=True)
            print("Committed and pushed.")
        else:
            print("No changes to commit.")
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    run_daily(dry_run=dry)
