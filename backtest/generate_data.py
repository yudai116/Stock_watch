#!/usr/bin/env python3
"""
generate_data.py — 2015-2024 実績ベース OHLCV シミュレーター v3

修正点:
  - 伊藤補正はベースボラで計算（eff_sigma で Itô 補正するとクラッシュ時に過剰補正）
  - 市場イベントの drift 強度を実績終値から逆算して調整
  - GARCH クリップを 1.8x に制限してバブル・暴落の連鎖を防止

最終価格の目安 (実際の2024年末):
  NVDA  $5→$800+  MSFT $45→$430  ASML $80→$800
  AMD   $3→$150   8035.T ¥6000→¥24000
"""
import json, time
from pathlib import Path
import numpy as np

OUT  = Path(__file__).parent / "price_data.json"

# ── 銘柄パラメータ (年率対数リターン, 年率ボラ) ──────────────────────────────
TICKER_PARAMS = {
    # CAGR(実績): NVDA~61%, MSFT~22%, ASML~24%, AMD~36%
    # mu は年率 log return (ln(終値/始値)/10年)
    "NVDA":   {"mu": 0.615, "sigma": 0.55, "S0":   5.0,   "beta": 1.55},
    "ASML":   {"mu": 0.240, "sigma": 0.27, "S0":  80.0,   "beta": 1.15},
    "MSFT":   {"mu": 0.220, "sigma": 0.21, "S0":  45.0,   "beta": 1.00},
    "8035.T": {"mu": 0.210, "sigma": 0.31, "S0":6000.0,   "beta": 1.10},
    "MRVL":   {"mu": 0.180, "sigma": 0.43, "S0":   8.0,   "beta": 1.35},
    "SMCI":   {"mu": 0.380, "sigma": 0.65, "S0":  12.0,   "beta": 1.45},
    "CRWD":   {"mu": 0.340, "sigma": 0.50, "S0":  60.0,   "beta": 1.30},
    "SOUN":   {"mu": 0.040, "sigma": 0.85, "S0":   8.0,   "beta": 1.15},
    "IONQ":   {"mu": 0.075, "sigma": 0.80, "S0":  10.0,   "beta": 1.05},
    "AMD":    {"mu": 0.360, "sigma": 0.48, "S0":   3.0,   "beta": 1.50},
    "TSM":    {"mu": 0.175, "sigma": 0.26, "S0":  20.0,   "beta": 1.00},
    "AMAT":   {"mu": 0.245, "sigma": 0.33, "S0":  25.0,   "beta": 1.25},
    "LRCX":   {"mu": 0.245, "sigma": 0.30, "S0":  70.0,   "beta": 1.25},
    "KLAC":   {"mu": 0.245, "sigma": 0.28, "S0":  60.0,   "beta": 1.20},
    "6857.T": {"mu": 0.195, "sigma": 0.32, "S0":2000.0,   "beta": 1.10},
}

# ── 市場イベント ─────────────────────────────────────────────────────────────
# (start_bar, end_bar, extra_annual_log_drift, vol_multiplier)
# extra_annual_log_drift: 年率換算の追加 log drift（個別銘柄の beta で調整）
# β=1.0 の銘柄への影響 = extra / 252 per day
# 期間・強度は SPY・NVDA 等の実績終値から逆算
MARKET_EVENTS = [
    # 2015/8 中国ショック: SPY -12% over 25 days → -0.5%/day log → ed=-1.2
    ( 143,  168,  -1.2,  2.2),
    # 2015/12-2016/2 底打ち: 緩やかな反発
    ( 168,  280,   0.25, 0.9),
    # 2016-2018 強気相場
    ( 280,  955,   0.15, 0.85),
    # 2018/Q4 急落: SPY -20% over 65 days → -0.35%/day → ed=-0.88
    ( 955, 1020,  -0.88, 1.8),
    # 2019 回復
    (1020, 1258,   0.30, 0.85),
    # 2020/2月後半 COVID 初期不安: 10日間
    (1258, 1268,  -1.00, 2.0),
    # 2020/3月 COVID 大暴落: SPY -34% over 24 days → -1.74%/day → ed=-1.74
    (1268, 1292,  -1.74, 3.5),
    # 2020/3末-4月 V字反発: +25% over 20 days → +1.25%/day → ed=+3.15
    (1292, 1312,   3.15, 2.5),
    # 2020/5-2021/11 QE 強気相場
    (1312, 1680,   0.35, 0.85),
    # 2022 利上げ相場: SPY -25% over 230 days → -0.12%/day → ed=-0.30
    (1680, 1910,  -0.30, 1.60),
    # 2022/10-2023/1 底打ち反発
    (1910, 1972,   0.50, 1.30),
    # 2023 AI ブーム回復
    (1972, 2230,   0.25, 0.88),
    # 2024 継続上昇
    (2230, 2520,   0.18, 0.90),
]

# NVDA 固有 AI ブーム上乗せ drift
NVDA_EXTRA = [
    (1990, 2100,  1.8, 1.10),  # 2023 ChatGPT 需要
    (2100, 2350,  1.2, 1.05),  # 2024 Blackwell
]


def make_trading_days(start: str, n: int) -> list[str]:
    from datetime import date, timedelta
    d, days = date.fromisoformat(start), []
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(1)
    return days


def _get_event(t: int, extra=None) -> tuple[float, float]:
    ed, vm = 0.0, 1.0
    for (s, e, d, v) in MARKET_EVENTS:
        if s <= t < e:
            ed, vm = d, v; break
    if extra:
        for (s, e, d, v) in extra:
            if s <= t < e:
                ed += d; vm = max(vm, v); break
    return ed, vm


def generate_ohlcv(ticker: str, p: dict, days: list[str], rng) -> list[dict]:
    T        = len(days)
    mu_d     = p["mu"]    / 252        # 日次 log drift
    sigma_d  = p["sigma"] / 252**0.5   # 日次ボラ
    beta     = p["beta"]
    extra    = NVDA_EXTRA if ticker == "NVDA" else None

    # 標準化 t 分布 (df=7, std=1)
    df = 7
    z_mkt  = rng.standard_t(df, T) / (df/(df-2))**0.5
    z_idio = rng.standard_t(df, T) / (df/(df-2))**0.5

    closes = np.empty(T); closes[0] = p["S0"]
    garch  = 1.0   # GARCH 状態 (ボラ乗数)

    for t in range(1, T):
        ed, vm = _get_event(t, extra)
        # 合成 z (市場 + 固有)
        rho = beta / (1 + beta**2)**0.5
        z   = rho * z_mkt[t] + (1-rho**2)**0.5 * z_idio[t]

        eff_vm   = vm * min(garch, 1.8)          # vol クリップ
        eff_sig  = sigma_d * eff_vm              # 実効ボラ

        # 伊藤補正はベースボラで（eff_sig で補正するとクラッシュ時に過剰減衰）
        drift    = mu_d - 0.5 * sigma_d**2 + (ed / 252) * beta
        log_ret  = drift + eff_sig * z
        closes[t] = max(closes[t-1] * np.exp(log_ret), 0.005)

        # GARCH(1,1): 絶対値ショックに緩やかに反応、減衰も穏やか
        abs_z    = abs(z)
        garch    = 0.92 * garch + 0.08 * abs_z
        garch    = float(np.clip(garch, 0.75, 1.8))

    # Open: overnight gap (前日終値±)
    z_gap  = rng.standard_normal(T)
    opens  = np.empty(T); opens[0] = p["S0"]
    for t in range(1, T):
        _, vm = _get_event(t, extra)
        gap = sigma_d * vm * 0.38 * z_gap[t]
        opens[t] = max(closes[t-1] * np.exp(gap), 0.005)

    # High / Low: intraday excursion
    z_hl = np.abs(rng.standard_normal((T, 2)))
    highs, lows = np.empty(T), np.empty(T)
    for t in range(T):
        _, vm = _get_event(t, extra)
        exc  = sigma_d * vm * 1.35
        thi  = max(opens[t], closes[t])
        tlo  = min(opens[t], closes[t])
        highs[t] = max(thi * np.exp(exc * z_hl[t,0]), thi)
        lows[t]  = min(tlo * np.exp(-exc * z_hl[t,1]), tlo)

    base_v = 800_000 if "T" not in ticker else 300_000
    vols   = (rng.lognormal(0, 0.4, T) * base_v).astype(int)

    dec = 0 if "T" in ticker else 4
    return [{
        "date":   days[t],
        "close":  round(float(closes[t]), dec),
        "open":   round(float(opens[t]),  dec),
        "high":   round(float(highs[t]),  dec),
        "low":    round(float(lows[t]),   dec),
        "volume": int(vols[t]),
    } for t in range(T)]


def main():
    t0 = time.time()
    print("=" * 62)
    print("  OHLCV シミュレーター v3  (2015-2024 実績パラメータ)")
    print("=" * 62)

    days = make_trading_days("2015-01-01", 2520)
    print(f"営業日: {len(days)}  ({days[0]} → {days[-1]})\n")

    # 複数シードで安定したシードを見つける
    best_seed, best_score = 42, -np.inf
    for seed in [42, 123, 7, 2024, 999]:
        rng = np.random.default_rng(seed)
        total = 0.0
        for ticker, p in list(TICKER_PARAMS.items())[:5]:
            rows = generate_ohlcv(ticker, p, days, rng)
            sf = rows[-1]["close"]; s0 = rows[0]["close"]
            cagr = (sf/s0)**(1/10) - 1
            target_cagr = np.exp(p["mu"]) - 1
            total -= (cagr - target_cagr)**2
        if total > best_score:
            best_score, best_seed = total, seed

    print(f"使用シード: {best_seed}\n")
    rng = np.random.default_rng(best_seed)

    result = {}
    print(f"{'銘柄':10s} {'開始':>9s} {'終値':>10s} {'実CAGR':>8s} {'目標':>7s} {'判定':>4s}")
    print("-" * 58)
    ok_count = 0
    for ticker, p in TICKER_PARAMS.items():
        rows = generate_ohlcv(ticker, p, days, rng)
        s0, sf = rows[0]["close"], rows[-1]["close"]
        cagr = (sf/s0)**(1/10) - 1
        tgt  = np.exp(p["mu"]) - 1
        ok   = abs(cagr - tgt) < 0.35
        if ok: ok_count += 1
        result[ticker] = rows
        print(f"{ticker:10s} {s0:>9.1f} {sf:>10.1f} "
              f"{cagr*100:>7.1f}% {tgt*100:>6.1f}% {'✓' if ok else '△'}")

    OUT.write_text(json.dumps(result, indent=0))
    print(f"\n✓ 保存: {OUT.name}  ({OUT.stat().st_size//1024} KB)")
    print(f"  銘柄数: {len(result)}  適合率: {ok_count}/{len(result)}")
    print(f"  生成時間: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
