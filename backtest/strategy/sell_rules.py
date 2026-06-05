"""
strategy/sell_rules.py — 売りルールエンジン

各銘柄の OHLCV から「各バー t でエントリーした場合の実現リターン」を事前計算する。
バイアス排除: エントリー価格は翌バーの始値 (open[t+1])。

【スイング売りルール】
  target5_stop3   : 利確+5% / ストップ-3%
  target10_stop5  : 利確+10% / ストップ-5%
  target15_stop5  : 利確+15% / ストップ-5%
  target20_stop7  : 利確+20% / ストップ-7%
  target25_stop10 : 利確+25% / ストップ-10%
  trail_5pct      : トレーリングストップ 5%

【デイトレ売りルール (10min足)】
  hold_2b         : 固定保有 2バー (20分)
  hold_4b         : 固定保有 4バー (40分)
  hold_6b         : 固定保有 6バー (1時間)
  hold_8b         : 固定保有 8バー (80分)
  target3_stop2   : 利確+3% / ストップ-2%
  target5_stop3   : 利確+5% / ストップ-3%
"""
from __future__ import annotations

import numpy as np

# ── 売りルール定義 ────────────────────────────────────────────────────────────

# 売りルール名 → (最大保有バー数,)。価格評価に使用。
SWING_SELL_RULES = [
    "target5_stop3", "target10_stop5", "target15_stop5",
    "target20_stop7", "target25_stop10", "trail_5pct",
]
DAY_SELL_RULES = [
    "hold_2b", "hold_4b", "hold_6b", "hold_8b",
    "target3_stop2", "target5_stop3",
]

# 売りルール → 最大保有バー数 (Sharpe年率化の代表値として使用しない)
SWING_SELL_HOLD = {
    "target5_stop3":   15,
    "target10_stop5":  25,
    "target15_stop5":  35,
    "target20_stop7":  45,
    "target25_stop10": 60,
    "trail_5pct":      30,
}
DAY_SELL_HOLD = {
    "hold_2b":      2,
    "hold_4b":      4,
    "hold_6b":      6,
    "hold_8b":      8,
    "target3_stop2": 4,
    "target5_stop3": 6,
}

# 日本語説明
SELL_RULE_JA = {
    "target5_stop3":   "利確+5% / ストップ-3%",
    "target10_stop5":  "利確+10% / ストップ-5%",
    "target15_stop5":  "利確+15% / ストップ-5%",
    "target20_stop7":  "利確+20% / ストップ-7%",
    "target25_stop10": "利確+25% / ストップ-10%",
    "trail_5pct":      "トレーリングストップ 5%",
    "hold_2b":         "固定保有 20min",
    "hold_4b":         "固定保有 40min",
    "hold_6b":         "固定保有 1時間",
    "hold_8b":         "固定保有 80min",
    "target3_stop2":   "利確+3% / ストップ-2%",
}


# ── コア計算 ──────────────────────────────────────────────────────────────────

def precompute_sell_outcomes(
    closes:     np.ndarray,
    opens:      np.ndarray,
    highs:      np.ndarray,
    lows:       np.ndarray,
    cost_rate:  float,
    sell_rules: list[str],
    max_hold:   int,
) -> dict[str, np.ndarray]:
    """
    各バー t でエントリーした場合の実現リターンを全売りルールで事前計算する。

    Parameters
    ----------
    closes, opens, highs, lows : np.ndarray (T,)
        OHLCV の各系列
    cost_rate : float
        往復取引コスト (例: 0.0016)
    sell_rules : list[str]
        計算する売りルール名リスト
    max_hold : int
        最大保有バー数 (この先を見ない上限)

    Returns
    -------
    dict[str, np.ndarray]
        {rule_name: (T,) の実現リターン配列}
        エントリーできない/保有中のバーは NaN。
    """
    T = len(closes)
    result: dict[str, np.ndarray] = {}

    for rule in sell_rules:
        outcomes = np.full(T, np.nan, dtype=np.float32)

        if rule.startswith("hold_"):
            n = int(rule.split("_")[1].replace("b", ""))
            for t in range(T - n - 1):
                entry = opens[t + 1]
                if entry <= 0 or np.isnan(entry):
                    continue
                exit_p = closes[t + n] if t + n < T else closes[-1]
                if np.isnan(exit_p):
                    continue
                ret = (exit_p - entry) / entry - cost_rate
                outcomes[t] = float(ret)

        elif rule.startswith("target") and "stop" in rule:
            # "target5_stop3" → ["5", "3"]
            parts   = rule.replace("target", "").split("_stop")
            tgt_pct = float(parts[0]) / 100.0
            stp_pct = float(parts[1]) / 100.0

            for t in range(T - 2):
                entry = opens[t + 1]
                if entry <= 0 or np.isnan(entry):
                    continue
                tgt_price = entry * (1 + tgt_pct)
                stp_price = entry * (1 - stp_pct)
                ret = None

                for j in range(t + 1, min(t + 1 + max_hold, T)):
                    h = highs[j] if not np.isnan(highs[j]) else closes[j]
                    l = lows[j]  if not np.isnan(lows[j])  else closes[j]
                    # ストップが先 (保守的: 悪い側を先に評価)
                    if l <= stp_price:
                        ret = -stp_pct - cost_rate
                        break
                    if h >= tgt_price:
                        ret = tgt_pct - cost_rate
                        break

                if ret is None:
                    # 最大保有日数経過 → 終値で決済
                    j_exit = min(t + max_hold, T - 1)
                    exit_p = closes[j_exit]
                    if not np.isnan(exit_p) and entry > 0:
                        ret = (exit_p - entry) / entry - cost_rate
                if ret is not None:
                    outcomes[t] = float(ret)

        elif rule == "trail_5pct":
            trail_pct = 0.05
            for t in range(T - 2):
                entry = opens[t + 1]
                if entry <= 0 or np.isnan(entry):
                    continue
                peak  = entry
                ret   = None
                for j in range(t + 1, min(t + 1 + max_hold, T)):
                    h = highs[j] if not np.isnan(highs[j]) else closes[j]
                    l = lows[j]  if not np.isnan(lows[j])  else closes[j]
                    if h > peak:
                        peak = float(h)
                    trail_stop = peak * (1 - trail_pct)
                    if l <= trail_stop:
                        ret = (trail_stop - entry) / entry - cost_rate
                        break
                if ret is None:
                    j_exit = min(t + max_hold, T - 1)
                    exit_p = closes[j_exit]
                    if not np.isnan(exit_p) and entry > 0:
                        ret = (exit_p - entry) / entry - cost_rate
                if ret is not None:
                    outcomes[t] = float(ret)

        result[rule] = outcomes

    return result
