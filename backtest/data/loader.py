"""
data/loader.py — price_data.json / price_data_intraday.json → strategy 層 ticker_data 変換

【ticker_data 形式 (strategy 層が要求)】
  {
    ticker: {
      "ind_scores":    np.ndarray (n_ind, T),  各指標スコア [0,25]
      "sell_outcomes": {rule: np.ndarray (T,)}, 各ルールの実現リターン
      "vol_ok":        np.ndarray (T,) bool,   ボラ計算に十分なデータがある行
      "closes":        np.ndarray (T,),
      "opens":         np.ndarray (T,),
      "highs":         np.ndarray (T,),
      "lows":          np.ndarray (T,),
      "volumes":       np.ndarray (T,),
      "returns":       np.ndarray (T,),
      "dates":         list[str],
    }
  }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from backtest.config import (
    PRICE_DATA_SWING, PRICE_DATA_DAY,
    COST_RATE_SWING, COST_RATE_DAY,
    SWING_MAX_HOLD, DAY_MAX_HOLD,
)
from backtest.strategy.compute import compute_swing_scores, compute_day_scores
from backtest.strategy.sell_rules import (
    SWING_SELL_RULES, DAY_SELL_RULES,
    precompute_sell_outcomes,
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _to_arrays(raw) -> dict:
    """
    生データ → numpy 配列変換 + log return 追加。

    対応フォーマット:
      - list[dict]:  [{date, open, high, low, close, volume}, ...]
      - dict:        {dates:[...], open:[...], close:[...], ...}
    """
    if isinstance(raw, list):
        # list-of-records フォーマット (既存 price_data.json)
        raw.sort(key=lambda r: r.get("date", ""))
        dates   = [r.get("date", "") for r in raw]
        closes  = np.array([r.get("close",  0.0) for r in raw], dtype=np.float64)
        opens   = np.array([r.get("open",   0.0) for r in raw], dtype=np.float64)
        highs   = np.array([r.get("high",   0.0) for r in raw], dtype=np.float64)
        lows    = np.array([r.get("low",    0.0) for r in raw], dtype=np.float64)
        volumes = np.array([r.get("volume", 0.0) for r in raw], dtype=np.float64)
    else:
        # dict フォーマット (fetch_alpaca 出力)
        dates   = raw.get("dates", [])
        closes  = np.array(raw.get("close",  []), dtype=np.float64)
        opens   = np.array(raw.get("open",   []), dtype=np.float64)
        highs   = np.array(raw.get("high",   []), dtype=np.float64)
        lows    = np.array(raw.get("low",    []), dtype=np.float64)
        volumes = np.array(raw.get("volume", []), dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(
            (closes[:-1] > 0) & (closes[1:] > 0),
            np.log(closes[1:] / closes[:-1]),
            np.nan,
        )
    returns = np.concatenate([[np.nan], ret])

    return dict(
        closes=closes, opens=opens, highs=highs,
        lows=lows, volumes=volumes, returns=returns, dates=dates,
    )


def _vol_ok_mask(returns: np.ndarray, min_window: int = 20) -> np.ndarray:
    """直近 min_window 本のリターンが揃っているか判定するブールマスク"""
    mask = np.zeros(len(returns), dtype=bool)
    for i in range(min_window, len(returns)):
        seg = returns[i - min_window: i]
        mask[i] = np.sum(~np.isnan(seg)) >= min_window // 2
    return mask


# ── 公開 API ─────────────────────────────────────────────────────────────────

def load_ticker_data(
    mode: str = "swing",
    tickers: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    raw OHLCV を読み込み、基本配列のみの ticker_data を返す。
    strategy データが不要な軽量ロード（レジーム検出など）に使用。
    """
    path = PRICE_DATA_SWING if mode == "swing" else PRICE_DATA_DAY
    raw  = _load_json(path)
    if not raw:
        print(f"[loader] {path} が空または存在しません")
        return {}

    result = {}
    for ticker, data in raw.items():
        if tickers and ticker not in tickers:
            continue
        arr = _to_arrays(data)
        if len(arr["closes"]) < 30:
            continue
        result[ticker] = arr

    print(f"[loader] {mode}: {len(result)} 銘柄ロード完了")
    return result


def build_strategy_data(
    mode: str = "swing",
    tickers: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    OHLCV → strategy 層向け ticker_data（指標スコア + 売りアウトカム付き）を構築。

    Parameters
    ----------
    mode : "swing" | "day"
    tickers : 対象ティッカーリスト (None = 全ティッカー)
    verbose : ログ出力フラグ

    Returns
    -------
    dict[ticker, {ind_scores, sell_outcomes, vol_ok, closes, ...}]
    """
    path = PRICE_DATA_SWING if mode == "swing" else PRICE_DATA_DAY
    raw  = _load_json(path)
    if not raw:
        print(f"[loader] {path} が空または存在しません")
        return {}

    if mode == "swing":
        cost_rate  = COST_RATE_SWING
        max_hold   = SWING_MAX_HOLD
        sell_rules = SWING_SELL_RULES
        score_fn   = compute_swing_scores
    else:
        cost_rate  = COST_RATE_DAY
        max_hold   = DAY_MAX_HOLD
        sell_rules = DAY_SELL_RULES
        score_fn   = compute_day_scores

    result = {}
    for ticker, data in raw.items():
        if tickers and ticker not in tickers:
            continue
        arr = _to_arrays(data)
        T   = len(arr["closes"])
        min_bars = 504 if mode == "swing" else 4914  # swing: 2年日足, day: 6ヶ月10分足
        if T < min_bars:
            if verbose:
                print(f"[loader] {ticker}: データ不足 ({T}本 < {min_bars}) スキップ")
            continue

        closes  = arr["closes"]
        opens   = arr["opens"]
        highs   = arr["highs"]
        lows    = arr["lows"]
        volumes = arr["volumes"]

        # (n_ind, T) スコア行列
        try:
            ind_scores = score_fn(closes, highs, lows, volumes)
        except Exception as e:
            if verbose:
                print(f"[loader] {ticker} 指標計算エラー: {e}")
            continue

        # 売りアウトカム
        outcomes = precompute_sell_outcomes(
            closes, opens, highs, lows,
            cost_rate, sell_rules, max_hold,
        )

        # ボラ計算可能マスク
        vol_ok = _vol_ok_mask(arr["returns"])

        result[ticker] = {
            **arr,
            "ind_scores":    ind_scores,   # (n_ind, T) float32
            "sell_outcomes": outcomes,      # {rule: (T,) float32}
            "vol_ok":        vol_ok,        # (T,) bool
        }

    if verbose:
        print(f"[loader] build_strategy_data: {len(result)} 銘柄完了 ({mode})")
    return result
