"""
regime/features.py — HMM 学習用の観測変数を構築する

【Group A】マクロ/リスクセンチメント
  1. vix_level      : log(VIX) の z-スコア正規化
  2. vix_momentum   : VIX の 5日変化率
  3. vix_regime     : (VIX - VIX_20d_SMA) / VIX_20d_SMA — VIX の平均乖離
  4. credit_spread  : HYG_5d_ret − LQD_5d_ret (ハイイールド vs 投資適格)
  5. risk_sentiment : HYG の 10日リターン (リスクオン/オフ)

【Group B】テックセクター
  6. soxx_ret5      : SOXX の 5日リターン
  7. soxx_vol20     : SOXX の 20日実現ボラティリティ
  8. smh_ret5       : SMH の 5日リターン
  9. soxx_vs_spy    : SOXX_5d_ret − SPY_5d_ret (テック相対強度)
 10. spy_vol20      : SPY の 20日実現ボラティリティ

全特徴量は最終的に z-スコア正規化して返す（HMM に均等スケールを保証）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).parent.parent


def _prices(data: dict, ticker: str, key: str = "close") -> np.ndarray:
    """指定ティッカーの価格系列を numpy array で返す。存在しない場合は空配列。"""
    d = data.get(ticker, {})
    vals = d.get(key, [])
    return np.array(vals, dtype=np.float64)


def _dates(data: dict, ticker: str) -> list[str]:
    return data.get(ticker, {}).get("dates", [])


def _returns(prices: np.ndarray, window: int = 1) -> np.ndarray:
    """n 日リターン (対数)。先頭 window 個は NaN。"""
    if len(prices) < window + 1:
        return np.full(len(prices), np.nan)
    log_p = np.log(np.where(prices > 0, prices, np.nan))
    ret = np.full(len(prices), np.nan)
    ret[window:] = log_p[window:] - log_p[:-window]
    return ret


def _realized_vol(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """rolling 実現ボラティリティ (年率換算しない、比率のまま)。"""
    daily_ret = _returns(prices, 1)
    rv = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        rv[i] = float(np.nanstd(daily_ret[i - window + 1 : i + 1]))
    return rv


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """rolling mean。先頭 window-1 個は NaN。"""
    result = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        result[i] = float(np.nanmean(x[i - window + 1 : i + 1]))
    return result


def _zscore(x: np.ndarray, window: int = 252) -> np.ndarray:
    """rolling z-スコア正規化。"""
    result = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        seg = x[i - window + 1 : i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 10:
            continue
        mu  = float(np.mean(seg))
        sig = float(np.std(seg))
        if sig > 1e-10:
            result[i] = (x[i] - mu) / sig
    return result


def build_features(
    macro_data: dict,
    align_dates: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    マクロデータから HMM 観測変数行列を構築する。

    Parameters
    ----------
    macro_data : dict
        {ticker: {dates, close, ...}} 形式。fetch_macro.py の出力。
    align_dates : list[str], optional
        この日付リストに揃える（None の場合は VIX の日付を使用）。

    Returns
    -------
    X : np.ndarray, shape (T, n_features)
        正規化済み特徴量行列。NaN を含む行は除去済み。
    valid_dates : list[str]
        X の各行に対応する日付。
    """
    # 共通日付インデックス (VIX を基準)
    if align_dates is None:
        ref_ticker = next(
            (t for t in ["^VIX", "VIX", "SPY"] if t in macro_data), None
        )
        if ref_ticker is None:
            raise ValueError("macro_data に ^VIX / SPY が含まれていません")
        align_dates = _dates(macro_data, ref_ticker)

    date_to_idx = {d: i for i, d in enumerate(align_dates)}
    T = len(align_dates)

    def align_array(ticker: str, key: str = "close") -> np.ndarray:
        """macro_data[ticker] を align_dates に揃えた配列を返す。"""
        d = macro_data.get(ticker, {})
        src_dates  = d.get("dates", [])
        src_values = d.get(key, [])
        arr = np.full(T, np.nan)
        for sd, sv in zip(src_dates, src_values):
            if sd in date_to_idx:
                arr[date_to_idx[sd]] = float(sv)
        return arr

    # --- 各価格系列を align_dates 軸で取得 ---
    vix   = align_array("^VIX")
    hyg   = align_array("HYG")
    lqd   = align_array("LQD")
    soxx  = align_array("SOXX")
    smh   = align_array("SMH")
    spy   = align_array("SPY")

    # --- 特徴量計算 ---
    # Group A: マクロ/リスクセンチメント
    log_vix     = np.where(vix > 0, np.log(vix), np.nan)
    f_vix_level = _zscore(log_vix, window=252)           # 1. VIX水準

    f_vix_mom   = _returns(vix, window=5)                # 2. VIX 5日変化率
    vix_sma20   = _rolling_mean(vix, 20)
    f_vix_reg   = np.where(                              # 3. VIX 平均乖離
        vix_sma20 > 1e-6,
        (vix - vix_sma20) / vix_sma20,
        np.nan,
    )

    hyg_ret5    = _returns(hyg, window=5)
    lqd_ret5    = _returns(lqd, window=5)
    f_credit    = hyg_ret5 - lqd_ret5                   # 4. クレジットスプレッド変化
    f_hyg_risk  = _returns(hyg, window=10)              # 5. HYG 10日リターン

    # Group B: テックセクター
    soxx_ret5   = _returns(soxx, window=5)
    spy_ret5    = _returns(spy, window=5)
    f_soxx_ret5 = soxx_ret5                              # 6. SOXX 5日リターン
    f_soxx_vol  = _realized_vol(soxx, window=20)         # 7. SOXX 実現ボラ
    f_smh_ret5  = _returns(smh, window=5)                # 8. SMH 5日リターン
    f_soxx_rel  = soxx_ret5 - spy_ret5                   # 9. SOXX vs SPY 相対強度
    f_spy_vol   = _realized_vol(spy, window=20)          # 10. SPY 実現ボラ

    # --- 行列組み立て (T × 10) ---
    feature_names = [
        "vix_level", "vix_momentum", "vix_regime",
        "credit_spread", "hyg_risk_sentiment",
        "soxx_ret5", "soxx_vol20", "smh_ret5",
        "soxx_vs_spy", "spy_vol20",
    ]
    raw = np.column_stack([
        f_vix_level, f_vix_mom, f_vix_reg,
        f_credit, f_hyg_risk,
        f_soxx_ret5, f_soxx_vol, f_smh_ret5,
        f_soxx_rel, f_spy_vol,
    ])  # (T, 10)

    # NaN 行を除去（HMM は NaN 非対応）
    valid_mask  = ~np.any(np.isnan(raw), axis=1)
    X           = raw[valid_mask]
    valid_dates = [align_dates[i] for i in range(T) if valid_mask[i]]

    # 最終 z-スコア正規化（全期間統計）
    mu  = X.mean(axis=0)
    sig = X.std(axis=0)
    sig = np.where(sig < 1e-10, 1.0, sig)
    X   = (X - mu) / sig

    print(
        f"[features] 特徴量 {X.shape[1]}次元, 有効日数 {len(valid_dates)} "
        f"({valid_dates[0] if valid_dates else '?'} 〜 {valid_dates[-1] if valid_dates else '?'})"
    )
    return X, valid_dates, feature_names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parent))
    from backtest.data.fetch_macro import load_macro_data
    from backtest.config import MACRO_DATA_PATH

    macro = load_macro_data(MACRO_DATA_PATH)
    X, dates, names = build_features(macro)
    print(f"特徴量行列: {X.shape}")
    print(f"特徴量名: {names}")
    print(f"統計: mean={X.mean(0).round(3)}, std={X.std(0).round(3)}")
