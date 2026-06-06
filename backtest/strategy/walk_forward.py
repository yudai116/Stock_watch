"""
strategy/walk_forward.py — Walk-Forward 検証

T_min バー全体を (n_folds+1) 等分して fold_size を決め、
各フォールドの訓練期間で GA を走らせて OOS で評価する。

【分割方法 (Expanding Window, n_folds=6)】
  fold_size = t_holdout // (n_folds + 1)  ← 等分で最終フォールドのOOSが空にならない
  fold 0: train=[0, S0)     OOS=[S0, E0)   E0-S0 = fold_size
  fold 1: train=[0, S1)     OOS=[S1, E1)   E1-S1 = fold_size
  ...
  fold 5: train=[0, S5)     OOS=[S5, E5)   E5-S5 = fold_size

WF安定性: 1 - std(OOS_Sharpe) / (|mean(OOS_Sharpe)| + ε)
  1.0=完全安定, <0=フォールドごとに悪化
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def wf_splits(
    T_min: int,
    t_holdout: int,
    n_folds: int = 6,
) -> list[tuple[int, int, int]]:
    """
    Walk-Forward の (train_end, oos_start, oos_end) タプルリストを返す。

    Parameters
    ----------
    T_min : int
        全データ長 (最短ティッカーのバー数)
    t_holdout : int
        最終ホールドアウト開始インデックス (= T_min * 0.80)
    n_folds : int
        フォールド数

    Returns
    -------
    list of (train_end, oos_start, oos_end)
    """
    oos_total = t_holdout
    fold_size = oos_total // (n_folds + 1)  # n_folds+1で等分→最終foldのOOSが空にならない
    splits = []
    for k in range(n_folds):
        oos_start = (k + 1) * fold_size
        oos_end   = min((k + 2) * fold_size, oos_total)
        train_end = oos_start
        if train_end < 20 or oos_end <= oos_start:
            continue
        splits.append((train_end, oos_start, oos_end))
    return splits


def run_walk_forward(
    ticker_data: dict,
    sell_rules: list[str],
    thresholds: list[int],
    T_min: int,
    t_holdout: int,
    bars_per_year: int,
    min_trades_oos: int = 10,
    crossover_only: bool = False,
    n_folds: int = 6,
    pop_size: int = 300,
    n_gens: int = 100,
    l2_lambda: float = 0.5,
    regime_states: Optional[np.ndarray] = None,
    **ga_kwargs,
) -> dict:
    """
    Walk-Forward 検証を実行する。

    Parameters
    ----------
    ticker_data : dict
    sell_rules : list[str]
    thresholds : list[int]
    T_min : int
    t_holdout : int
    bars_per_year : int
    min_trades_oos : int
    crossover_only : bool
    n_folds : int
    pop_size, n_gens : int
    l2_lambda : float
    regime_states : np.ndarray, optional
        HMM レジームラベル配列。指定時、高VIXレジーム ({2,3}) のバーは
        OOS評価でシグナル生成をスキップする。

    Returns
    -------
    dict:
        {
          "folds": [...],
          "avg_oos_sharpe": float,
          "wf_stability": float,
          "overfit_ratio": float,
          "n_folds": int,
        }
    """
    from backtest.strategy.ga_optimizer import optimize_all_sells, detailed_eval

    splits     = wf_splits(T_min, t_holdout, n_folds)
    wf_folds   = []
    oos_sharpes: list[float] = []
    train_sharpes: list[float] = []

    print(f"[WF] {len(splits)} フォールド")
    for fi, (train_end, oos_start, oos_end) in enumerate(splits):
        print(f"\n  ── Fold {fi} | train=[0,{train_end}) OOS=[{oos_start},{oos_end}) ──")

        opt = optimize_all_sells(
            ticker_data, sell_rules, thresholds, train_end, bars_per_year,
            min_trades=max(10, 30 * train_end // t_holdout),
            crossover_only=crossover_only,
            pop_size=pop_size, n_gens=n_gens,
            l2_lambda=l2_lambda, **ga_kwargs,
        )
        best_w   = opt["best_weights"]
        best_sell = opt["best_sell"]
        best_thr  = opt["best_threshold"]
        train_shp = opt["best_fitness"]

        # OOS 評価 (regime_filter適用)
        oos_stats = detailed_eval(
            ticker_data, best_w, best_sell, best_thr,
            oos_start, oos_end, bars_per_year, crossover_only,
            regime_states=regime_states,
        )
        oos_shp = oos_stats["sharpe"] if oos_stats["n_trades"] >= min_trades_oos else 0.0

        wf_folds.append({
            "fold":        fi,
            "train_end":   train_end,
            "oos_start":   oos_start,
            "oos_end":     oos_end,
            "train_sharpe": round(train_shp, 4),
            "oos_sharpe":   round(oos_shp, 4),
            "oos_n_trades": oos_stats["n_trades"],
            "oos_win_rate": oos_stats["win_rate"],
            "best_sell":    best_sell,
            "best_threshold": best_thr,
            "best_weights": best_w.tolist(),
        })
        oos_sharpes.append(oos_shp)
        train_sharpes.append(train_shp)
        print(f"    train_sharpe={train_shp:.3f}  oos_sharpe={oos_shp:.3f}  n_trades={oos_stats['n_trades']}")

    # WF 統計
    oos_arr   = np.array(oos_sharpes)
    oos_valid = oos_arr[np.abs(oos_arr) > 1e-6]
    if len(oos_valid) >= 2:
        wf_stability = float(np.clip(
            1. - np.std(oos_valid) / (abs(np.mean(oos_valid)) + 1e-6),
            -1.0, 1.0,
        ))
    else:
        wf_stability = 0.0

    avg_oos     = float(np.mean(oos_arr)) if len(oos_arr) > 0 else 0.
    avg_train   = float(np.mean(train_sharpes)) if train_sharpes else 0.
    overfit_ratio = round(avg_oos / avg_train, 4) if abs(avg_train) > 1e-6 else 0.

    return {
        "folds":           wf_folds,
        "avg_oos_sharpe":  round(avg_oos, 4),
        "wf_stability":    round(wf_stability, 4),
        "overfit_ratio":   overfit_ratio,
        "n_folds":         len(wf_folds),
    }
