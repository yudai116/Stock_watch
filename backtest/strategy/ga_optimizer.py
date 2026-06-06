"""
strategy/ga_optimizer.py — 遺伝的アルゴリズムによる指標重み最適化

【Sharpe 年率化 — v16 修正版】
  旧式: sqrt(bars_per_year / hold_bars) — 常時取引前提で取引数が少ない戦略を過大評価
  新式: sqrt(n_trades / years)          — 実取引数ベース。GA がチェリーピックを追求しない

【フィットネス関数】
  fitness = Sharpe_IS × 0.6 + Sharpe_val × 0.4 − L2_penalty
  IS:  0〜75% の訓練データ内分割
  val: 75〜100% の訓練データ内分割
  L2:  ga_l2_lambda × ||weights||² / n_indicators
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# ── バッチ Sharpe 評価 ────────────────────────────────────────────────────────

def batch_sharpe(
    ticker_data: dict,
    weight_matrix: np.ndarray,
    sell_name: str,
    thresholds: list[int],
    t_start: int,
    t_end: int,
    min_trades: int,
    bars_per_year: int,
    crossover_only: bool = False,
    regime_states: "Optional[np.ndarray]" = None,
    high_vol_regimes: frozenset = frozenset({2, 3}),
) -> np.ndarray:
    """
    重み行列の全個体について Sharpe を一括計算する。

    Parameters
    ----------
    ticker_data : dict
        {ticker: {ind_scores: (n_ind, T), sell_outcomes: {rule: (T,)}, vol_ok: (T,), ...}}
    weight_matrix : np.ndarray (N, n_ind)
        N 個体の指標重み行列
    sell_name : str
        使用する売りルール名
    thresholds : list[int]
        試す買い閾値リスト
    t_start, t_end : int
        評価期間 (スライス [t_start:t_end])
    min_trades : int
        Sharpe 計算に必要な最小取引数
    bars_per_year : int
        年間バー数 (252=日足, 9828=10min足)
    crossover_only : bool
        True の場合、閾値を「下から越えた」初日のみエントリー (swing 用)
    regime_states : np.ndarray, optional
        HMM レジームラベル配列 (長さ T_total)。指定時、high_vol_regimes に含まれる
        バーはシグナル生成をスキップする (高VIX期間の取引回避)。
    high_vol_regimes : frozenset
        スキップ対象のレジーム番号集合。デフォルト {2, 3}。

    Returns
    -------
    np.ndarray (N, len(thresholds))
        各個体・閾値の Sharpe 値。条件未達は NaN。
    """
    N    = len(weight_matrix)
    n_th = len(thresholds)
    years = (t_end - t_start) / bars_per_year

    acc_n   = np.zeros((N, n_th), dtype=np.float64)
    acc_sum = np.zeros((N, n_th), dtype=np.float64)
    acc_sq  = np.zeros((N, n_th), dtype=np.float64)

    wm = weight_matrix.astype(np.float32)

    # レジームフィルターマスク: high_vol_regimes に含まれるバーは取引不可
    if regime_states is not None:
        regime_slice = regime_states[t_start:t_end]
        regime_ok = np.ones(t_end - t_start, dtype=bool)
        for rv in high_vol_regimes:
            regime_ok &= (regime_slice != rv)
    else:
        regime_ok = None

    for td in ticker_data.values():
        ind  = td["ind_scores"][:, t_start:t_end].astype(np.float32)
        sout = td["sell_outcomes"][sell_name][t_start:t_end].astype(np.float32)
        vmask = td["vol_ok"][t_start:t_end]
        if regime_ok is not None:
            vmask = vmask & regime_ok
        valid = ~np.isnan(sout) & vmask

        comp = wm @ ind  # (N, T_slice)

        for ti, thr in enumerate(thresholds):
            if crossover_only:
                above      = comp >= thr
                prev_above = np.roll(above, 1, axis=1)
                prev_above[:, 0] = False
                mask = above & ~prev_above & valid
            else:
                mask = (comp >= thr) & valid

            tr   = np.where(mask, sout, np.float32(0.))
            is_t = mask
            n_   = is_t.sum(1).astype(np.float64)
            s_   = (tr * is_t).sum(1).astype(np.float64)
            sq_  = (tr ** 2 * is_t).sum(1).astype(np.float64)
            acc_n[:,  ti] += n_
            acc_sum[:, ti] += s_
            acc_sq[:,  ti] += sq_

    shp = np.full((N, n_th), np.nan)
    for ti in range(n_th):
        n  = acc_n[:, ti]
        s  = acc_sum[:, ti]
        sq = acc_sq[:, ti]
        ok  = n >= min_trades
        avg = np.where(ok, s / np.where(n > 0, n, 1.), np.nan)
        var = np.where(ok & (n > 1), sq / np.where(n > 0, n, 1.) - avg ** 2, np.nan)
        std = np.sqrt(np.maximum(var, 0.))
        # ── 年率化: 実取引数ベース (v16修正) ──────────────────────────────────
        factor = np.sqrt(np.maximum(n, 1.) / max(years, 0.5))
        shp[:, ti] = np.where(ok & (std > 1e-10), avg / std * factor, np.nan)

    return shp


def detailed_eval(
    ticker_data: dict,
    weights: np.ndarray,
    sell_name: str,
    threshold: int,
    t_start: int,
    t_end: int,
    bars_per_year: int,
    crossover_only: bool = False,
    regime_states: Optional[np.ndarray] = None,
    high_vol_regimes: frozenset = frozenset({2, 3}),
) -> dict:
    """
    単一重みベクトルの詳細評価 (Sharpe, n_trades, win_rate, avg_return, max_dd, profit_factor)

    Parameters
    ----------
    regime_states : np.ndarray, optional
        HMM レジームラベル配列 (長さ T_total)。指定時、high_vol_regimes に含まれる
        バーはシグナル生成をスキップする (高VIX期間の取引回避)。
    high_vol_regimes : frozenset
        スキップ対象のレジーム番号集合。デフォルト {2, 3}。
    """
    years = (t_end - t_start) / bars_per_year
    n = 0; s = 0.; sq = 0.; wins = 0
    gross_win = 0.; gross_loss = 0.
    equity = 1.0; peak = 1.0; max_dd = 0.
    wm = weights.reshape(1, -1).astype(np.float32)

    # レジームフィルターマスク: high_vol_regimes に含まれるバーは取引不可
    if regime_states is not None:
        regime_slice = regime_states[t_start:t_end]
        regime_ok = np.ones(t_end - t_start, dtype=bool)
        for rv in high_vol_regimes:
            regime_ok &= (regime_slice != rv)
    else:
        regime_ok = None

    for td in ticker_data.values():
        ind   = td["ind_scores"][:, t_start:t_end].astype(np.float32)
        sout  = td["sell_outcomes"][sell_name][t_start:t_end].astype(np.float32)
        vmask = td["vol_ok"][t_start:t_end]
        if regime_ok is not None:
            vmask = vmask & regime_ok
        valid = ~np.isnan(sout) & vmask
        comp  = (wm @ ind)[0]

        if crossover_only:
            above      = comp >= threshold
            prev_above = np.roll(above, 1)
            prev_above[0] = False
            mask = above & ~prev_above & valid
        else:
            mask = (comp >= threshold) & valid

        tr = sout[mask]
        if len(tr) == 0:
            continue

        n    += len(tr)
        s    += float(tr.sum())
        sq   += float((tr ** 2).sum())
        wins += int((tr > 0).sum())
        gross_win  += float(tr[tr > 0].sum()) if (tr > 0).any() else 0.
        gross_loss += float(abs(tr[tr < 0].sum())) if (tr < 0).any() else 0.
        for r in tr:
            equity *= (1 + float(r))
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    if n == 0:
        return {"sharpe": 0., "n_trades": 0, "win_rate": 0., "avg_return": 0.,
                "max_dd": 0., "profit_factor": 0.}

    avg = s / n
    var = sq / n - avg ** 2
    std = np.sqrt(max(var, 0.))
    factor = np.sqrt(n / max(years, 0.5))
    sharpe = (avg / std * factor) if std > 1e-10 else 0.
    pf     = (gross_win / gross_loss) if gross_loss > 1e-8 else (9.999 if gross_win > 0 else 0.)

    return {
        "sharpe":        round(float(sharpe), 4),
        "n_trades":      n,
        "win_rate":      round(wins / n, 4),
        "avg_return":    round(avg * 100, 4),  # %表示
        "max_dd":        round(-max_dd * 100, 4),
        "profit_factor": round(pf, 4),
    }


# ── 遺伝的アルゴリズム ────────────────────────────────────────────────────────

def run_ga(
    ticker_data: dict,
    sell_name: str,
    threshold: int,
    t_train_end: int,
    bars_per_year: int,
    thresholds: Optional[list[int]] = None,
    min_trades: int = 30,
    crossover_only: bool = False,
    pop_size: int = 300,
    n_gens: int = 100,
    n_elite: int = 15,
    tourn_size: int = 7,
    mut_sigma: float = 0.10,
    mut_prob: float = 0.25,
    l2_lambda: float = 0.5,
    prior_weights: Optional[np.ndarray] = None,
    random_seed: int = 42,
    regime_states: Optional[np.ndarray] = None,
    high_vol_regimes: frozenset = frozenset({2, 3}),
) -> tuple[np.ndarray, float]:
    """
    遺伝的アルゴリズムで最適指標重みを探索する。

    Parameters
    ----------
    ticker_data : dict
        バックテストデータ
    sell_name : str
        売りルール名
    threshold : int
        買い閾値 (単一値)
    t_train_end : int
        訓練期間の終端インデックス
    bars_per_year : int
        年間バー数
    thresholds : list[int], optional
        GAフィットネス計算に使う閾値リスト（Noneの場合はthreshold単一値）
    min_trades : int
        最小取引数
    crossover_only : bool
        True: クロスオーバーエントリー
    pop_size, n_gens, n_elite : int
        GA ハイパーパラメータ
    tourn_size : int
        トーナメント選択サイズ
    mut_sigma, mut_prob : float
        突然変異の標準偏差と確率
    l2_lambda : float
        L2 正則化強度
    prior_weights : np.ndarray, optional
        初期集団の中心 (DOE や事前知識から)
    random_seed : int
        乱数シード
    regime_states : np.ndarray, optional
        HMM レジームラベル配列。指定時、high_vol_regimes のバーはフィットネス計算で除外。
    high_vol_regimes : frozenset
        スキップ対象のレジーム番号集合。デフォルト {2, 3}。

    Returns
    -------
    best_weights : np.ndarray (n_ind,)
    best_fitness : float
    """
    np.random.seed(random_seed)

    # indicator 数を tier_data から推定
    sample_td   = next(iter(ticker_data.values()))
    n_ind       = sample_td["ind_scores"].shape[0]
    inner_split = int(t_train_end * 0.75)
    val_min_t   = max(3, min_trades // 4)

    if thresholds is None:
        thresholds = [threshold]

    # 初期集団: Dirichlet 分布で各重みを非負に (合計≒4)
    if prior_weights is not None:
        alpha = np.maximum(prior_weights / prior_weights.sum() * n_ind, 0.5)
    else:
        alpha = np.ones(n_ind)
    pop = np.random.dirichlet(alpha, pop_size) * 4.0  # (pop_size, n_ind)

    def fitness(wm: np.ndarray) -> np.ndarray:
        """IS + val 二段階フィットネス - L2 ペナルティ"""
        shp_is  = batch_sharpe(ticker_data, wm, sell_name, thresholds, 0, inner_split,
                                min_trades, bars_per_year, crossover_only,
                                regime_states=regime_states,
                                high_vol_regimes=high_vol_regimes)
        shp_val = batch_sharpe(ticker_data, wm, sell_name, thresholds, inner_split, t_train_end,
                                val_min_t, bars_per_year, crossover_only,
                                regime_states=regime_states,
                                high_vol_regimes=high_vol_regimes)
        fit_is  = np.where(np.isnan(shp_is[:,  0]), -np.inf, shp_is[:,  0])
        fit_val = np.where(np.isnan(shp_val[:, 0]), 0.0,     shp_val[:, 0])
        combined = 0.6 * fit_is + 0.4 * fit_val
        penalty  = l2_lambda * np.sum(wm ** 2, axis=1) / n_ind
        return combined - penalty

    # ── GA ループ ──────────────────────────────────────────────────────────
    PATIENCE   = 50
    MIN_GENS   = 60
    no_improve = 0
    prev_best  = -np.inf

    for gen in range(n_gens):
        fit = fitness(pop)

        elite_idx = np.argsort(fit)[::-1][:n_elite]
        elite     = pop[elite_idx].copy()
        best_fit  = float(fit[elite_idx[0]])

        if gen % 30 == 0:
            print(f"    [GA gen {gen + 1:3d}/{n_gens}] best_fit={best_fit:.3f}")

        # 早期停止
        if best_fit > prev_best * 1.0001:
            no_improve = 0; prev_best = best_fit
        else:
            no_improve += 1
        if gen >= MIN_GENS and no_improve >= PATIENCE:
            print(f"    [GA] 早期停止 gen={gen + 1} (改善停止 {PATIENCE}世代)")
            break

        # トーナメント選択
        parents = []
        for _ in range(pop_size - n_elite):
            cands  = np.random.choice(pop_size, tourn_size, replace=False)
            winner = cands[np.argmax(fit[cands])]
            parents.append(pop[winner])

        # 算術交叉
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            a  = np.random.uniform(0.3, 0.7)
            c1 = a * parents[i] + (1 - a) * parents[i + 1]
            c2 = (1 - a) * parents[i] + a * parents[i + 1]
            offspring.extend([c1, c2])
        if len(offspring) < pop_size - n_elite:
            offspring.append(parents[-1])
        offspring = np.array(offspring[:pop_size - n_elite])

        # Gaussian 突然変異
        mut_mask  = np.random.rand(*offspring.shape) < mut_prob
        offspring += np.where(mut_mask, np.random.normal(0, mut_sigma, offspring.shape), 0.)
        offspring  = np.clip(offspring, 0., 5.)

        pop = np.vstack([elite, offspring])

    # 最終評価
    final_fit = fitness(pop)
    best_idx  = int(np.argmax(final_fit))
    best_w    = pop[best_idx]
    best_f    = float(final_fit[best_idx]) if np.isfinite(final_fit[best_idx]) else 0.

    return best_w, best_f


def optimize_all_sells(
    ticker_data: dict,
    sell_rules: list[str],
    thresholds: list[int],
    t_train_end: int,
    bars_per_year: int,
    min_trades: int = 30,
    crossover_only: bool = False,
    pop_size: int = 300,
    n_gens: int = 100,
    regime_states: Optional[np.ndarray] = None,
    high_vol_regimes: frozenset = frozenset({2, 3}),
    **ga_kwargs,
) -> dict:
    """
    全売りルールで GA を実行し、最良の (sell_name, weights, threshold) を返す。

    Parameters
    ----------
    regime_states : np.ndarray, optional
        HMM レジームラベル配列。指定時、high_vol_regimes のバーは probe / GA 訓練で除外。
    high_vol_regimes : frozenset
        スキップ対象のレジーム番号集合。デフォルト {2, 3}。

    Returns
    -------
    dict: {
      "best_sell": str,
      "best_weights": np.ndarray,
      "best_threshold": int,
      "best_fitness": float,
      "all_results": {sell_name: {"weights": ..., "threshold": ..., "fitness": ...}}
    }
    """
    all_results: dict = {}

    # 閾値の事前選択 (MC probe 300サンプル)
    sample_td = next(iter(ticker_data.values()))
    n_ind     = sample_td["ind_scores"].shape[0]
    np.random.seed(0)
    probe_w   = np.random.dirichlet(np.ones(n_ind), 300) * 4.0

    best_thresh_per_sell: dict[str, int] = {}
    for sell_name in sell_rules:
        shp_mat = batch_sharpe(ticker_data, probe_w, sell_name, thresholds, 0, t_train_end,
                                min_trades, bars_per_year, crossover_only,
                                regime_states=regime_states,
                                high_vol_regimes=high_vol_regimes)
        col_means = np.nanmean(shp_mat, axis=0)
        best_ti   = int(np.nanargmax(col_means)) if not np.all(np.isnan(col_means)) else 0
        best_thresh_per_sell[sell_name] = thresholds[best_ti]

    # 各売りルールで GA 実行
    for i, sell_name in enumerate(sell_rules, 1):
        thresh = best_thresh_per_sell[sell_name]
        print(f"  [{i}/{len(sell_rules)}] {sell_name} (thresh={thresh})")
        best_w, best_f = run_ga(
            ticker_data, sell_name, thresh, t_train_end,
            bars_per_year, thresholds=[thresh],
            min_trades=min_trades, crossover_only=crossover_only,
            pop_size=pop_size, n_gens=n_gens,
            regime_states=regime_states,
            high_vol_regimes=high_vol_regimes,
            **ga_kwargs,
        )
        all_results[sell_name] = {
            "weights":   best_w,
            "threshold": thresh,
            "fitness":   best_f,
        }
        print(f"    → fitness={best_f:.4f}")

    # 最良選択
    best_sell = max(all_results, key=lambda k: all_results[k]["fitness"])
    best      = all_results[best_sell]
    return {
        "best_sell":      best_sell,
        "best_weights":   best["weights"],
        "best_threshold": best["threshold"],
        "best_fitness":   best["fitness"],
        "all_results":    all_results,
    }
