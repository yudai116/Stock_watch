"""
pipeline.py — 3層アーキテクチャのメインオーケストレーター

【実行フロー】
  1. [データ層]    マクロデータ / 価格データ取得 (fetch_macro / fetch_alpaca)
  2. [レジーム層]  HMM 学習 → regime_signals.json 保存
  3. [戦略層]      ticker_data 構築 → Walk-Forward GA 最適化
  4. [リスク層]    ポジションサイジング検証
  5. [出力]        results_*.json 保存

使用方法:
  python -m backtest.pipeline --mode swing
  python -m backtest.pipeline --mode day
  python -m backtest.pipeline --mode all
  python -m backtest.pipeline --skip-fetch --skip-regime  (データ再利用)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from backtest.config import (
    MACRO_DATA_PATH,
    REGIME_SIGNALS_PATH,
    RESULTS_SWING_PATH,
    RESULTS_DAY_PATH,
    ARTIFACTS_DIR,
    TICKERS_SWING,
    TICKERS_DAY,
    BARS_PER_YEAR_SWING,
    BARS_PER_YEAR_DAY,
    WF_N_FOLDS,
    WF_TRAIN_RATIO,
    GA_POP_SWING, GA_GENS_SWING, GA_ELITE_SWING,
    GA_POP_DAY,   GA_GENS_DAY,   GA_ELITE_DAY,
    GA_TOURN_SIZE, GA_MUT_SIGMA, GA_MUT_PROB,
    GA_L2_LAMBDA,
    MIN_TRADES, MIN_TRADES_OOS,
    HMM_N_ITER, HMM_RANDOM_SEED,
    RISK_STOP_IN_REGIME,
    SWING_CROSSOVER_ONLY,
)

HERE = Path(__file__).parent


# ── ステップ 1: データ取得 ──────────────────────────────────────────────────────

def step_fetch_macro() -> bool:
    """マクロデータを yfinance で取得して macro_data.json へ保存"""
    print("\n[Step 1] マクロデータ取得")
    try:
        from backtest.data.fetch_macro import fetch_macro_data
        from backtest.config import TICKERS_MACRO, MACRO_LOOKBACK_YEARS
        fetch_macro_data(
            tickers=TICKERS_MACRO,
            lookback_years=MACRO_LOOKBACK_YEARS,
            output_path=MACRO_DATA_PATH,
        )
        return True
    except Exception as e:
        print(f"  [WARNING] マクロデータ取得失敗: {e}")
        return False


def step_fetch_price(mode: str) -> bool:
    """価格データを Alpaca から取得 (増分更新)"""
    print(f"\n[Step 1] 価格データ取得 ({mode})")
    try:
        from backtest.data.fetch_alpaca import fetch_stock_data
        from backtest.config import (
            SWING_TIMEFRAME, DAY_TIMEFRAME,
            SWING_LOOKBACK_YEARS, DAY_LOOKBACK_YEARS,
            PRICE_DATA_SWING, PRICE_DATA_DAY,
        )
        if mode == "swing":
            tickers     = TICKERS_SWING
            timeframe   = SWING_TIMEFRAME
            years       = SWING_LOOKBACK_YEARS
            output_path = PRICE_DATA_SWING
        else:
            tickers     = TICKERS_DAY
            timeframe   = DAY_TIMEFRAME
            years       = DAY_LOOKBACK_YEARS
            output_path = PRICE_DATA_DAY

        fetch_stock_data(tickers, timeframe=timeframe, lookback_years=years,
                         output_path=output_path)
        return True
    except Exception as e:
        print(f"  [WARNING] 価格データ取得失敗: {e}")
        return False


# ── ステップ 2: HMM レジーム検出 ────────────────────────────────────────────────

def step_regime() -> bool:
    """HMM モデルを学習してレジーム信号を保存"""
    print("\n[Step 2] HMM レジーム検出")
    if not MACRO_DATA_PATH.exists():
        print(f"  [SKIP] {MACRO_DATA_PATH} が存在しません (fetch_macro 先に実行)")
        return False
    try:
        from backtest.regime.hmm_model import run_regime_detection
        from backtest.data.fetch_macro import load_macro_data
        from backtest.config import HMM_MIN_STATES, HMM_MAX_STATES
        macro_data = load_macro_data(MACRO_DATA_PATH)
        run_regime_detection(
            macro_data=macro_data,
            output_path=REGIME_SIGNALS_PATH,
            min_states=HMM_MIN_STATES,
            max_states=HMM_MAX_STATES,
            n_iter=HMM_N_ITER,
            random_seed=HMM_RANDOM_SEED,
        )
        return True
    except Exception as e:
        print(f"  [ERROR] レジーム検出失敗: {e}")
        import traceback; traceback.print_exc()
        return False


# ── ステップ 3: 戦略最適化 ──────────────────────────────────────────────────────

def _build_regime_states(ticker_data: dict, mode: str) -> "np.ndarray | None":
    """ticker_data の dates と regime_signals.json を対応させて regime_states 配列を作成"""
    if not REGIME_SIGNALS_PATH.exists():
        return None

    signals = json.loads(REGIME_SIGNALS_PATH.read_text())
    # {"date": regime_label} の辞書を作成
    regime_map = {s["timestamp"]: s["regime_label"] for s in signals}

    # 最初のティッカーの dates を使用（全ティッカーで共通日付と仮定）
    sample_ticker = next(iter(ticker_data.values()))
    dates = sample_ticker["dates"]
    T = len(dates)

    # デフォルト0（低ボラ）で初期化
    regime_states = np.zeros(T, dtype=np.int32)
    matched = 0
    for i, d in enumerate(dates):
        # DAYモードは日付prefix（最初10文字）でマッチング
        key = d[:10]
        if key in regime_map:
            regime_states[i] = regime_map[key]
            matched += 1

    print(f"  [Regime] {matched}/{T} バーのレジーム信号をマッチング完了")
    return regime_states


def step_strategy(mode: str) -> dict:
    """
    Walk-Forward GA 最適化を実行して結果 dict を返す。
    """
    print(f"\n[Step 3] 戦略最適化 ({mode})")

    from backtest.data.loader import build_strategy_data
    from backtest.strategy.walk_forward import run_walk_forward
    from backtest.strategy.sell_rules import SWING_SELL_RULES, DAY_SELL_RULES

    # --- データ準備 ---
    tickers = TICKERS_SWING if mode == "swing" else TICKERS_DAY
    ticker_data = build_strategy_data(mode=mode, tickers=tickers)

    if not ticker_data:
        print("  [ERROR] ticker_data が空です")
        return {}

    # --- レジーム状態配列の構築 ---
    regime_states = _build_regime_states(ticker_data, mode)

    # T_min (最短ティッカーのバー数)
    T_min = min(len(v["closes"]) for v in ticker_data.values())
    t_holdout = int(T_min * WF_TRAIN_RATIO)
    print(f"  T_min={T_min}, t_holdout={t_holdout}, mode={mode}")

    bars_per_year = BARS_PER_YEAR_SWING if mode == "swing" else BARS_PER_YEAR_DAY
    sell_rules    = SWING_SELL_RULES     if mode == "swing" else DAY_SELL_RULES
    min_trades_oos = MIN_TRADES_OOS

    # BUY 閾値リスト (スコア合計の目安: 8指標×25点=200点、60〜120が現実的)
    thresholds = list(range(60, 121, 5))

    pop_size = GA_POP_SWING if mode == "swing" else GA_POP_DAY
    n_gens   = GA_GENS_SWING if mode == "swing" else GA_GENS_DAY

    n_elite = GA_ELITE_SWING if mode == "swing" else GA_ELITE_DAY

    # --- Walk-Forward 実行 ---
    wf_result = run_walk_forward(
        ticker_data      = ticker_data,
        sell_rules       = sell_rules,
        thresholds       = thresholds,
        T_min            = T_min,
        t_holdout        = t_holdout,
        bars_per_year    = bars_per_year,
        min_trades_oos   = min_trades_oos,
        crossover_only   = SWING_CROSSOVER_ONLY if mode == "swing" else False,
        n_folds          = WF_N_FOLDS,
        pop_size         = pop_size,
        n_gens           = n_gens,
        l2_lambda        = GA_L2_LAMBDA,
        n_elite          = n_elite,
        tourn_size       = GA_TOURN_SIZE,
        mut_sigma        = GA_MUT_SIGMA,
        mut_prob         = GA_MUT_PROB,
        regime_states    = regime_states,
        high_vol_regimes = frozenset(RISK_STOP_IN_REGIME),
    )

    # --- ホールドアウト最終評価 ---
    from backtest.strategy.ga_optimizer import optimize_all_sells, detailed_eval

    print("\n  [最終ホールドアウト評価]")
    opt = optimize_all_sells(
        ticker_data, sell_rules, thresholds,
        t_holdout, bars_per_year,
        min_trades    = MIN_TRADES,
        crossover_only= SWING_CROSSOVER_ONLY if mode == "swing" else False,
        pop_size      = pop_size,
        n_gens        = n_gens,
        l2_lambda     = GA_L2_LAMBDA,
        n_elite       = n_elite,
        tourn_size    = GA_TOURN_SIZE,
        mut_sigma     = GA_MUT_SIGMA,
        mut_prob      = GA_MUT_PROB,
    )

    holdout_stats = detailed_eval(
        ticker_data,
        opt["best_weights"],
        opt["best_sell"],
        opt["best_threshold"],
        t_holdout, T_min,
        bars_per_year,
        crossover_only=SWING_CROSSOVER_ONLY if mode == "swing" else False,
        regime_states=regime_states,
        high_vol_regimes=frozenset(RISK_STOP_IN_REGIME),
    )

    result = {
        "mode":          mode,
        "T_min":         T_min,
        "t_holdout":     t_holdout,
        "n_tickers":     len(ticker_data),
        "wf":            wf_result,
        "best_sell":     opt["best_sell"],
        "best_threshold":opt["best_threshold"],
        "best_weights":  opt["best_weights"].tolist(),
        "train_sharpe":  round(opt["best_fitness"], 4),
        "holdout": {
            "sharpe":         round(holdout_stats["sharpe"], 4),
            "n_trades":       holdout_stats["n_trades"],
            "win_rate":       round(holdout_stats["win_rate"], 4),
            "avg_return":     round(holdout_stats["avg_return"], 6),
            "profit_factor":  round(holdout_stats.get("profit_factor", 0.0), 4),
            "max_dd":         round(holdout_stats.get("max_dd", 0.0), 4),
        },
    }

    # 印字サマリー
    h = result["holdout"]
    w = result["wf"]
    print(f"\n  ── {mode.upper()} 結果サマリー ──")
    print(f"  best_sell={result['best_sell']}  threshold={result['best_threshold']}")
    print(f"  train_sharpe={result['train_sharpe']:.3f}")
    print(f"  holdout: sharpe={h['sharpe']:.3f}  n_trades={h['n_trades']}  "
          f"win_rate={h['win_rate']:.2%}  pf={h['profit_factor']:.2f}")
    print(f"  WF: avg_oos={w['avg_oos_sharpe']:.3f}  "
          f"stability={w['wf_stability']:.3f}  "
          f"overfit_ratio={w['overfit_ratio']:.3f}")

    return result


# ── ステップ 4: リスク検証（サマリー出力）──────────────────────────────────────

def step_risk_summary(ticker_data: dict, mode: str) -> None:
    """ポジションサイジングのサンプル計算を表示"""
    from backtest.risk.position_sizer import PositionSizer, RegimeFilter

    regime_filter = None
    if REGIME_SIGNALS_PATH.exists():
        regime_filter = RegimeFilter(REGIME_SIGNALS_PATH, RISK_STOP_IN_REGIME)

    sizer = PositionSizer(
        target_vol     = 0.15,
        max_position_frac = 0.20,
        regime_filter  = regime_filter,
        method         = "vol_target",
        bars_per_year  = BARS_PER_YEAR_SWING if mode == "swing" else BARS_PER_YEAR_DAY,
    )

    print(f"\n[Step 4] リスク検証 ({mode}) — サンプル5銘柄")
    for ticker, arr in list(ticker_data.items())[:5]:
        rets  = arr["returns"]
        price = float(arr["closes"][-1])
        date  = arr["dates"][-1][:10] if arr["dates"] else "2024-01-01"
        n = sizer.size(ticker, date, rets, capital=1_000_000, price=price)
        value = n * price
        print(f"  {ticker:6s}  price={price:8.2f}  shares={n:5d}  value=${value:,.0f}")


# ── 結果保存 ─────────────────────────────────────────────────────────────────

def save_results(result: dict, mode: str) -> Path:
    path = RESULTS_SWING_PATH if mode == "swing" else RESULTS_DAY_PATH
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[保存] {path}")
    return path


# ── メイン ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stock backtesting pipeline")
    parser.add_argument("--mode",         default="all",
                        choices=["swing", "day", "all"])
    parser.add_argument("--skip-fetch",   action="store_true",
                        help="データ取得をスキップ (既存データを使用)")
    parser.add_argument("--skip-regime",  action="store_true",
                        help="HMM レジーム検出をスキップ")
    parser.add_argument("--skip-strategy",action="store_true",
                        help="GA 最適化をスキップ")
    args = parser.parse_args()

    modes = ["swing", "day"] if args.mode == "all" else [args.mode]
    t0    = time.time()

    # Step 1: データ取得
    if not args.skip_fetch:
        step_fetch_macro()
        for m in modes:
            step_fetch_price(m)

    # Step 2: レジーム検出
    if not args.skip_regime:
        step_regime()

    # Step 3 & 4: 戦略最適化 + リスク検証
    if not args.skip_strategy:
        for m in modes:
            result = step_strategy(m)
            if result:
                save_results(result, m)
                from backtest.data.loader import load_ticker_data
                td = load_ticker_data(mode=m)
                step_risk_summary(td, m)

    elapsed = time.time() - t0
    print(f"\n✓ パイプライン完了 ({elapsed/60:.1f}分)")


if __name__ == "__main__":
    main()
