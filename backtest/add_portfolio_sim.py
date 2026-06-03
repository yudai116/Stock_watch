#!/usr/bin/env python3
"""
既存の strategy_results_{mode}.json に portfolio_simulation を追加する。
完全な再トレーニング不要 - price_data_intraday.json と既存の最適パラメータを使用。
"""
import json, sys, time
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# strategy_search.py から必要な関数と定数をインポート
from strategy_search import (
    _load_ticker_data,
    collect_trade_records,
    simulate_portfolio,
    IND_NAMES_SWING,
    IND_NAMES_DAY,
    SWING_SELL_HOLD,
    DAY_SELL_HOLD,
)


def run_for_mode(mode: str):
    result_path = HERE / f"strategy_results_{mode}.json"
    if not result_path.exists():
        print(f"ERROR: {result_path} not found"); return

    print(f"\n{'='*60}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*60}")

    result = json.loads(result_path.read_text())
    summary = result["summary"]
    wf = result.get("walk_forward", {})

    best_sell   = summary["best_sell_rule"]
    best_thresh = int(summary["best_threshold"])
    ind_names   = IND_NAMES_SWING if mode == "swing" else IND_NAMES_DAY
    sell_hold   = SWING_SELL_HOLD if mode == "swing" else DAY_SELL_HOLD

    bw_dict = summary["best_weights"]
    # Handle indicator renames (ROC → RVOL) between versions
    RENAME_MAP = {"ROC": "RVOL", "RVOL": "ROC"}
    best_w = np.array(
        [bw_dict.get(nm, bw_dict.get(RENAME_MAP.get(nm, ""), 0.0)) for nm in ind_names],
        dtype=np.float32
    )

    train_bars = wf.get("train_bars", 0)
    test_bars  = wf.get("test_bars", 0)
    T_min      = train_bars + test_bars
    t_holdout  = train_bars
    co         = (mode == "swing")

    print(f"  best_sell={best_sell}  threshold={best_thresh}")
    print(f"  t_holdout={t_holdout}  T_min={T_min}  crossover_only={co}")

    print("  価格データ読み込み中...")
    ticker_data, T_min, t_holdout = _load_ticker_data(mode)
    print(f"  データ: T_min={T_min}  t_holdout={t_holdout}")

    print("  取引記録収集中...")
    t0 = time.time()
    trade_recs = collect_trade_records(
        ticker_data, best_w, best_sell, best_thresh,
        t_holdout, T_min, crossover_only=co
    )
    print(f"  取引数: {len(trade_recs)}件  ({time.time()-t0:.1f}s)")

    print("  ポートフォリオシミュレーション中...")
    port_sim = {
        "n_trades":   len(trade_recs),
        "equal":      simulate_portfolio(trade_recs, best_thresh, "equal"),
        "score_prop": simulate_portfolio(trade_recs, best_thresh, "score_prop"),
        "frac_kelly": simulate_portfolio(trade_recs, best_thresh, "frac_kelly"),
    }

    for method in ["equal", "score_prop", "frac_kelly"]:
        s = port_sim[method]
        print(f"  [{method:10s}] 最終値={s['final_value']:.2f} "
              f"リターン={s['total_return_pct']:+.1f}%  "
              f"最大DD={s['max_drawdown_pct']:.1f}%  "
              f"Sharpe={s['sharpe']:.3f}")

    result["portfolio_simulation"] = port_sim
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"  保存完了: {result_path}")


if __name__ == "__main__":
    modes = sys.argv[1:] if sys.argv[1:] else ["swing", "day"]
    for mode in modes:
        run_for_mode(mode)
    print("\n完了。")
