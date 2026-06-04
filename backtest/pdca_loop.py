#!/usr/bin/env python3
"""
PDCA ループ判断エンジン。
バックテスト結果を読んで次のパラメータを決定し、
pdca_state.json を更新してアクションを JSON で出力する。

出力 JSON:
  {"action": "trigger", "params": {...}, "reason": "...", "run_count": N}
  {"action": "stop",    "reason": "...", "run_count": N}

使い方:
  python3 backtest/pdca_loop.py
  python3 backtest/pdca_loop.py --dry-run   # 状態変更なしで判断だけ表示
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

HERE       = Path(__file__).parent
STATE_F    = HERE / "pdca_state.json"
SWING_F    = HERE / "strategy_results_swing.json"
DAY_F      = HERE / "strategy_results_day.json"
REPORT_F   = HERE / "pdca_log.md"
TRIGGER_F  = HERE / "pdca_trigger.json"

# ── 実用水準目標 ──────────────────────────────────────────────────────────────
TARGETS = {
    "sharpe":        1.5,
    "wf_stability":  0.7,
    "profit_factor": 1.5,
    "win_rate":      0.55,
    "n_trades":      50,
}

# ── 安全上限 ──────────────────────────────────────────────────────────────────
MAX_RUNS           = 30    # 最大実行回数
MAX_L2_LAMBDA      = 5.0   # L2正則化の上限
NO_IMPROVE_LIMIT   = 5     # 連続改善なし → 違うアプローチを試す
MIN_THRESHOLD      = 30    # 閾値の下限


def _load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None

    # キー統一
    sm = d.get("summary", {})
    wf = d.get("walk_forward", {})
    d  = dict(d)
    d.setdefault("wf_stability",  wf.get("wf_stability")  or sm.get("wf_stability") or 0.)
    d.setdefault("overfit_ratio", wf.get("overfit_ratio") or sm.get("overfit_ratio") or 0.)
    d.setdefault("avg_oos_sharpe",wf.get("avg_oos_sharpe") or sm.get("avg_oos_sharpe") or 0.)
    d.setdefault("best_threshold", sm.get("best_threshold"))

    if not d.get("holdout_stats"):
        if wf.get("test_sharpe") is not None:
            d["holdout_stats"] = {
                "sharpe":        wf.get("test_sharpe", 0.),
                "n_trades":      wf.get("test_n_trades", 0),
                "win_rate":      wf.get("test_win_rate", 0.),
                "max_dd":        wf.get("test_max_dd", 0.),
                "profit_factor": wf.get("test_profit_factor", 0.),
            }
        elif sm:
            top = (d.get("top100") or [{}])[0]
            d["holdout_stats"] = {
                "sharpe":        sm.get("best_sharpe", 0.),
                "n_trades":      top.get("n_trades", 0),
                "win_rate":      top.get("win_rate", 0.),
                "max_dd":        top.get("max_dd", 0.),
                "profit_factor": top.get("profit_factor", 0.),
            }

    ps = d.get("portfolio_simulation")
    if ps and not d.get("portfolio_sim"):
        d["portfolio_sim"] = ps

    return d


def _metrics(d: dict) -> dict:
    hs = d.get("holdout_stats", {})
    ps = d.get("portfolio_sim", {}) or {}
    return {
        "sharpe":        hs.get("sharpe", 0.),
        "n_trades":      hs.get("n_trades", 0),
        "win_rate":      hs.get("win_rate", 0.),
        "max_dd":        hs.get("max_dd", 0.),
        "wf_stability":  d.get("wf_stability", 0.),
        "overfit_ratio": d.get("overfit_ratio", 0.),
        "profit_factor": hs.get("profit_factor") or ps.get("profit_factor", 0.),
    }


def _is_practical(m: dict) -> bool:
    return (
        m["sharpe"]        >= TARGETS["sharpe"]        and
        m["wf_stability"]  >= TARGETS["wf_stability"]  and
        m["profit_factor"] >= TARGETS["profit_factor"] and
        m["win_rate"]      >= TARGETS["win_rate"]      and
        m["n_trades"]      >= TARGETS["n_trades"]
    )


def _passed_count(m: dict) -> int:
    return sum([
        m["sharpe"]        >= TARGETS["sharpe"],
        m["wf_stability"]  >= TARGETS["wf_stability"],
        m["profit_factor"] >= TARGETS["profit_factor"],
        m["win_rate"]      >= TARGETS["win_rate"],
        m["n_trades"]      >= TARGETS["n_trades"],
    ])


def decide_next_params(
    state: dict,
    swing_m: dict | None,
    day_m:   dict | None,
    swing_d: dict | None,
    day_d:   dict | None,
) -> tuple[dict, str]:
    """
    現在の結果を見て次の (params, reason) を決定する。
    過学習対策と信号数増加を優先しつつ、段階的に調整する。
    """
    cur = state["current_params"]
    l2  = cur["ga_l2_lambda"]
    off = cur["threshold_offset"]

    issues_swing = []
    issues_day   = []

    if swing_m:
        if swing_m["n_trades"] < TARGETS["n_trades"]:
            issues_swing.append("low_trades")
        if swing_m["wf_stability"] < 0.5:
            issues_swing.append("overfit")
        if swing_m["overfit_ratio"] > 2.0:
            issues_swing.append("overfit")
        if swing_m["sharpe"] < 0.5 and swing_m["n_trades"] >= 20:
            issues_swing.append("low_sharpe")

    if day_m:
        if day_m["n_trades"] < TARGETS["n_trades"]:
            issues_day.append("low_trades")
        if day_m["wf_stability"] < 0.5 or day_m["wf_stability"] < 0:
            issues_day.append("overfit")
        if day_m["overfit_ratio"] > 2.0:
            issues_day.append("overfit")
        if day_m["sharpe"] < 0.5 and day_m["n_trades"] >= 20:
            issues_day.append("low_sharpe")

    all_issues = set(issues_swing + issues_day)

    new_l2  = l2
    new_off = off
    reasons = []

    run = state["run_count"]
    streak = state.get("no_improve_streak", 0)

    # ── 判断ロジック ──────────────────────────────────────────────────────────
    # 優先1: 取引数不足 → 閾値を下げる
    if "low_trades" in all_issues and off > -20:
        new_off = max(off - 5, -20)
        reasons.append(f"取引数不足 → 閾値オフセット {off}→{new_off}")

    # 優先2: 過学習 → L2強化
    if "overfit" in all_issues and l2 < MAX_L2_LAMBDA:
        new_l2 = min(round(l2 * 1.5, 2), MAX_L2_LAMBDA)
        reasons.append(f"過学習 → L2={l2:.2f}→{new_l2:.2f}")

    # 優先3: Sharpe低 + 取引数十分 → 両方調整
    if "low_sharpe" in all_issues and "low_trades" not in all_issues:
        if l2 < MAX_L2_LAMBDA:
            new_l2 = min(round(l2 + 0.3, 2), MAX_L2_LAMBDA)
            reasons.append(f"Sharpe低 → L2={l2:.2f}→{new_l2:.2f}")

    # 改善が止まった場合: より大胆な調整
    if streak >= 3:
        if new_l2 == l2 and new_off == off:
            # リセット: デフォルトから再スタート
            new_l2 = 1.0
            new_off = -10
            reasons.append(f"改善停止({streak}回) → パラメータリセット L2=1.0, offset=-10")

    # 変更なし → 安定している可能性
    if not reasons:
        reasons.append("大きな問題なし → パラメータ維持")

    new_params = {
        "ga_l2_lambda":    new_l2,
        "threshold_offset": new_off,
        "mode": "both",
    }
    return new_params, " / ".join(reasons)


def run(dry_run: bool = False) -> dict:
    # ── 状態ロード ─────────────────────────────────────────────────────────────
    state: dict = json.loads(STATE_F.read_text()) if STATE_F.exists() else {
        "run_count": 0, "stop_requested": False,
        "current_params": {"ga_l2_lambda": 0.5, "threshold_offset": 0, "mode": "both"},
        "history": [], "best": {"swing_sharpe": 0., "day_sharpe": 0.},
        "no_improve_streak": 0,
    }

    # ── 結果ロード ─────────────────────────────────────────────────────────────
    swing_d = _load_result(SWING_F)
    day_d   = _load_result(DAY_F)
    swing_m = _metrics(swing_d) if swing_d else None
    day_m   = _metrics(day_d)   if day_d   else None

    now_ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_num  = state["run_count"]

    # ── 停止チェック ───────────────────────────────────────────────────────────
    # 1. ユーザー要求
    if state.get("stop_requested"):
        result = {"action": "stop", "reason": "ユーザーによる停止要求", "run_count": run_num}
        print(json.dumps(result, ensure_ascii=False))
        return result

    # 2. 最大実行回数
    if run_num >= MAX_RUNS:
        result = {"action": "stop", "reason": f"最大実行回数({MAX_RUNS}回)に達しました", "run_count": run_num}
        print(json.dumps(result, ensure_ascii=False))
        return result

    # 3. 実用水準達成
    swing_ok = swing_m is not None and _is_practical(swing_m)
    day_ok   = day_m   is not None and _is_practical(day_m)
    if swing_ok and day_ok:
        result = {
            "action": "stop",
            "reason": f"スイング・デイトレ両方が実用水準に達しました！",
            "swing_passed": _passed_count(swing_m),
            "day_passed":   _passed_count(day_m),
            "run_count": run_num,
        }
        print(json.dumps(result, ensure_ascii=False))
        return result

    # ── 改善評価 ───────────────────────────────────────────────────────────────
    prev_best_swing = state["best"].get("swing_sharpe", 0.)
    prev_best_day   = state["best"].get("day_sharpe", 0.)
    cur_swing_shp   = swing_m["sharpe"] if swing_m else 0.
    cur_day_shp     = day_m["sharpe"]   if day_m   else 0.

    improved = (cur_swing_shp > prev_best_swing + 0.02 or
                cur_day_shp   > prev_best_day   + 0.02)

    streak = state.get("no_improve_streak", 0)
    new_streak = 0 if improved else streak + 1

    # ── 次パラメータ決定 ───────────────────────────────────────────────────────
    new_params, reason = decide_next_params(state, swing_m, day_m, swing_d, day_d)

    # ── 状態更新 ───────────────────────────────────────────────────────────────
    history_entry = {
        "ts":        now_ts,
        "run":       run_num,
        "params":    state["current_params"],
        "swing":     swing_m,
        "day":       day_m,
        "swing_passed": _passed_count(swing_m) if swing_m else 0,
        "day_passed":   _passed_count(day_m)   if day_m   else 0,
        "decision":  reason,
    }
    state["history"].append(history_entry)
    state["run_count"] = run_num + 1
    state["current_params"] = new_params
    state["best"]["swing_sharpe"] = max(prev_best_swing, cur_swing_shp)
    state["best"]["day_sharpe"]   = max(prev_best_day,   cur_day_shp)
    state["no_improve_streak"] = new_streak

    if not dry_run:
        STATE_F.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        # pdca_trigger.json を書き出す → push で workflow_dispatch の代わりに起動
        TRIGGER_F.write_text(json.dumps({
            "ga_l2_lambda":    new_params["ga_l2_lambda"],
            "threshold_offset": new_params["threshold_offset"],
            "mode":            new_params["mode"],
            "triggered_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_count":       run_num + 1,
            "reason":          reason,
        }, ensure_ascii=False, indent=2))

    # ── 結果出力 ───────────────────────────────────────────────────────────────
    swing_pass_str = f"{_passed_count(swing_m)}/5" if swing_m else "未実行"
    day_pass_str   = f"{_passed_count(day_m)}/5"   if day_m   else "未実行"

    result = {
        "action":           "trigger",
        "params":           new_params,
        "reason":           reason,
        "run_count":        run_num + 1,
        "swing_passed":     swing_pass_str,
        "day_passed":       day_pass_str,
        "no_improve_streak": new_streak,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    run(dry_run=dry)
