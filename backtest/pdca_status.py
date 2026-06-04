#!/usr/bin/env python3
"""
PDCAサイクル現状表示スクリプト。
Claude Code の SessionStart フックで自動実行され、
直前のバックテスト結果を要約して次の最適化アクションを提示する。

手動実行:
  python3 backtest/pdca_status.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE   = Path(__file__).parent
SWING  = HERE / "strategy_results_swing.json"
DAY    = HERE / "strategy_results_day.json"

WIDTH = 66

# 実用水準の目標値
TARGETS = {
    "sharpe":       1.5,   # OOS Sharpe
    "wf_stability": 0.7,   # WF安定性
    "profit_factor": 1.5,  # Profit Factor
    "win_rate":     0.55,  # 勝率
    "n_trades":     50,    # 取引数
}


def bar(v: float, lo: float, hi: float, width: int = 20) -> str:
    norm = max(0., min(1., (v - lo) / max(hi - lo, 1e-6)))
    filled = int(norm * width)
    return f"{'█' * filled}{'░' * (width - filled)} {v:.3f}"


def grade(v: float, good: float, warn: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if v >= good:   return "✓"
        if v >= warn:   return "△"
        return "✗"
    else:
        if v <= good:   return "✓"
        if v <= warn:   return "△"
        return "✗"


def normalize(d: dict) -> dict:
    """旧/新フォーマットを統一して holdout_stats・portfolio_sim・wf_stability を揃える。

    フォーマット判別:
      v8+  : result_version >= 8 → walk_forward / portfolio_simulation キーを使用
      旧   : summary キーのみ → summary + top100 から補完
    """
    d = dict(d)
    sm  = d.get("summary", {})
    wf  = d.get("walk_forward", {})
    ver = d.get("result_version") or d.get("version") or 0

    # ── wf_stability / overfit_ratio / avg_oos_sharpe の統一 ──
    d.setdefault("wf_stability",  wf.get("wf_stability")  or sm.get("wf_stability"))
    d.setdefault("overfit_ratio", wf.get("overfit_ratio") or sm.get("overfit_ratio"))
    d.setdefault("avg_oos_sharpe", wf.get("avg_oos_sharpe") or sm.get("avg_oos_sharpe"))

    # ── best_sell_rule / threshold / weights の統一 ──
    d.setdefault("best_sell_rule", sm.get("best_sell_rule"))
    d.setdefault("best_threshold", sm.get("best_threshold"))
    d.setdefault("best_weights",   sm.get("best_weights", {}))

    # ── holdout_stats の統一 ──
    if not d.get("holdout_stats"):
        if wf.get("test_sharpe") is not None:
            # v8+ フォーマット: walk_forward に正式OOS統計あり
            d["holdout_stats"] = {
                "sharpe":   wf.get("test_sharpe", 0.),
                "n_trades": wf.get("test_n_trades", 0),
                "win_rate": wf.get("test_win_rate", 0.),
                "max_dd":   wf.get("test_max_dd", 0.),
            }
        elif sm:
            # 旧フォーマット: summary + top100 から補完
            top = (d.get("top100") or [{}])[0]
            d["holdout_stats"] = {
                "sharpe":   sm.get("best_sharpe", 0.),
                "n_trades": top.get("n_trades", 0),
                "win_rate": top.get("win_rate", 0.),
                "max_dd":   top.get("max_dd", 0.),
            }
            d["_legacy_format"] = True

    # ── portfolio_sim の統一 ──
    if not d.get("portfolio_sim"):
        ps = d.get("portfolio_simulation")  # v8+ キー名
        if ps:
            d["portfolio_sim"] = ps

    return d


def show_result(path: Path, label: str) -> dict | None:
    if not path.exists():
        return None
    try:
        d = normalize(json.loads(path.read_text()))
    except Exception:
        return None

    print(f"\n{'─'*WIDTH}")
    print(f"  {'【' + label + '】':^{WIDTH-4}}")
    print(f"{'─'*WIDTH}")

    gen   = d.get("generated_at", "不明")
    mode  = d.get("mode", "?")
    ntick = d.get("n_tickers", "?")
    ver   = d.get("result_version", "旧")
    legacy = " [旧フォーマット]" if d.get("_legacy_format") else ""
    print(f"  生成日時 : {gen}   モード={mode}   銘柄={ntick}社   v{ver}{legacy}")

    bsell  = d.get("best_sell_rule") or "?"
    bthresh= d.get("best_threshold") or "?"
    bw     = d.get("best_weights") or {}
    print(f"  売りルール: {bsell}   買い閾値: {bthresh}")
    if bw:
        sorted_w = sorted(bw.items(), key=lambda x: x[1], reverse=True)
        top3 = "  ".join(f"{k}={v:.2f}" for k, v in sorted_w[:3])
        print(f"  指標TOP3 : {top3}")

    # WF安定性・過学習
    wf_stab = d.get("wf_stability")
    ov_rat  = d.get("overfit_ratio")
    avg_oos = d.get("avg_oos_sharpe")
    if wf_stab is not None:
        wf_mark = grade(wf_stab, 0.7, 0.5)
        print(f"\n  ── ウォークフォワード安定性 ──")
        print(f"  WF安定性    : {bar(wf_stab, -1, 1, 16)} {wf_mark}")
        if avg_oos is not None:
            print(f"  平均OOS Sharpe: {avg_oos:.4f}")
        if ov_rat is not None:
            ov_mark = grade(ov_rat, 1.3, 2.0, higher_is_better=False)
            print(f"  過学習倍率  : {ov_rat:.3f}x  {ov_mark}  (訓練/テスト Sharpe比, 1.0=理想)")
            if ov_rat > 3.0:
                print(f"  ⚠  過学習が深刻。L2正則化(GA_L2_LAMBDA)の大幅強化を推奨")
            elif ov_rat > 2.0:
                print(f"  △  過学習の疑い。GA_L2_LAMBDA を 0.5→1.0 に増加して再実行を推奨")

    # ホールドアウト成績
    hs = d.get("holdout_stats", {})
    if hs:
        shp = hs.get("sharpe", 0.)
        n   = hs.get("n_trades", 0)
        wr  = hs.get("win_rate", 0.)
        dd  = hs.get("max_dd", 0.)
        shp_mark = grade(shp, TARGETS["sharpe"], 0.5)
        n_mark   = grade(n, TARGETS["n_trades"], 20)
        wr_mark  = grade(wr, TARGETS["win_rate"], 0.45)
        print(f"\n  ── ホールドアウト成績 (OOS) ──")
        print(f"  Sharpe    : {bar(shp, -1, 3)} {shp_mark}  (目標 >{TARGETS['sharpe']})")
        print(f"  取引数    : {n}件 {n_mark}   勝率: {wr:.1%} {wr_mark}   最大DD: {dd:.1f}%")

        # 異常値警告
        if abs(shp) > 10:
            print(f"  ⚠⚠ Sharpe={shp:.1f} は非現実的。データリーク・先読みバイアスを確認！")

    # ポートフォリオシミュレーション
    ps = d.get("portfolio_sim", {})
    if ps:
        pf  = ps.get("profit_factor", 0.)
        mcs = ps.get("multi_crit_score")
        sp  = ps.get("score_prop", {})
        eq  = ps.get("equal", {})
        pf_mark = grade(pf, TARGETS["profit_factor"], 1.0)
        print(f"\n  ── ポートフォリオシミュレーション ──")
        print(f"  Profit Factor : {pf:.2f} {pf_mark}  (目標 >{TARGETS['profit_factor']})")
        if mcs is not None:
            print(f"  多基準スコア  : {mcs:.4f}  (Sharpe×0.4 + Calmar×0.35 + PF×0.25)")
        if sp:
            print(f"  スコア比例法  : リターン={sp.get('total_return_pct',0):+.1f}%  "
                  f"Sharpe={sp.get('sharpe',0):.3f}  DD={sp.get('max_drawdown_pct',0):.1f}%")
        if eq:
            print(f"  均等法        : リターン={eq.get('total_return_pct',0):+.1f}%  "
                  f"Sharpe={eq.get('sharpe',0):.3f}  DD={eq.get('max_drawdown_pct',0):.1f}%")

        # レジーム別分析
        rs = ps.get("regime_stats", {})
        if rs:
            print(f"\n  ── 相場レジーム別成績 ──")
            for rname, stats in rs.items():
                n = stats.get("n_trades", 0)
                if n == 0:
                    print(f"  {rname:18s}: データなし")
                    continue
                wr_r  = stats.get("win_rate", 0.)
                avg   = stats.get("avg_return", 0.)
                pf_r  = stats.get("profit_factor", 0.)
                mark  = grade(pf_r, 1.3, 1.0)
                print(f"  {rname:18s}: {n}件  勝率={wr_r:.1%}  平均={avg:+.2f}%  PF={pf_r:.2f} {mark}")

    # 実用水準チェック
    _check_target(d)

    return d


def _check_target(d: dict) -> None:
    """実用水準に達しているかチェック。"""
    hs  = d.get("holdout_stats", {})
    ps  = d.get("portfolio_sim", {})
    shp = hs.get("sharpe", 0.)
    n   = hs.get("n_trades", 0)
    wr  = hs.get("win_rate", 0.)
    pf  = ps.get("profit_factor", 0.)
    wf  = d.get("wf_stability") or 0.

    checks = [
        ("Sharpe > 1.5",        shp >= 1.5),
        ("WF安定性 > 0.7",     wf  >= 0.7),
        ("PF > 1.5",           pf  >= 1.5),
        ("勝率 > 55%",         wr  >= 0.55),
        ("取引数 > 50",        n   >= 50),
    ]
    passed = sum(1 for _, ok in checks if ok)
    total  = len(checks)

    print(f"\n  ── 実用水準チェック ({passed}/{total} 達成) ──")
    for label, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label}")
    if passed == total:
        print(f"  🎉 実用水準に達しています！")
    else:
        print(f"  → あと {total - passed} 項目の改善が必要")


def diagnose(swing_d: dict | None, day_d: dict | None) -> None:
    print(f"\n{'═'*WIDTH}")
    print(f"  {'【PDCA 次のアクション】':^{WIDTH-4}}")
    print(f"{'═'*WIDTH}")

    actions: list[tuple[int, str]] = []

    def get_val(d, *keys, default=None):
        if d is None:
            return default
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
            if cur is None:
                return default
        return cur

    for mode, d in [("スイング", swing_d), ("デイトレ", day_d)]:
        if d is None:
            actions.append((10, f"[{mode}] バックテスト未実行 → GitHub Actions backtest.yml を起動"))
            continue

        shp = get_val(d, "holdout_stats", "sharpe") or 0.
        n   = get_val(d, "holdout_stats", "n_trades") or 0
        wr  = get_val(d, "holdout_stats", "win_rate") or 0.
        wf  = d.get("wf_stability") or 1.
        pf  = get_val(d, "portfolio_sim", "profit_factor") or 0.
        ov  = d.get("overfit_ratio") or 1.

        rs = get_val(d, "portfolio_sim", "regime_stats") or {}
        worst_regime = None; worst_pf = 9999.
        for rname, stats in rs.items():
            if stats.get("n_trades", 0) >= 5:
                rp = stats.get("profit_factor", 0.)
                if rp < worst_pf:
                    worst_pf = rp; worst_regime = rname

        # 異常値チェック（最優先）
        if abs(shp) > 10:
            actions.append((0, f"[{mode}] ⚠⚠ Sharpe={shp:.1f} は異常値 → データリーク調査が最優先"))
        if wf < 0:
            actions.append((0, f"[{mode}] ⚠⚠ WF安定性={wf:.2f} が負 → フォールドごとに悪化。データ・コード確認必須"))

        if n < 30:
            thresh = d.get("best_threshold") or 70
            actions.append((1, f"[{mode}] 取引数={n}件 → 買い閾値 {thresh}→{max(thresh-10,30)} に下げて再実行"))
        if 0 < shp < 0.8:
            actions.append((2, f"[{mode}] Sharpe={shp:.2f} 低い → 売りルール再選択 + GA再最適化"))
        if 0 < shp and wf < 0.5:
            actions.append((3, f"[{mode}] WF安定性={wf:.2f} → GA_L2_LAMBDA を 0.5→1.5 に強化して再実行"))
        if 0 < ov < 99 and ov > 2.0:
            actions.append((3, f"[{mode}] 過学習倍率={ov:.2f}x → GA_L2_LAMBDA={min(ov, 3.0):.1f} を試す"))
        if pf > 0 and pf < 1.0:
            actions.append((4, f"[{mode}] PF={pf:.2f} < 1 → ストップロス設定を見直し"))
        if wr > 0 and wr < 0.45:
            actions.append((5, f"[{mode}] 勝率={wr:.1%} 低い → 買い閾値を上げて精度優先へ"))
        if worst_regime and worst_pf < 0.8:
            actions.append((6, f"[{mode}] {worst_regime}でPF={worst_pf:.2f} → この相場専用パラメータを検討"))

        # 旧フォーマット通知
        if d.get("_legacy_format"):
            actions.append((9, f"[{mode}] 旧フォーマット → v8フォーマットで再実行するとレジーム分析が有効化"))

    if not actions:
        print("  ✓ 現時点で深刻な問題なし — 次回バックテストで定期確認を継続")
    else:
        actions.sort(key=lambda x: x[0])
        for i, (_, msg) in enumerate(actions[:6], 1):
            print(f"  {i}. {msg}")

    print(f"\n  バックテスト実行: GitHub Actions → backtest.yml → [Run workflow]")
    print(f"  手元実行(swing):  python3 backtest/strategy_search.py --mode swing --phase all")
    print(f"  手元実行(day):    python3 backtest/strategy_search.py --mode day   --phase all")
    print(f"{'─'*WIDTH}\n")


def show_loop_state() -> None:
    """pdca_state.json / pdca_trigger.json の現状を表示する。"""
    state_f   = HERE / "pdca_state.json"
    trigger_f = HERE / "pdca_trigger.json"

    print(f"\n{'═'*WIDTH}")
    print(f"  {'【PDCAループ ステータス】':^{WIDTH-4}}")
    print(f"{'═'*WIDTH}")

    if state_f.exists():
        try:
            s = json.loads(state_f.read_text())
            run_count  = s.get("run_count", 0)
            stop_req   = s.get("stop_requested", False)
            cur_params = s.get("current_params", {})
            streak     = s.get("no_improve_streak", 0)
            history    = s.get("history", [])
            last       = history[-1] if history else {}

            status_str = "停止要求あり" if stop_req else f"実行中 (run #{run_count})"
            print(f"  ループ状態  : {status_str}")
            print(f"  現在パラメータ: L2={cur_params.get('ga_l2_lambda','?')}  "
                  f"offset={cur_params.get('threshold_offset','?')}  "
                  f"mode={cur_params.get('mode','?')}")
            print(f"  改善なし連続: {streak}回")

            if last:
                sw_p = last.get("swing_passed", "?")
                dy_p = last.get("day_passed", "?")
                dec  = last.get("decision", "")
                ts   = last.get("ts", "")
                print(f"  前回結果    : swing={sw_p}/5  day={dy_p}/5  ({ts})")
                if dec:
                    print(f"  前回決定    : {dec}")
        except Exception as e:
            print(f"  [ERROR] pdca_state.json 読み込み失敗: {e}")
    else:
        print("  pdca_state.json が見つかりません (ループ未開始)")

    if trigger_f.exists():
        try:
            t = json.loads(trigger_f.read_text())
            print(f"\n  最終トリガー: run#{t.get('run_count','?')}  "
                  f"{t.get('triggered_at','?')}")
            print(f"  トリガー理由: {t.get('reason','?')}")
        except Exception:
            pass

    print(f"\n  GitHub Actions: https://github.com/yudai116/Stock_watch/actions")
    print(f"  バックテスト  : https://github.com/yudai116/Stock_watch/actions/workflows/backtest.yml")
    print(f"  PDCA改善ログ  : https://github.com/yudai116/Stock_watch/actions/workflows/pdca_improve.yml")
    print(f"{'─'*WIDTH}")


def main() -> None:
    print(f"\n{'═'*WIDTH}")
    print(f"  {'PDCA ステータス':^{WIDTH-4}}")
    print(f"{'═'*WIDTH}")

    show_loop_state()

    swing_d = show_result(SWING, "スイング最適化結果")
    day_d   = show_result(DAY,   "デイトレ最適化結果")

    if swing_d is None and day_d is None:
        print("\n  バックテスト結果ファイルが見つかりません。")
        print("  GitHub Actions で backtest.yml を実行してください。\n")
        return

    diagnose(swing_d, day_d)


if __name__ == "__main__":
    main()
