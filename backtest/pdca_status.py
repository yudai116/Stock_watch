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

WIDTH = 62

def bar(label: str, v: float, lo: float, hi: float, width: int = 20) -> str:
    norm = max(0., min(1., (v - lo) / max(hi - lo, 1e-6)))
    filled = int(norm * width)
    return f"{'█' * filled}{'░' * (width - filled)} {v:.3f}"

def show_result(path: Path, label: str) -> dict | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None

    print(f"\n{'─'*WIDTH}")
    print(f"  {'【' + label + '】':^{WIDTH-4}}")
    print(f"{'─'*WIDTH}")

    gen   = d.get("generated_at", "不明")
    mode  = d.get("mode", "?")
    ntick = d.get("n_tickers", "?")
    print(f"  生成日時 : {gen}   モード={mode}   銘柄={ntick}社")

    bsell  = d.get("best_sell_rule", d.get("top100", [{}])[0].get("sell_rule", "?"))
    bthresh= d.get("best_threshold", d.get("top100", [{}])[0].get("buy_threshold", "?"))
    bw     = d.get("best_weights", {})
    print(f"  売りルール: {bsell}   買い閾値: {bthresh}")
    if bw:
        sorted_w = sorted(bw.items(), key=lambda x: x[1], reverse=True)
        top3 = "  ".join(f"{k}={v:.2f}" for k, v in sorted_w[:3])
        print(f"  指標TOP3 : {top3}")

    # ホールドアウト成績
    hs = d.get("holdout_stats", {})
    if hs:
        shp = hs.get("sharpe", 0.)
        n   = hs.get("n_trades", 0)
        wr  = hs.get("win_rate", 0.)
        dd  = hs.get("max_dd", 0.)
        print(f"\n  ── ホールドアウト (OOS 後ろ20%) ──")
        print(f"  Sharpe   : {bar(shp, -1, 3)}")
        print(f"  取引数   : {n}件   勝率: {wr:.1%}   最大DD: {dd:.1f}%")

    # ポートフォリオシミュレーション
    ps = d.get("portfolio_sim", {})
    if ps:
        pf  = ps.get("profit_factor", 0.)
        mcs = ps.get("multi_crit_score", None)
        sp  = ps.get("score_prop", {})
        eq  = ps.get("equal", {})
        print(f"\n  ── ポートフォリオシミュレーション (ホールドアウト期間) ──")
        print(f"  Profit Factor : {pf:.2f}   多基準スコア: {mcs if mcs else '─'}")
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
                    print(f"  {rname:16s}: データなし")
                    continue
                wr  = stats.get("win_rate", 0.)
                avg = stats.get("avg_return", 0.)
                pf_ = stats.get("profit_factor", 0.)
                print(f"  {rname:16s}: {n}件  勝率={wr:.1%}  平均={avg:+.2f}%  PF={pf_:.2f}")

    # WF安定性
    wf_stab = d.get("wf_stability", None)
    if wf_stab is not None:
        print(f"\n  WF安定性 (1=完璧): {wf_stab:.3f}  "
              f"{'⚠ 過学習の疑い' if wf_stab < 0.5 else '✓ 良好' if wf_stab > 0.7 else '△ 要確認'}")

    return d


def diagnose(swing_d: dict | None, day_d: dict | None) -> None:
    """結果を診断し、次のPDCAアクションを提示する。"""
    print(f"\n{'═'*WIDTH}")
    print(f"  {'【PDCA 次のアクション】':^{WIDTH-4}}")
    print(f"{'═'*WIDTH}")

    actions: list[tuple[int, str]] = []  # (priority, message)

    def get_val(d: dict | None, *keys, default=None):
        if d is None:
            return default
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, None)
            if cur is None:
                return default
        return cur

    for mode, d in [("スイング", swing_d), ("デイトレ", day_d)]:
        if d is None:
            actions.append((10, f"[{mode}] バックテスト未実行 → GitHub Actions で backtest.yml を起動"))
            continue

        # ホールドアウトSharpe
        shp = get_val(d, "holdout_stats", "sharpe", default=0.)
        n   = get_val(d, "holdout_stats", "n_trades", default=0)
        wf  = get_val(d, "wf_stability", default=1.)
        pf  = get_val(d, "portfolio_sim", "profit_factor", default=0.)

        # 相場レジームのmaxと最弱
        rs  = get_val(d, "portfolio_sim", "regime_stats") or {}
        worst_regime = None; worst_pf = 9999.
        for rname, stats in rs.items():
            if stats.get("n_trades", 0) >= 5:
                rp = stats.get("profit_factor", 0.)
                if rp < worst_pf:
                    worst_pf = rp; worst_regime = rname

        # 診断ルール
        if n < 30:
            actions.append((1, f"[{mode}] ホールドアウト取引数が少ない ({n}件) → 買い閾値を下げる or データ期間を延ばす"))
        if shp < 0.3:
            actions.append((2, f"[{mode}] Sharpe={shp:.2f} 低い → 売りルールの再選択、指標ウェイトのGA再最適化"))
        if shp > 0. and wf < 0.5:
            actions.append((3, f"[{mode}] WF安定性={wf:.2f} 低い (過学習) → L2正則化強化 (GA_L2_LAMBDA)"))
        if pf < 1.0:
            actions.append((4, f"[{mode}] Profit Factor={pf:.2f} < 1 → 損失が利益を上回る。ストップロス設定を見直し"))
        if worst_regime and worst_pf < 0.8:
            actions.append((5, f"[{mode}] {worst_regime}でPF={worst_pf:.2f} — この相場でのパフォーマンスが弱い。専用パラメータを検討"))

    if not actions:
        print("  ✓ 現時点で深刻な問題なし — 次回バックテストで定期確認を継続")
    else:
        actions.sort(key=lambda x: x[0])
        for i, (_, msg) in enumerate(actions[:5], 1):
            print(f"  {i}. {msg}")

    print(f"\n  バックテスト実行: GitHub Actions → backtest.yml → [Run workflow]")
    print(f"  または手元実行:    python3 backtest/strategy_search.py --mode swing --phase assemble")
    print(f"{'─'*WIDTH}\n")


def main() -> None:
    print(f"\n{'═'*WIDTH}")
    print(f"  {'PDCA ステータス':^{WIDTH-4}}")
    print(f"{'═'*WIDTH}")

    swing_d = show_result(SWING, "スイング最適化結果")
    day_d   = show_result(DAY,   "デイトレ最適化結果")

    if swing_d is None and day_d is None:
        print("\n  バックテスト結果ファイルが見つかりません。")
        print("  GitHub Actions で backtest.yml を実行してください。\n")
        return

    diagnose(swing_d, day_d)


if __name__ == "__main__":
    main()
