#!/usr/bin/env python3
"""
バックテスト結果をレポートファイルに追記するスクリプト。
Monitor から呼ばれるほか、手動でも実行可能。

使い方:
  python3 backtest/append_report.py [swing|day|both]
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE      = Path(__file__).parent
SWING_F   = HERE / "strategy_results_swing.json"
DAY_F     = HERE / "strategy_results_day.json"
REPORT_F  = HERE / "pdca_log.md"

TARGETS = {
    "sharpe": 1.5, "wf_stability": 0.7,
    "profit_factor": 1.5, "win_rate": 0.55, "n_trades": 50,
}


def _g(ok: bool) -> str:
    return "✓" if ok else "✗"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None

    # ── キー統一 ──
    sm  = d.get("summary", {})
    wf  = d.get("walk_forward", {})
    d   = dict(d)

    d.setdefault("wf_stability",   wf.get("wf_stability")  or sm.get("wf_stability"))
    d.setdefault("overfit_ratio",  wf.get("overfit_ratio") or sm.get("overfit_ratio"))
    d.setdefault("avg_oos_sharpe", wf.get("avg_oos_sharpe") or sm.get("avg_oos_sharpe"))
    d.setdefault("best_sell_rule", sm.get("best_sell_rule"))
    d.setdefault("best_threshold", sm.get("best_threshold"))
    d.setdefault("best_weights",   sm.get("best_weights", {}))

    if not d.get("holdout_stats"):
        if wf.get("test_sharpe") is not None:
            d["holdout_stats"] = {
                "sharpe":   wf.get("test_sharpe", 0.),
                "n_trades": wf.get("test_n_trades", 0),
                "win_rate": wf.get("test_win_rate", 0.),
                "max_dd":   wf.get("test_max_dd", 0.),
            }
        elif sm:
            top = (d.get("top100") or [{}])[0]
            d["holdout_stats"] = {
                "sharpe":   sm.get("best_sharpe", 0.),
                "n_trades": top.get("n_trades", 0),
                "win_rate": top.get("win_rate", 0.),
                "max_dd":   top.get("max_dd", 0.),
            }

    if not d.get("portfolio_sim"):
        ps = d.get("portfolio_simulation")
        if ps:
            d["portfolio_sim"] = ps

    return d


def _prev_entry(mode: str) -> dict | None:
    """レポートから直前の同モードの成績を取り出す（テキスト解析）。"""
    if not REPORT_F.exists():
        return None
    text = REPORT_F.read_text()
    # 最後の該当ブロックを探す
    tag = f"<!-- metrics:{mode}"
    idx = text.rfind(tag)
    if idx == -1:
        return None
    end = text.find("-->", idx)
    if end == -1:
        return None
    try:
        return json.loads(text[idx + len(tag):end].strip())
    except Exception:
        return None


def _format_delta(now: float | None, prev: float | None, fmt: str = ".3f",
                  higher_is_better: bool = True) -> str:
    if now is None or prev is None:
        return ""
    delta = now - prev
    if abs(delta) < 1e-6:
        return " (変化なし)"
    arrow = "↑" if delta > 0 else "↓"
    sign  = "+" if delta > 0 else ""
    good  = (delta > 0) == higher_is_better
    mark  = "✓" if good else "✗"
    return f" {arrow}{sign}{delta:{fmt}} {mark}"


def build_entry(d: dict, mode: str, label: str) -> str:
    now_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gen_ts  = d.get("generated_at", "不明")
    ntick   = d.get("n_tickers", "?")
    ver     = d.get("result_version") or d.get("version") or "?"
    sell    = d.get("best_sell_rule") or "?"
    thresh  = d.get("best_threshold") or "?"
    bw      = d.get("best_weights") or {}

    hs  = d.get("holdout_stats", {})
    shp = hs.get("sharpe", 0.)
    n   = hs.get("n_trades", 0)
    wr  = hs.get("win_rate", 0.)
    dd  = hs.get("max_dd", 0.)

    ps  = d.get("portfolio_sim", {})
    pf  = ps.get("profit_factor", 0.) if ps else 0.
    mcs = ps.get("multi_crit_score") if ps else None
    sp  = ps.get("score_prop", {}) if ps else {}
    eq  = ps.get("equal", {}) if ps else {}

    wf_stab = d.get("wf_stability") or 0.
    ov_rat  = d.get("overfit_ratio") or 0.
    avg_oos = d.get("avg_oos_sharpe") or 0.

    # 実用水準チェック
    checks = {
        "sharpe":        shp >= TARGETS["sharpe"],
        "wf_stability":  wf_stab >= TARGETS["wf_stability"],
        "profit_factor": pf >= TARGETS["profit_factor"],
        "win_rate":      wr >= TARGETS["win_rate"],
        "n_trades":      n  >= TARGETS["n_trades"],
    }
    passed = sum(checks.values())

    # 直前との差分
    prev = _prev_entry(mode)
    prev_shp = prev.get("sharpe") if prev else None
    prev_wf  = prev.get("wf_stability") if prev else None
    prev_pf  = prev.get("profit_factor") if prev else None
    prev_n   = prev.get("n_trades") if prev else None

    d_shp = _format_delta(shp, prev_shp)
    d_wf  = _format_delta(wf_stab, prev_wf)
    d_pf  = _format_delta(pf, prev_pf)
    d_n   = _format_delta(n, prev_n, fmt=".0f")

    # 診断アクション
    actions = []
    if abs(shp) > 10:
        actions.append(f"⚠⚠ OOS Sharpe={shp:.1f} 異常値 → データリーク・先読みバイアスを確認")
    if wf_stab < 0:
        actions.append(f"⚠⚠ WF安定性={wf_stab:.3f} 負値 → フォールドごとに悪化。データ確認必須")
    if n < 30:
        actions.append(f"取引数={n}件 → 買い閾値 {thresh}→{max(int(thresh)-10, 30) if isinstance(thresh, int) else '?'} に下げる")
    if 0 < shp < 0.8:
        actions.append(f"Sharpe={shp:.3f} 低い → 売りルール再選択 + GA再最適化")
    if 0 < shp and wf_stab < 0.5:
        actions.append(f"WF安定性={wf_stab:.3f} → GA_L2_LAMBDA を 0.5→1.5 に強化して再実行")
    if 0 < ov_rat < 99 and ov_rat > 2.0:
        actions.append(f"過学習倍率={ov_rat:.2f}x → GA_L2_LAMBDA={min(ov_rat, 3.0):.1f} を試す")
    if pf > 0 and pf < 1.0:
        actions.append(f"PF={pf:.2f} < 1 → ストップロス設定を見直し")
    if not actions:
        actions.append("現時点で深刻な問題なし。次回定期確認を継続")

    # 上位指標
    top_w = ""
    if bw:
        top3 = sorted(bw.items(), key=lambda x: x[1], reverse=True)[:3]
        top_w = "  ".join(f"{k}={v:.2f}" for k, v in top3)

    lines = [
        f"---",
        f"",
        f"## {now_ts} ｜ {label} (v{ver}, {ntick}社)",
        f"",
        f"> 生成日時: {gen_ts}  |  売りルール: `{sell}`  |  買い閾値: {thresh}",
        f"",
        f"### 成績サマリー",
        f"",
        f"| 指標 | 値 | 目標 | 達成 | 前回比 |",
        f"|---|---|---|---|---|",
        f"| OOS Sharpe         | {shp:.4f} | >{TARGETS['sharpe']} | {_g(checks['sharpe'])} | {d_shp} |",
        f"| 取引数             | {n}件     | >{TARGETS['n_trades']} | {_g(checks['n_trades'])} | {d_n} |",
        f"| 勝率               | {wr:.1%}  | >{TARGETS['win_rate']:.0%} | {_g(checks['win_rate'])} |  |",
        f"| 最大DD             | {dd:.1f}% |  |  |  |",
        f"| WF安定性           | {wf_stab:.4f} | >{TARGETS['wf_stability']} | {_g(checks['wf_stability'])} | {d_wf} |",
        f"| 過学習倍率         | {ov_rat:.3f}x | ≤1.5 | {_g(ov_rat <= 1.5)} |  |",
        f"| 平均OOS Sharpe(WF) | {avg_oos:.4f} |  |  |  |",
        f"| Profit Factor      | {pf:.3f} | >{TARGETS['profit_factor']} | {_g(checks['profit_factor'])} | {d_pf} |",
    ]
    if mcs is not None:
        lines.append(f"| 多基準スコア       | {mcs:.4f} |  |  |  |")

    lines += [
        f"",
        f"**実用水準: {passed}/5 達成**",
    ]

    # ポートフォリオ
    if sp or eq:
        lines += [f"", f"### ポートフォリオシミュレーション", f""]
        if sp:
            lines.append(
                f"- スコア比例法: リターン={sp.get('total_return_pct',0):+.1f}%  "
                f"Sharpe={sp.get('sharpe',0):.3f}  DD={sp.get('max_drawdown_pct',0):.1f}%"
            )
        if eq:
            lines.append(
                f"- 均等法:       リターン={eq.get('total_return_pct',0):+.1f}%  "
                f"Sharpe={eq.get('sharpe',0):.3f}  DD={eq.get('max_drawdown_pct',0):.1f}%"
            )

    # レジーム別
    rs = ps.get("regime_stats", {}) if ps else {}
    if rs:
        lines += [f"", f"### 相場レジーム別成績", f""]
        lines.append("| レジーム | 取引数 | 勝率 | 平均リターン | PF |")
        lines.append("|---|---|---|---|---|")
        for rname, stats in rs.items():
            rn = stats.get("n_trades", 0)
            rw = stats.get("win_rate", 0.)
            ra = stats.get("avg_return", 0.)
            rp = stats.get("profit_factor", 0.)
            lines.append(f"| {rname} | {rn}件 | {rw:.1%} | {ra:+.2f}% | {rp:.2f} |")

    # 指標ウェイト
    if top_w:
        lines += [f"", f"### 指標ウェイト TOP3", f"", f"`{top_w}`"]

    # アクション
    lines += [f"", f"### 次のアクション", f""]
    for i, a in enumerate(actions, 1):
        lines.append(f"{i}. {a}")

    lines += [f""]

    # 差分追跡用の機械可読メタデータ（非表示）
    metrics = {
        "sharpe":        shp,
        "n_trades":      n,
        "win_rate":      wr,
        "wf_stability":  wf_stab,
        "overfit_ratio": ov_rat,
        "profit_factor": pf,
    }
    lines.append(f"<!-- metrics:{mode} {json.dumps(metrics)} -->")
    lines.append(f"")

    return "\n".join(lines) + "\n"


def run(modes: list[str]) -> None:
    entries = []

    if "swing" in modes:
        d = _load(SWING_F)
        if d:
            entries.append(build_entry(d, "swing", "スイング最適化結果"))
        else:
            entries.append(f"---\n\n## {datetime.now():%Y-%m-%d %H:%M:%S} ｜ スイング結果ファイルなし\n\n")

    if "day" in modes:
        d = _load(DAY_F)
        if d:
            entries.append(build_entry(d, "day", "デイトレ最適化結果"))
        else:
            entries.append(f"---\n\n## {datetime.now():%Y-%m-%d %H:%M:%S} ｜ デイトレ結果ファイルなし\n\n")

    if not entries:
        print("レポート対象なし")
        return

    # ヘッダーがなければ作成
    if not REPORT_F.exists():
        REPORT_F.write_text(
            "# PDCA バックテスト実行ログ\n\n"
            "> このファイルはバックテスト完了のたびに自動追記されます。\n\n"
        )

    with REPORT_F.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry)

    print(f"[append_report] {REPORT_F} に追記しました ({len(entries)}モード)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "both":
        run(["swing", "day"])
    elif args[0] in ("swing", "day"):
        run([args[0]])
    else:
        print(f"使い方: python3 backtest/append_report.py [swing|day|both]")
        sys.exit(1)
