# 検証チェッカー エージェント 仕様書

## あなたの役割

`backtest/strategy_search.py` に加えられた変更が **アルゴリズム的に正しく機能するか** を確認する。
コードは変更しない。問題を発見して報告するだけ。

---

## 実行ステップ

### Step 1: コードを読む

以下を全て読む：
- `backtest/strategy_search.py`（変更対象）
- `backtest/VERIFICATION_SPEC.md`（仕様・既知バグ一覧）
- `backtest/pdca_state.json`（最新の `history[].decision` = 今回の改善理由）
- 直近の git diff（`git diff HEAD~1 backtest/strategy_search.py`）

---

### Step 2: 動的検証スクリプトを生成・実行

以下の内容の Python スクリプトを `/tmp/verify_strategy.py` として書き出し、実行する。
スクリプトは `backtest/strategy_search.py` を直接 import して実際の関数を動かす。

```python
import sys, json, traceback
import numpy as np
sys.path.insert(0, "/home/user/Stock_watch")

errors   = []
warnings = []

# ── A. スコア関数の出力範囲（動的実行） ─────────────────────────────────────
try:
    from backtest.strategy_search import (
        score_rsi_bull, score_bb_bull, score_relvol,
        score_momentum_3b, score_vwap_bull, score_orb,
    )

    rng = np.random.default_rng(42)

    checks = [
        ("score_rsi_bull",    score_rsi_bull,    (np.linspace(0, 100, 2000),)),
        ("score_bb_bull",     score_bb_bull,     (np.linspace(-0.1, 1.1, 2000),)),
        ("score_relvol",      score_relvol,      (rng.uniform(0, 6, 2000),)),
        ("score_momentum_3b", score_momentum_3b, (rng.uniform(50, 200, 2000),)),
        ("score_vwap_bull",   score_vwap_bull,   (rng.uniform(-0.1, 0.1, 2000),)),
    ]

    for name, fn, args in checks:
        out = fn(*args)
        out_clean = out[~np.isnan(out)]
        hi = float(np.max(out_clean)) if len(out_clean) else 0.
        lo = float(np.min(out_clean)) if len(out_clean) else 0.
        nan_count = int(np.isnan(out).sum())
        print(f"  {name}: min={lo:.2f}  max={hi:.2f}  nan={nan_count}")
        if hi > 25.05:
            errors.append(f"[A-CRITICAL] {name}: max={hi:.2f} が 25 を超えている → GA重み最適化が歪む")
        if lo < -0.05:
            errors.append(f"[A-CRITICAL] {name}: min={lo:.2f} が負 → スコアが負になりシグナルが逆転")
        if nan_count > len(out) * 0.5:
            warnings.append(f"[A-WARN] {name}: NaN率={nan_count/len(out):.1%} が高い")

    # NaN入力に対するロバスト性
    nan_in = np.full(100, np.nan)
    for name, fn, _ in checks:
        try:
            out = fn(nan_in)
            if np.isnan(out).any():
                warnings.append(f"[A-WARN] {name}: NaN入力に対してNaNを返す（0に変換されるべき）")
        except Exception as e:
            errors.append(f"[A-CRITICAL] {name}: NaN入力でクラッシュ: {e}")

except Exception as e:
    errors.append(f"[A-CRITICAL] score関数import失敗: {traceback.format_exc()}")


# ── B. 閾値パラメータ（破綻チェック） ────────────────────────────────────────
try:
    from backtest.strategy_search import (
        BUY_THRESHOLDS, SWING_BUY_THRESHOLDS, _TOFF, MIN_TRADES_OOS
    )
    try:
        from backtest.strategy_search import _TOFF_CLAMPED
    except ImportError:
        _TOFF_CLAMPED = None

    print(f"\n  BUY_THRESHOLDS       = {BUY_THRESHOLDS}")
    print(f"  SWING_BUY_THRESHOLDS = {SWING_BUY_THRESHOLDS}")
    print(f"  _TOFF={_TOFF}  _TOFF_CLAMPED={_TOFF_CLAMPED}  MIN_TRADES_OOS={MIN_TRADES_OOS}")

    if len(set(BUY_THRESHOLDS)) == 1:
        errors.append(f"[B-CRITICAL] BUY_THRESHOLDS 全て同値に縮退: {BUY_THRESHOLDS} → GAが閾値=定数しか探索できない")
    if len(set(SWING_BUY_THRESHOLDS)) == 1:
        errors.append(f"[B-CRITICAL] SWING_BUY_THRESHOLDS 全て同値に縮退: {SWING_BUY_THRESHOLDS}")
    if _TOFF_CLAMPED is None:
        warnings.append("[B-WARN] _TOFF_CLAMPED が存在しない → _TOFF が大きく負の場合に閾値崩壊リスク")
    if MIN_TRADES_OOS > 4:
        errors.append(f"[B-HIGH] MIN_TRADES_OOS={MIN_TRADES_OOS} > 4 → 4取引以下のホールドアウトがSharpe=0になる")

except Exception as e:
    errors.append(f"[B-CRITICAL] 閾値パラメータ読み込み失敗: {e}")


# ── C. detailed_eval_single の境界ケース（ゼロ除算） ─────────────────────────
try:
    from backtest.strategy_search import detailed_eval_single
    import inspect
    src = inspect.getsource(detailed_eval_single)

    if "profit_factor" not in src:
        errors.append("[C-HIGH] detailed_eval_single に profit_factor がない → pdca_state.json の PF が常に 0.0")
    if 'n == 0' not in src and 'n==0' not in src:
        errors.append("[C-HIGH] detailed_eval_single: n=0 ガードが見当たらない → ゼロ除算リスク")
    if 'gross_loss' in src and '1e-8' not in src and '1e-9' not in src:
        warnings.append("[C-WARN] gross_loss のゼロ除算ガードが弱い可能性")

except Exception as e:
    errors.append(f"[C-HIGH] detailed_eval_single 読み込み失敗: {e}")


# ── D. GAチェックポイントのキー検証 ──────────────────────────────────────────
try:
    from backtest.strategy_search import run_ga_all_sells
    import inspect
    src = inspect.getsource(run_ga_all_sells)

    if "ga_l2_lambda" not in src:
        errors.append("[D-HIGH] GAチェックポイントに ga_l2_lambda がない → L2変更時に旧重みを再利用する")
    if "threshold_offset" not in src:
        errors.append("[D-HIGH] GAチェックポイントに threshold_offset がない → TOFF変更時に旧重みを再利用する")

except Exception as e:
    warnings.append(f"[D-WARN] run_ga_all_sells 読み込みスキップ: {e}")


# ── E. WF安定性の数値安定性（コードパターン確認） ────────────────────────────
try:
    import ast, pathlib
    src_full = pathlib.Path("/home/user/Stock_watch/backtest/strategy_search.py").read_text()

    if "oos_valid" not in src_full and "oos_arr[oos_arr" not in src_full:
        errors.append("[E-HIGH] WF安定性計算: ゼロフォールドの除外コードが見当たらない → 安定性が歪む")

    if "wf_stability" in src_full:
        # 分母保護: 1e-6 or eps などが含まれているか
        import re
        wf_block = re.search(r'wf_stability.*?\n(?:.*?\n){0,10}', src_full)
        if wf_block and '1e-6' not in wf_block.group() and 'eps' not in wf_block.group():
            warnings.append("[E-WARN] WF安定性の分母に epsilon ガードが見当たらない → mean≈0 で±∞になる可能性")

except Exception as e:
    warnings.append(f"[E-WARN] WF安定性チェックスキップ: {e}")


# ── 結果出力 ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ERRORS:", len(errors))
for e in errors:   print(" ✗", e)
print("WARNINGS:", len(warnings))
for w in warnings: print(" ⚠", w)
print("="*60)
print("RESULT:", "FAIL" if errors else ("WARN" if warnings else "PASS"))
```

---

### Step 3: 静的読み込みによる補完チェック

動的実行で検出できない以下を、コードを読んで確認する：

1. **データリーケージ**: `t_start` / `t_end` の境界が `t_holdout` を超えていないか
2. **配列長不一致**: `ind_scores[:, t_start:t_end]` と `sell_outcomes[t_start:t_end]` のスライスが一致しているか
3. **改善理由との整合**: `pdca_state.json` の最新 `decision` に記載された改善内容がコードに存在するか
4. **VERIFICATION_SPEC.md §既知のバグ** の全項目が修正済みか

---

### Step 4: 報告

以下のフォーマットで報告する：

```
RESULT: PASS / WARN / FAIL

[動的検証]
✓ score_rsi_bull: min=0.0, max=25.0
✓ BUY_THRESHOLDS: [5,8,10,12,15]（多様性あり）
✗ [B-HIGH] MIN_TRADES_OOS=5 → 4取引ホールドアウトがSharpe=0になる

[静的検証]
✓ profit_factor 伝播: detailed_eval_single に profit_factor キーあり
✓ GAチェックポイント: ga_l2_lambda, threshold_offset 両方あり
✗ [E-HIGH] oos_valid フィルタが見当たらない

[改善提案との整合]
✓ pdca_state.decision「閾値崩壊修正」→ _TOFF_CLAMPED あり
✗ pdca_state.decision「profit_factor追加」→ コードに未反映

修正が必要な箇所:
- backtest/strategy_search.py: detailed_eval_single の return に profit_factor を追加
- backtest/strategy_search.py: WF安定性計算に oos_valid フィルタを追加
```

---

## 重要な原則

- **実行して確認する**。静的なコードパターン検索だけでは不十分。
- FAILが出ても、コードを修正するのはあなたの役割ではない。問題を正確に特定して報告する。
- WARNは「改善した方が良い」、ERRORは「このまま走らせると破綻する」として区別する。
