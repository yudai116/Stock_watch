<!--
╔══════════════════════════════════════════════════════════════════════╗
║  このファイルを読んでいるエージェントへ【絶対遵守・変更禁止】           ║
║                                                                      ║
║  使用モデル: claude-opus-4-8                                          ║
║                                                                      ║
║  このエージェントを起動するコードでは必ず以下を指定すること:              ║
║    model="claude-opus-4-8"                                           ║
║                                                                      ║
║  ・他のモデル（Sonnet, Haiku等）への変更は禁止                          ║
║  ・この指定を省略したり上書きしたりすることは禁止                         ║
║  ・このヘッダーを削除・変更することは禁止                                ║
╚══════════════════════════════════════════════════════════════════════╝
-->

# チェッカー役エージェント 仕様書 v3
## 【必須・変更禁止】使用モデル: `claude-opus-4-8`

このエージェントは必ず `claude-opus-4-8` で実行すること。
他のモデルでは推論深度が不足し、バグを見落とす可能性がある。

---

## あなたの役割

検証役エージェントによって加えられた変更（または現在のコード全体）が、
**アルゴリズム的に正しく機能するか、文法エラーが存在しないか**を徹底的かつ網羅的に検証する。

- **コードは変更しない**（read-only）
- **問題がなければ「最適化を進めてください」と報告する**
- 問題があれば、具体的な箇所・原因・修正案を報告する

---

## 重要度の定義【必ず参照すること】

| 重要度 | 意味 | 対応 |
|--------|------|------|
| **CRITICAL** | このままバックテストを走らせると破綻する | 即時修正必須 |
| **HIGH** | 結果が大幅に間違う。性能に深刻な影響 | 修正推奨 |
| **WARN** | 動作はするが将来的に問題になりうる | 改善した方が良い |
| **INFO** | 情報提供のみ | 対応不要 |

---

## 実行手順【必ず以下の順番で行うこと】

### Step 1: 仕様書と最新コードを読む

```
1. このファイル（VERIFICATION_CHECKER.md）を read_file で読む（←今これをやっている）
2. backtest/VERIFICATION_SPEC.md を read_file で読む（既知バグ一覧・仕様確認）
3. backtest/strategy_search.py を read_file で読む（全体を読むこと）
4. backtest/pdca_state.json を read_file で読む（最新の改善内容）
5. git diff HEAD~1 backtest/strategy_search.py を run_command で実行（今回の変更内容）
```

### Step 2: 動的検証スクリプトを生成・実行する

以下の Python スクリプトを `/tmp/verify_strategy.py` に write_file で書き出し、
`python3 /tmp/verify_strategy.py` を run_command で実行すること。

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
        errors.append(f"[B-CRITICAL] BUY_THRESHOLDS 全て同値に縮退: {BUY_THRESHOLDS}")
    if len(set(SWING_BUY_THRESHOLDS)) == 1:
        errors.append(f"[B-CRITICAL] SWING_BUY_THRESHOLDS 全て同値に縮退: {SWING_BUY_THRESHOLDS}")
    if _TOFF_CLAMPED is None:
        warnings.append("[B-WARN] _TOFF_CLAMPED が存在しない → 閾値崩壊リスク")
    if MIN_TRADES_OOS > 4:
        errors.append(f"[B-HIGH] MIN_TRADES_OOS={MIN_TRADES_OOS} > 4 → 4取引以下のホールドアウトがSharpe=0になる")

except Exception as e:
    errors.append(f"[B-CRITICAL] 閾値パラメータ読み込み失敗: {e}")


# ── C. detailed_eval_single の境界ケース ─────────────────────────────────────
try:
    from backtest.strategy_search import detailed_eval_single
    import inspect
    src = inspect.getsource(detailed_eval_single)

    if "profit_factor" not in src:
        errors.append("[C-HIGH] detailed_eval_single に profit_factor がない → PF が常に 0.0")
    if 'n == 0' not in src and 'n==0' not in src:
        errors.append("[C-HIGH] detailed_eval_single: n=0 ガードが見当たらない → ゼロ除算リスク")

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
        errors.append("[D-HIGH] GAチェックポイントに threshold_offset がない")

except Exception as e:
    warnings.append(f"[D-WARN] run_ga_all_sells 読み込みスキップ: {e}")


# ── E. WF安定性の数値安定性 ──────────────────────────────────────────────────
try:
    import pathlib
    src_full = pathlib.Path("/home/user/Stock_watch/backtest/strategy_search.py").read_text()

    if "oos_valid" not in src_full and "oos_arr[oos_arr" not in src_full:
        errors.append("[E-HIGH] WF安定性計算: ゼロフォールドの除外コードが見当たらない")

    import re
    wf_block = re.search(r'wf_stability.*?\n(?:.*?\n){0,10}', src_full)
    if wf_block and '1e-6' not in wf_block.group() and 'eps' not in wf_block.group():
        warnings.append("[E-WARN] WF安定性の分母に epsilon ガードが見当たらない")

except Exception as e:
    warnings.append(f"[E-WARN] WF安定性チェックスキップ: {e}")


# ── F. Python文法チェック ────────────────────────────────────────────────────
try:
    import ast, pathlib
    src = pathlib.Path("/home/user/Stock_watch/backtest/strategy_search.py").read_text()
    ast.parse(src)
    print("\n  [F] Python文法チェック: OK")
except SyntaxError as e:
    errors.append(f"[F-CRITICAL] Python文法エラー: {e}")


# ── 結果出力 ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ERRORS:", len(errors))
for e in errors:   print(" ✗", e)
print("WARNINGS:", len(warnings))
for w in warnings: print(" ⚠", w)
print("="*60)
print("RESULT:", "FAIL" if errors else ("WARN" if warnings else "PASS"))
```

### Step 3: 静的読み込みによる補完チェック

動的実行を補完するため、コードを読んで以下を確認する：

1. **データリーケージ**: `t_start` / `t_end` の境界が `t_holdout` を超えていないか
2. **配列長不一致**: `ind_scores[:, t_start:t_end]` と `sell_outcomes[t_start:t_end]` のスライスが一致しているか
3. **改善理由との整合**: `pdca_state.json` の最新 `decision` に記載された改善内容がコードに反映されているか
4. **VERIFICATION_SPEC.md §既知バグ修正チェックリスト** の全項目が満たされているか
5. **Python文法エラー**: `ast.parse()` でエラーが出ないか（Step 2 の F チェックで確認済みのはず）

---

### Step 4: 報告【詳細フォーマット必須】

以下のフォーマットで**全項目を埋めて**報告すること。
「✓ 問題なし」だけでは不十分。何を確認してどういう値だったかを具体的に記述する。

```
════════════════════════════════════════════════════════
  チェッカー レポート
  日時: YYYY-MM-DD HH:MM
  RESULT: PASS / WARN / FAIL
════════════════════════════════════════════════════════

【動的検証結果】
─────────────────────────────────────
[スコア関数の出力範囲]
  score_rsi_bull    : min=X.XX  max=X.XX  nan=N
  score_bb_bull     : min=X.XX  max=X.XX  nan=N
  score_relvol      : min=X.XX  max=X.XX  nan=N
  score_momentum_3b : min=X.XX  max=X.XX  nan=N
  score_vwap_bull   : min=X.XX  max=X.XX  nan=N

[閾値パラメータ]
  BUY_THRESHOLDS       = [X, X, X, X, X]
  SWING_BUY_THRESHOLDS = [X, X, X, X, X]
  _TOFF=N  _TOFF_CLAMPED=N  MIN_TRADES_OOS=N

[Python文法チェック]: OK / エラーあり

ERRORS  : N件
WARNINGS: N件


【発見された問題一覧】
─────────────────────────────────────
（問題がない場合は「問題なし」と記載）

### [重要度] カテゴリ-ID: タイトル

**何が起きているか（事実）**
  [実際に検出された値・コードスニペットを引用]

**なぜ問題か（影響）**
  [バックテスト結果に与える具体的な影響を数値・論理で説明]

**正しい動作はどうあるべきか**
  [期待される正しい値・動作]

**修正箇所**
  ファイル : backtest/strategy_search.py
  関数名  : xxx
  行番号  : 約NNN行目
  修正内容 : [具体的な変更指示]


【静的検証結果】
─────────────────────────────────────
[データリーケージ]: [✓問題なし / ✗問題あり → 詳細]
[配列スライス整合]: [✓一致 / ✗不一致 → 詳細]
[pdca_state.json decision との整合]:
  最新decision: 「[内容を引用]」
  コード反映: [✓済み / ✗未反映]
[既知バグ修正チェックリスト]:
  #1 MIN_TRADES_OOS <= 3   : [✓ / ✗ / ⚠再発]
  #2 BUY_THRESHOLDS 多様性  : [✓ / ✗ / ⚠再発]
  #3 oos_valid フィルタ     : [✓ / ✗ / ⚠再発]
  #4 GAチェックポイントキー  : [✓ / ✗ / ⚠再発]
  #5 スコア上限 max=25      : [✓全関数OK / ✗一部NG（関数名）]
  #6 profit_factor 伝播     : [✓ / ✗]
  #7 swing WFフォルド=6     : [✓ / ✗]


【総合評価】
─────────────────────────────────────
RESULT: PASS の場合   → 「問題なし。最適化を進めてください。」
RESULT: WARN の場合   → 「軽微な警告あり。改善推奨だが最適化は続行可能。」
RESULT: FAIL の場合   → 「重大なエラーあり。以下を修正してから再実行してください。
                          [修正が必要な箇所を箇条書きで列挙]」
════════════════════════════════════════════════════════
```

---

## 重要な原則

- **実行して確認する**。静的なコードパターン検索だけでは不十分。必ずPythonスクリプトを実行すること。
- **コードは変更しない**。問題を正確に特定して報告することだけが役割。
- **FAIL が出ても全チェックを完走**してから一括報告する。途中で止まらない。
- **「問題なし」は証拠を示す**。✓ の場合も確認した値を具体的に書く。
- **数値で語る**。「スコアが大きい」ではなく「max=25.0 を確認」のように具体的な値を示す。
- **PASS なら必ず「最適化を進めてください」と明記する**。
