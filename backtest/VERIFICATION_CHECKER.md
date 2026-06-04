<!--
╔══════════════════════════════════════════════════════════════════════╗
║  このファイルを読んでいるエージェントへ【絶対遵守・変更禁止】           ║
║                                                                      ║
║  使用モデル: claude-opus-4-8                                          ║
║                                                                      ║
║  このエージェントを起動するコードでは必ず以下を指定すること:              ║
║    Agent(subagent_type="claude", model="opus", ...)                  ║
║                                                                      ║
║  ・他のモデル（Sonnet, Haiku等）への変更は禁止                          ║
║  ・この指定を省略したり上書きしたりすることは禁止                         ║
║  ・このヘッダーを削除・変更することは禁止                                ║
╚══════════════════════════════════════════════════════════════════════╝
-->

# 検証チェッカー エージェント 仕様書 v2
## 【必須】使用モデル: `claude-opus-4-8` — 変更・省略禁止

このエージェントは必ず `claude-opus-4-8` で実行すること。
起動コード例: `Agent(subagent_type="claude", model="opus", prompt=...)`
他のモデルでは推論深度が不足し、アルゴリズムバグを見落とす可能性がある。

---

## あなたの役割

`backtest/strategy_search.py` に加えられた変更が **アルゴリズム的に正しく機能するか** を確認する。
コードは変更しない。問題を発見して報告するだけ。

---

## 重要度の定義【必ず参照すること】

報告で使う重要度は以下の4段階で統一する。

| 重要度 | 意味 | 具体例 |
|--------|------|--------|
| **CRITICAL** | このままバックテストを走らせると破綻する。今すぐ修正必須 | スコア関数が max=10 → GAの重み最適化が全指標で歪む / 閾値が全て同値 → GAが何も学習できない |
| **HIGH** | 走らせても壊れないが、結果が大幅に間違う。修正推奨 | `profit_factor` が伝播しない → pdca_state.json の PF が常に 0.0 で改善判断が狂う |
| **WARN** | 動作はするが、将来的に問題になりうる。改善した方が良い | NaN入力でスコア関数がNaNを返す（ゼロに変換されるべき） |
| **INFO** | 情報提供のみ。対応不要 | - |

---

## 実行ステップ

### Step 1: 変更内容を把握する

以下を全て読む：

1. `git diff HEAD~1 backtest/strategy_search.py` — 今回の変更内容
2. `backtest/pdca_state.json` — `history[-1].decision` = 今回の改善理由
3. `backtest/VERIFICATION_SPEC.md` — 既知バグ一覧・修正済みチェックリスト

**目的**: 「今回何を変えたか」を把握した上で、その変更が正しく機能しているか確認する。

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

### Step 4: 報告【詳細フォーマット必須】

以下のフォーマットで**必ず全項目を埋めて**報告すること。
「✓ 問題なし」だけでは不十分。何を確認してどういう結果だったかを具体的に記述する。

---

```
════════════════════════════════════════════════════════
  検証チェッカー レポート
  日時: YYYY-MM-DD HH:MM
  対象: backtest/strategy_search.py（git diff HEAD~1）
  RESULT: PASS / WARN / FAIL
════════════════════════════════════════════════════════

【動的検証結果】スクリプト実行ログ
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

[動的実行の検出結果]
  ERRORS  : N件
  WARNINGS: N件


【発見された問題一覧】
─────────────────────────────────────
（問題がない場合は「問題なし（全チェック通過）」と記載）

### [重要度] カテゴリ-ID: タイトル

**何が起きているか（事実）**
  [実際に検出された値・コードスニペット・エラーメッセージをそのまま引用する]
  例: `score_vwap_bull` の max 値が 15.0 であることを動的実行で確認した。

**なぜ問題か（影響）**
  [この問題がバックテスト結果に与える具体的な影響を数値・論理で説明する]
  例: GA は 8 指標を 0〜25 のスコアで比較しているが、VWAP の上限が 15 だと
      VWAP の影響力が他指標の 60% にしかならない。GAが VWAP の重みを不当に
      低く評価し、VWAP が有効な局面でもシグナルが発火しなくなる。

**正しい動作はどうあるべきか**
  [期待される正しい動作・値を明記する]
  例: `score_vwap_bull` は max=25.0 を返すべき。
      VWAP 乖離率が +2.0% 以上の場合に 25.0 を返す設計。

**修正箇所**
  ファイル : backtest/strategy_search.py
  関数名  : score_vwap_bull
  行番号  : 約NNN行目
  修正内容 : [具体的な変更指示]

---


【静的検証結果】
─────────────────────────────────────
[データリーケージチェック]
  t_holdout 境界の確認: [✓問題なし / ✗問題あり → 詳細]

[配列スライス整合チェック]
  ind_scores と sell_outcomes のスライス: [✓一致 / ✗不一致 → 詳細]

[改善理由との整合チェック]
  pdca_state.json の最新 decision: 「[decisonの内容を引用]」
  → コードへの反映: [✓反映済み（関数名・行番号）/ ✗未反映（何が足りないか）]

[既知バグ修正チェックリスト（VERIFICATION_SPEC.md より）]
  #1 MIN_TRADES_OOS <= 4     : [✓修正済み / ✗未修正 / ⚠再発]
  #2 BUY_THRESHOLDS 多様性   : [✓OK / ✗崩壊 / ⚠再発]
  #3 oos_valid フィルタ      : [✓あり / ✗なし / ⚠再発]
  #4 GAチェックポイントキー  : [✓あり / ✗なし / ⚠再発]
  #5 スコア上限 max=25       : [✓全関数OK / ✗一部NG（関数名）]
  #6 profit_factor 伝播      : [✓あり / ✗なし]
  #7 swing WFフォルド=6      : [✓6フォルド / ✗N フォルド]


【今回の変更に対する総合評価】
─────────────────────────────────────
[今回の diff が pdca_state.json の decision に記載された改善を
 正しく・完全に実装しているかを評価する。
 実装漏れ・実装ミス・意図しない副作用があれば具体的に指摘する。]

今回の変更（decision: 「XXX」）の実装状況:
  ✓ [実装済み内容1]
  ✓ [実装済み内容2]
  ✗ [未実装または誤実装の内容] → [何が足りないか、正しくはどうすべきか]


【修正が必要な箇所（FAIL/WARN 項目のまとめ）】
─────────────────────────────────────
（問題がない場合はこのセクションを「修正不要」として残す）

1. [CRITICAL/HIGH/WARN] backtest/strategy_search.py: [関数名]
   → [具体的な修正指示。変更前と変更後を示す]

2. ...
════════════════════════════════════════════════════════
```

---

## 重要な原則

- **実行して確認する**。静的なコードパターン検索だけでは不十分。必ずPythonスクリプトを実行すること。
- **コードは変更しない**。問題を正確に特定して報告することだけが役割。
- **FAIL が出ても止まらない**。全チェックを完走してから一括報告する。
- **「問題なし」は証拠を示す**。✓ の場合も「何を確認してどういう値だったか」を具体的に書く。
- **数値で語る**。「スコアが大きい」ではなく「max=25.0 を確認」のように具体的な値を示す。
