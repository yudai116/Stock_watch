# VERIFICATION_CHECKER.md — チェッカーエージェント指示書

チェッカーは **「提案された改善がコードに正しく反映されているか」** を確認するread-onlyエージェントである。
コードは変更しない。PASS/FAILと問題箇所のみを返す。

---

## 必須ステップ（毎回全ステップ実行）

### Step 0: 変更差分の把握

```bash
git diff HEAD~1 --stat          # 変更ファイル一覧
git diff HEAD~1 -- <file>       # 各ファイルの具体的変更
```

変更されたファイル、追加されたパラメータ名、削除されたロジックを全て把握する。

---

### Step 1: 【最重要】呼び出し元監査 (Caller Audit)

> **これが最も見逃しやすいミスの源泉。** 関数定義にパラメータを追加しても、
> 呼び出し元が渡していなければ実質的に未使用で完全に無効になる。

**変更された関数ごとに以下を実行する:**

```bash
# 1. 変更された関数名を特定
# 例: run_walk_forward, detailed_eval, optimize_all_sells 等

# 2. その関数を呼び出している全ての箇所をgrepで検索
grep -rn "run_walk_forward\|detailed_eval" /home/user/Stock_watch/backtest/ --include="*.py"

# 3. 各呼び出し箇所を Read で開き、新規パラメータが渡されているか確認
```

**確認項目:**
- 新しく追加されたパラメータが全ての呼び出し元で渡されているか
- デフォルト値(None等)のまま放置されている呼び出し元がないか
- `pipeline.py` が特に重要（エントリーポイント）— 必ず最初に確認する

**判定基準:**
- 新パラメータが1箇所でも渡されていない呼び出し元がある → **FAIL (Critical)**

---

### Step 2: 動的検証（実際にコードを実行）

変更された関数を合成データで実際に実行し、挙動を確認する。

```python
import sys
sys.path.insert(0, '/home/user/Stock_watch')
# ...各関数のテストコード...
```

**必ず確認する動的検証項目:**

#### A. スコア関数の出力範囲 [0, 25]
```python
from backtest.strategy.indicators import *
import numpy as np
# 各スコア関数を境界値で実行して [0, 25] 範囲外がないか確認
for fn, args in [...]:
    out = fn(*args)
    assert out.max() <= 25.01, f"{fn.__name__}: max={out.max()}"
    assert out.min() >= -0.01, f"{fn.__name__}: min={out.min()}"
```

#### B. 新パラメータの実際の効果確認
追加されたパラメータ（例: `regime_states`）を実際に渡したケースと渡さないケースで
出力が変わることを確認する。変わらない場合はパラメータが内部で使われていない。

```python
# 例: regime_states の効果確認
stats_without = fn(..., regime_states=None)
stats_with    = fn(..., regime_states=some_filter_array)
assert stats_with['n_trades'] < stats_without['n_trades'], "フィルターが効いていない"
```

#### C. 閾値・定数の崩壊チェック
```python
from backtest.config import BUY_THRESHOLDS, WF_N_FOLDS, MIN_TRADES_OOS
assert len(set(BUY_THRESHOLDS)) > 1, "BUY_THRESHOLDSが単一値に縮退"
assert WF_N_FOLDS >= 4, f"フォールド数が少なすぎる: {WF_N_FOLDS}"
```

---

### Step 3: 静的コード読み込み確認

変更ファイルを Read ツールで開き、以下を確認する:

1. **新パラメータが関数本体で実際に使われているか**（定義はあっても使われていない場合がある）
2. **条件分岐が正しいか**（`if regime_states is not None:` の形式か）
3. **型の整合性**（`np.ndarray` vs `list` 等のミスマッチ）
4. **インポート文の漏れ**（新しく使う型やモジュールがimportされているか）
5. **`pipeline.py` の全呼び出し箇所**（Step 1と重複するが必ず目視確認する）

---

### Step 4: インポート・文法エラーチェック

```python
import subprocess
result = subprocess.run(
    ['python3', '-c',
     'import backtest.strategy.indicators; '
     'import backtest.strategy.ga_optimizer; '
     'import backtest.strategy.walk_forward; '
     'import backtest.config; '
     'import backtest.pipeline; '
     'print("全モジュール OK")'],
    capture_output=True, text=True, cwd='/home/user/Stock_watch'
)
print(result.stdout)
if result.returncode != 0:
    print("FAIL:", result.stderr)
```

---

## 出力フォーマット

```
RESULT: PASS または FAIL

[Step 1] 呼び出し元監査
✓/✗ run_walk_forward() の呼び出し元 (pipeline.py L162): regime_states=... 確認
✓/✗ detailed_eval() の呼び出し元 (pipeline.py L199): regime_states=... 確認

[Step 2] 動的検証
✓/✗ score_ema200_swing: max=25.0, min=0.0, 30%乖離→25pts
✓/✗ regime_statesフィルター効果: フィルターなし=200取引, あり=100取引

[Step 3] 静的確認
✓/✗ pipeline.py L162: regime_states=regime_states 渡し確認
✓/✗ detailed_eval() 内部で regime_states が vmask に AND 結合されているか

[Step 4] インポート
✓/✗ 全5モジュール インポート成功

問題あり項目（FAILの場合のみ）:
- ファイル名:行番号: 具体的な問題と修正案
```

---

## 過去の重大ミス事例（毎回このリストをチェックリストとして使用）

| # | ミスの種類 | 見逃した理由 | 対策 |
|---|-----------|------------|------|
| 1 | `regime_states` を `pipeline.py` で渡し忘れ | 関数定義側しか確認しなかった | Step 1の呼び出し元監査を最初に実行 |
| 2 | `detailed_eval()` の `max(years, 0.1)` 修正漏れ | `batch_sharpe()` のみ修正、同一ロジックの別関数を見落とし | 同じロジックが複数箇所にある場合は全て確認 |
| 3 | `oos_valid` の `!= 0.0` フィルター（浮動小数点） | 静的に正しく見えた | `1e-6` 等の境界値テストを動的検証に追加 |
| 4 | `profit_factor` の伝播漏れ | 返り値dictのキーを全確認しなかった | 変更関数の返り値dictの全キーをリストアップして確認 |

---

## 重要な制約

- チェッカーはコードを変更しない（read-only）
- Step 1（呼び出し元監査）を省略しない
- 「定義にある = 動作している」は誤り。必ず呼び出し元まで追跡する
- `pipeline.py` は全ての呼び出しのエントリーポイント — 毎回必ず確認する
