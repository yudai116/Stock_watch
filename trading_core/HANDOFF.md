# HANDOFF.md — 引き継ぎ文書（次セッションのAIは最初にこれを読むこと）

> 最終更新: 2026-07-07 / 作成: Claude (Fable 5) セッション
> 対象読者: このリポジトリを引き継ぐ AI アシスタント（Opus 等）と開発者本人
> **読む順序: 本書 → CLAUDE.md → SPEC_ADDENDUM_v2.md → SPEC.md**

---

## 0. 一言サマリー

米国テック株スイングトレードの自動売買システム。**コードは v2 仕様の完全実装済み
（テスト127件全通過）**。実データ投入済みで、**データはブランチにも同梱**
（`datastore/` を force-add 済み → クラウド側でも全測定を実行可能）。

**Phase 3 は最終裁定済み（2026-07-08、詳細 = `reports/phase3_final_adjudication.md`）**:
フローF **[D] を採択**。本番戦略 = **3d レジームオーバーレイ**（QQQコア＋単純
フィルタ、調整パラメータ0個）。WFA-OOS: Sharpe 0.85 / CAGR 17.8% / MaxDD -18.8%。
ホールドアウト(1回開封済み): Sharpe 0.99 / CAGR 15.5% / MaxDD -11.7% — 挙動再現。
HMM廃止（単純フィルタに完敗）。グリッド最適化は3a/3b双方でOOS悪化＝選択ノイズと
結論。Phase 4 (alt-data) は実施しない。
**次の仕事 = ライブ運用ランナーの実装 → Saxo SIMでペーパー3か月（addendum G）**。
ホールドアウトは開封済みのため再利用禁止。

---

## 1. 正典と絶対ルール

- 仕様の優先順位: **SPEC_ADDENDUM_v2.md > SPEC.md**。矛盾したら v2 が勝つ。
- 絶対ルール（CLAUDE.md に詳細）:
  1. **先読み禁止**。全データ参照は bitemporal store の `as_of` API 経由のみ。
     features/signals/risk に「未来摂動テスト」必須。
  2. 成績として報告してよいのは **OOS・コスト控除後** のみ。
  3. パラメータは `config/*.yaml` に集約。ハードコード禁止。
  4. 全試行（グリッド・GA・手動調整）を `optimize/trial_logger.py` で記録
     （DSR の分母）。
  5. **commit/push はユーザー承認後**（調査・ローカル編集は自動でよい）。
     ※ 本セッションではユーザーが都度 push を承認してきた実績あり。
  6. 関数シグネチャ変更時は全呼び出し元を監査。

## 2. リポジトリ / ブランチ状態

- 作業ブランチ: `claude/tech-swing-trading-system-i9zzjw`（origin に push 済み）
- 本体は `trading_core/` 以下で完全自己完結。旧システム（リポジトリ直下の
  `backtest/` 等）への依存ゼロ。**将来 auto_trader リポジトリのルートへ移植予定**
  （ユーザーの本来の希望。このセッションのアクセス制約で Stock_watch に構築した）。
- テスト: `cd trading_core && uv sync && uv run pytest` → **122 passed** を維持すること。

## 3. アーキテクチャ地図（どこに何があるか）

| パス | 役割 | 状態 |
|---|---|---|
| `data/bitemporal_store.py` | event_ts/available_ts 二軸ストア。唯一のデータ取得経路 | 完成 |
| `data/polygon_ingest.py` | **主データ源**（10年・廃止銘柄対応）。生バー+分割/配当 | 完成 |
| `data/alpaca_ingest.py` | 副データ源（無料IEX、履歴6年）。**今後の日次更新用** | 完成 |
| `data/news_ingest.py` | Alpaca News（Phase 4 用） | 完成・未使用 |
| `data/vix_ingest.py` | FRED VIXCLS（キー不要） | 完成・投入済み |
| `data/edgar_ingest.py` | EDGAR XBRL ファンダ（Phase 4 用） | 完成・未使用 |
| `data/pit_universe.py` | 四半期PITユニバース + `membership_provider` | 完成 |
| `data/ingest_universe.py` | universe.yaml の全候補117銘柄を一括取得 | 完成 |
| `data/dedupe_daily.py` | 混在ソース重複の掃除（§8参照） | 完成・実行済み |
| `data/export_bars.py` | CSV バックアップ | 完成・実行済み |
| `data/coverage_audit.py` | 廃止銘柄カバレッジ監査（`--source polygon`） | 92%=Sufficient |
| `features/simple_regime.py` | 単純レジーム（QQQ>200MA & VIX<25）= 現行本線 | 完成 |
| `features/regime_hmm.py` | HMM（BIC状態数選択・WF推論）= 挑戦者 | 完成・比較未実施 |
| `features/daily_features.py` | 日足シグナル特徴量 `build_signal_frame` | 完成 |
| `signals/composite.py` | **3b フル戦略**（Donchianブレイク+ゲート+ヘッジ+場中ストップ） | 完成 |
| `signals/rs_rotation.py` | **3a 週次RSローテーション**（パラメータ3個） | 完成 |
| `signals/index_hedge.py` | ベア時 QQQ CFD ショートヘッジ（β×30-70%） | 完成 |
| `signals/gates.py` | 決算ブラックアウト/ニュースショック/踏み上げ回避 | 完成・Phase4で活性化 |
| `risk/*` | ボラターゲット/確信度乗数/ポートフォリオ制約/DD制御 | 完成 |
| `backtest/engine.py` | 日足イベント駆動。翌日寄付約定+**場中ストップ**(安値タッチ/ギャップ寄付) | 完成 |
| `backtest/costs.py` | 現物(金利なし)/指数CFD(金利あり)の2プロファイル | 完成 |
| `optimize/grid_runner.py` | **主最適化**: 粗グリッド+台地選択(6パラメータ上限) | 完成 |
| `optimize/ga_runner.py` | GA（副経路・実験用） | 完成 |
| `validation/phase3_runner.py` | **中枢**。3a/3b をWFAで測定→QQQゲート→DSR→分岐判定 | 完成 |
| `validation/dsr.py, cpcv.py, walk_forward.py, monte_carlo.py` | 統計装置 | 完成（CPCVは未結線） |
| `execution/saxo_client.py` | Saxo OAuth + 現物/指数CFD発注（Phase 5 用） | 完成・未検証 |

## 4. ユーザー環境（重要）

- **Windows 11 / PowerShell / `C:\projects\projects\auto-trader\Stock_watch\trading_core`**
- Python 3.11+ / `uv`。実行はすべて `uv run python -m ...` 形式。
- `.env`（ユーザーPCにあり、git外）: `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`
  （ペーパー口座・IEXフィード）/ `ALPACA_FEED=iex` / `EDGAR_USER_AGENT` /
  `POLYGON_API_KEY`。
- **Polygon は解約予定/済み**。日足10年×117銘柄+コーポレートアクション756件は
  取得済みで `datastore/`（Parquet+SQLite、git外）に永久保存。CSV複製も
  `exports/1d/` に作成済み（116銘柄・247,954本）。
- Polygon の intraday（1h足）はプラン権限外で 401 → **1h履歴は未取得のまま解約**。
  設計上問題なし（シグナルは日足。ライブの1hは無料Alpacaで賄う）。
- 今後の日次更新: `uv run python -m data.alpaca_ingest --symbols ... --timeframes 1d`
  （無料IEX。日足打刻は正規化済みでPolygon履歴と正しく重なる）。

## 5. 計測の現在地

### 5.1 実施済み測定（時系列）
1. **旧34銘柄・PIT** での Phase 3（grid無し・デフォルトパラメータ）:
   - 3a: OOS Sharpe 0.45 / CAGR 7.6% / MaxDD -24.7% / QQQゲートFAIL → [B]
   - 3b: OOS Sharpe 0.85 / CAGR 3.5% / MaxDD -5.6% / **QQQゲートPASS** → [A]
   - ただし 3b は CAGR が合格ライン(≥10%)未達、DSR 0.751(<0.95)
   - OOS窓は2021-10〜2025-01（2022年ベア込み）。ホールドアウトは2025-01-06以降
2. **117銘柄・クリーンデータ・PIT(38四半期・毎期40銘柄)** での Phase 3
   （grid無し・デフォルトパラメータ、OOS=2020-10〜2025-01、判読済み 2026-07-07）:
   - 3a: Sharpe 0.47 / **CAGR 16.4%**(34銘柄時7.6%から倍増=プール拡大が機能) /
     MaxDD **-28.2%**(上限25%超過) / DSR 0.126(N=61) / ゲートFAIL → **[B]**
   - 3b: Sharpe 0.52 / CAGR 2.5%(露出不足が構造課題) / MaxDD -5.4% /
     DSR 0.086(N=65) / ゲートFAIL → **[B]**
   - この OOS 窓の QQQ B&H は Sharpe **1.05**(AI相場直撃区間)= ゲートが非常に高い
   - 両変種 [B] → フローFどおり **B1ラダー+グリッド** が確定した次工程
3. B1グリッド(`composite_b1`: time_stop=15/partial=none固定で81組合せ)を実装済み。
   `--composite-grid composite_b1` が既定。
4. **3aグリッド結果（判読済み 2026-07-08）: 最適化で悪化** — OOS Sharpe
   0.47→0.24 / CAGR 16.4%→8.3% / DSR 0.012(N=424) → **[C]**。
   フォールド毎の訓練選択パラメータがOOSでデフォルトに負けた＝ローテーションの
   パラメータは期間安定せず、これ以上の最適化は選択ノイズ。**3aのグリッド再実行は
   不要**（デフォルトパラメータ版を3aの確定値として扱う）。
5. フローF [C]→[D] に備え **variant「3d」= レジームオーバーレイ**
   （QQQコア保有、bull=投資/range=保持/bear=現金化、探索パラメータ0個、
   `signals/regime_overlay.py`）をランナーに結線済み。
   **引き継ぎ後の仕事: (a) 3bのB1グリッド結果の判読（夜間実行分）、
   (b) `--variants 3d` の測定（1回のバックテストのみ・数分）、
   (c) 3b vs 3d vs QQQゲートで最終分岐を裁定**。
6. **3d測定結果（判読済み 2026-07-08）: 全変種中最良** —
   Sharpe 0.85 / CAGR 17.8% / MaxDD **-18.8%**(QQQ -35.1%のほぼ半分) /
   Calmar 1.72 / WF効率 1.23 / DSR 0.186(N=498) → R5合格ライン5項目中4クリア
   （残りはdsr@95のみ。N=498は全変種の試行込みで、0パラメータの3dには過剰に
   保守的な分母である点は解釈に付記してよい）。QQQゲートはSharpe僅差FAIL
   (0.85 vs 1.05、AI強気相場のOOS窓) → [B]。
   **残る測定は2つだけ**: (a) 3bのB1グリッド（夜間）、(b) `--regime hmm
   --variants 3d`（レジーム判定の質=3dの全てなので、R4のHMM対決はここが本丸）。
   両方出たら最終裁定: 3bが[A]なら3b本線 / 届かなければ **[D]確定で3dを本番戦略**
   （[D]の成功基準はQQQ Sharpe超えではなくDD圧縮。SPECの合理的着地点）→
   ホールドアウト1回 → ペーパーへ。

### 5.2 判定フレームワーク（数字を暗記せず config を見よ: `params.yaml`）
- R5 合格ライン: Sharpe≥0.6 / CAGR≥10% / MaxDD≤25% / WF効率≥0.5 / DSR>0.95相当
- **QQQ必須ゲート**: QQQ買い持ちに Sharpe で勝ち、MaxDD で浅い。落ちたら他が
  良くても不合格
- 分岐: [A] Sharpe≥0.8+ゲートPASS → Phase 4 / [B] 0.4〜0.8等 → 簡素化ラダー /
  [C] <0.4 → 3a を本線化 / [D] 全滅 → リスク管理オーバーレイへ転換
- レポートには `pass lines (R5): sharpe OK / cagr NG / ...` の逐条行が出る。
  **分岐ラベルだけ見ず、この行を必ず確認**（[A]でもCAGR未達がありうる）

## 6. 次にやることの詳細手順（この順で）

### Step 1: 117銘柄レポートの判読（最初の仕事）
ユーザーに `reports/phase3_report.md`（または画面）を貼ってもらい判定:
- `universe: PIT quarterly (NN quarters)` であること（STATICなら回し直し）
- 3a が旧34銘柄時より改善しているか（選抜が効き始めたか）
- 3b の CAGR が 10% に近づいたか、ゲート維持か
- 判定に応じ Step 2 へ（どの分岐でも Step 2 のグリッドは価値がある）

### Step 2: グリッド最適化（夜間実行を案内）
```powershell
uv run python -m validation.phase3_runner --grid
```
- 3b は 486組合せ×4フォールド → **数時間**。117銘柄なので旧実行より遅い。
  変種を絞るなら `--variants 3b`。出力: `reports/phase3_report_grid.md`
- 判定: CAGR≥10% と DSR≥0.95 が立ったか。IS Sharpe>2.0 の過学習アラート監視。
- **CAGRが依然未達の場合の診断方針**（実装はユーザー承認後）:
  露出不足が主因の可能性が高い。診断順: (a) 取引数と平均保有率（equity curve
  の投下資本比率）を集計 → (b) `fixed.portfolio_heat_max_pct`(6%) と
  `max_concurrent_positions`(8) の感度をアブレーションとして計測（無断で
  リスク上限を緩めない。trial_logger 記録必須）

### Step 3: HMM 対決（R4 の宿題）
```powershell
uv run python -m validation.phase3_runner --regime hmm --variants 3b
```
- 出力: `reports/phase3_report_hmm.md`。simple 版と同一設定の OOS 比較。
- **HMM が OOS で simple に勝てなければ HMM を廃止**（複雑さは成果で買う）。
  勝敗は Sharpe/Calmar/MaxDD の総合（同等なら simple 採用）。

### Step 4: 分岐処理
- **[A] 到達**（Sharpe≥0.8+ゲート+CAGR≥10%+DSR≥0.95 全部）→ Phase 4 へ。
  アブレーション順序は固定: **ファンダ品質→ニュースゲート→センチメントHMM→Trends**
  （`validation/ablation_runner.py` + `adoption` 設定が実装済み）。
  データ準備: `data/edgar_ingest.py`（要 EDGAR_USER_AGENT）、
  `data/news_ingest.py`（要Alpacaキー）、FinBERT スコア
  （`altdata/sentiment_scorer.py`、**model_revision の実SHA固定が必須**、
  `uv sync --extra altdata` で transformers 導入）。
- **[B]** → 簡素化ラダーを1段ずつ: B1 パラメータ6→4（time_stop/部分利確を固定）
  → B2 エントリーを週次RSローテ置換 → B3 レジームをsimple固定。
  各段で同一WFA再測定、改善したら確定。
- **[C]** → 3a を本線化し、3a のグリッド（18組合せ・軽い）で台地選択。
- **[D]**（全変種ゲート不合格）→ QQQ/SMH コア保有+レジーム防衛への転換を
  ユーザーと合意してから設計。

### Step 5: ホールドアウト（1回だけ・厳守）
全アブレーション完了後、**最終確定した1構成のみ**:
```python
from validation.phase3_runner import evaluate_holdout, default_params
# params は確定したもの。variant は "3a" か "3b"
```
- **選択の材料に使ったら無効**。2回目はない。結果が悪ければ設計へ戻る
  （ホールドアウトの再利用はしない）。

### Step 6: Phase 5（ペーパー→実弾、addendum G）
1. `execution/saxo_client.py` を SIM 環境で疎通（ユーザーが Saxo AppKey 取得後）
2. **ペーパー最低3か月**。日次: 引け後にシグナル計算→翌朝執行のオペレーション
   スクリプトが必要（**未実装**。live runner は次の大きな実装タスク）
3. 実測1トレード期待値 < バックテスト値の50% → 実弾禁止で原因究明
4. 実弾は資金25%から。四半期照合。**キルスイッチ: 実DD が検証MaxDD×1.25 で全解消**

## 7. バックログ（優先度順・未実装）

1. **ライブ運用ランナー**（Phase 5 前提。日次シグナル→Saxo発注→ポジション照合）
2. CPCV をランナーへ結線（現在WFAのみ。D8 は WFA×CPCV 併用を要求）
3. クロスソース価格検証（Polygon vs Alpaca 乖離>0.5%でブロック、旧CLAUDE.md要件）
4. quality_check を phase3_runner 実行前に自動実行
5. 1h執行オーバーレイのアブレーション（日足シグナル+場中ブレイク執行。優先度低）
6. ALTR のセクター誤分類修正（Altera のつもりが Altair Engineering。実害軽微）
7. META の FB 時代履歴の名寄せ（現状2021年以降のみ。実害軽微）
8. trends_collector の日次タスク登録（D10、実験扱い）

## 8. 実際に踏んだバグと教訓（同種の再発に注意）

| 事象 | 根因 | 教訓 |
|---|---|---|
| Alpaca `KeyError:'symbol'` | 無料口座で `feed="sip"` → 空応答 | 既定は IEX。`ALPACA_FEED` で切替 |
| `KeyError:'ALPACA_SECRET_KEY'` | 環境変数名の揺れ | `config_loader.alpaca_credentials()` が3系統の名前を吸収 |
| **VIX全行NaN→無取引の空バックテスト** | 打刻違いの `reindex().ffill()` | 時刻整合は `reindex(method="ffill")`。**テストは本番と同じ打刻で書く** |
| **QQQ 4003本（日足二重登録）** | Alpaca(04:00)とPolygon(00:00)の打刻混在 | 日足tsは取得時に正規化。`dedupe_daily` は冪等 |
| `KeyError:'event_ts'` | アクション0件銘柄で空DataFrame | 空リスト→DataFrame は列が消える。ガード必須 |
| 3aが好成績に見えた（デモ） | 生存者バイアス+合成QQQ | ランナーはPITユニバース必須化済み（`--static-universe` で明示承諾のみ許可） |
| `Series == pytest.approx` が偽 | approx はSeriesと要素比較しない | `np.allclose` を使う |
| ユーザーがチャットにAPIキーを貼った | — | 貼られたら再発行を案内。.env に直接書かせる |

## 9. ユーザーとの働き方（このプロジェクトの流儀）

- ユーザーは日本語。**専門用語は噛み砕き、手順は1コマンドずつ**
  PowerShell コピペ可能な形で渡す（実績のあるスタイル）。
- ユーザーはスクリーンショットで結果を貼る。エラーは自動で直して先へ進める
  ことを好む（「自律的に進めてください」の明示指示あり）。
- 大きな設計判断（リスク上限の変更、Phase 移行、実弾）は必ず確認を取る。
  実装の細部は任されている。
- 成績の解釈は**正直に**。良い数字でも留保（DSR未達・CAGR未達・バイアス）を
  必ず明示する。これがこのプロジェクトの信頼の基盤。
