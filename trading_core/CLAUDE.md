# trading_core — Claude Code 作業規約

## 最重要
- **正典の優先順位: `SPEC_ADDENDUM_v2.md` > `SPEC.md`**。矛盾する場合は v2 addendum を優先する。
  v2 は SPEC.md の改訂差分（R1執行=現物ロング＋指数CFDヘッジ / R2日足シグナル /
  R3グリッド+台地選択 / R4レジーム二重化 / R5 QQQ必須ゲート / F分岐フロー / H Phase 3a・3b二本立て）。
- 矛盾する判断が必要になったら該当仕様書を先に更新してから実装する。
- 本ディレクトリは旧システム（Stock_watch の `backtest/` 等）から**完全に独立**している。
  旧システムのアルゴリズムを参照・流用しない。参照してよいのは銘柄リスト等のデータのみ。
- 最終的に `auto_trader` リポジトリのルートとなる前提で、`trading_core/` 外への依存を作らない。

## 絶対ルール（違反は重大バグ扱い）
1. **先読みバイアス禁止**: 特徴量・シグナル・ゲート・サイジングは `available_date <= as_of`
   のデータのみ参照する。データ取得は `data/bitemporal_store.py` の `as_of` API 経由のみ。
   別の取得経路を作らない。
2. features / signals / risk の全モジュールに **available_date 違反（先読み）検出の単体テスト**を必ず付ける。
3. 成績として報告してよいのは **WFA/CPCV のアウトオブサンプル、コスト控除後**の数字のみ。
4. パラメータのハードコード禁止。`config/*.yaml` に集約する。
5. APIキー・シークレットは `.env` のみ（`.gitignore` 済み）。コミット前にシークレット混入チェック。
6. 全 GA 試行・アブレーション・手動閾値調整を `optimize/trial_logger.py` で SQLite に記録する
   （DSR の試行数補正の分母）。
7. 関数シグネチャを変えたら全呼び出し元を監査する。

## v2 の要点（実装済み・変更時は必ず整合を保つ）
- **執行 (R1)**: ロング＝現物株（`instrument="cash"`、金利なし）。個別株ショートは廃止。
  ベア/クライシス時のみ指数CFDヘッジ（`signals/index_hedge.py`、β露出の30〜70%）。
- **時間軸 (R2)**: シグナル・特徴量・レジームは日足（`features/daily_features.build_signal_frame`）。
  1h足は執行タイミングと場中ストップ監視のみ。エンジンは resting stop で場中逆指値を再現。
- **最適化 (R3)**: 主経路は `optimize/grid_runner.py`（粗グリッド＋台地選択、戦略あたり6パラメータ上限）。
  GAは実験用の副経路。全試行は trial_logger 必須。
- **レジーム (R4)**: `features/simple_regime.py`（QQQ>200MA & VIX<閾値）が基準。
  HMMがOOSでこれに勝てなければHMM廃止。
- **評価 (R5)**: QQQ買い持ちに Sharpe で勝ち MaxDD で浅いこと（必須ゲート）。
  合格ライン Sharpe≥0.6 / CAGR≥10% / MaxDD≤25%。IS Sharpe>2.0 は過学習アラート。
- **分岐 (F)**: Phase 3 結果で [A]alt-dataへ / [B]簡素化ラダー / [C]RSローテーションへピボット /
  [D]リスク管理オーバーレイ。判定は `validation/baseline_compare.classify_branch`。

## フェーズ運用
- SPEC.md §9 ＋ addendum §H の受入基準を満たすまで次 Phase に進まない。
- 現状: Phase 1〜2 実装済み、Phase 3a/3b の測定基盤実装済み（データ投入待ち）。
  - Phase 1 の (b)(c) 監査: `python -m data.coverage_audit` / `python -m data.news_audit`（要Alpacaキー）
  - VIX取込: `python -m data.vix_ingest`（FRED VIXCLS、キー不要）
  - Phase 3 測定: `python -m validation.phase3_runner`（`--grid` で fold毎グリッド＋台地選択）

## 開発環境
- Python 3.11+ / `uv` 管理。`uv sync` → `uv run pytest` でテスト実行。
- タイムスタンプは全て UTC の timezone-aware で扱う。
- FinBERT はモデルバージョン固定・スコアはキャッシュ済みのもののみバックテストで参照。
