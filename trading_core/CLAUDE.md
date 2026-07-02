# trading_core — Claude Code 作業規約

## 最重要
- **SPEC.md が唯一の正**。矛盾する判断が必要になったら SPEC.md を先に更新してから実装する。
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

## フェーズ運用
- SPEC.md §9 の受入基準を満たすまで次 Phase に進まない。
- 現状: Phase 1〜2 実装済み（受入テストは `tests/` 参照）。Phase 1 の (b)(c) 監査は
  Alpaca API キー設定後に `python -m data.coverage_audit` / `python -m data.news_audit` で実行。

## 開発環境
- Python 3.11+ / `uv` 管理。`uv sync` → `uv run pytest` でテスト実行。
- タイムスタンプは全て UTC の timezone-aware で扱う。
- FinBERT はモデルバージョン固定・スコアはキャッシュ済みのもののみバックテストで参照。
