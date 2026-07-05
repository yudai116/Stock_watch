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

## フェーズ運用
- SPEC.md §9 の受入基準を満たすまで次 Phase に進まない。
- 現状: Phase 1〜2 実装済み（受入テストは `tests/` 参照）。Phase 1 の (b)(c) 監査は
  Alpaca API キー設定後に `python -m data.coverage_audit` / `python -m data.news_audit` で実行。

## 開発環境
- Python 3.11+ / `uv` 管理。`uv sync` → `uv run pytest` でテスト実行。
- タイムスタンプは全て UTC の timezone-aware で扱う。
- FinBERT はモデルバージョン固定・スコアはキャッシュ済みのもののみバックテストで参照。
