# SPEC.md — 米国テック株スイングトレード自動売買システム 実装仕様書

> 本書は Claude.ai チャットでの設計フェーズの成果物であり、Claude Code での実装の唯一の正とする。
> 実装中に本書と矛盾する判断が必要になった場合は、本書を更新してから実装すること。

-----

## 0. 確定済みの設計決定（Decision Log）

|#  |決定事項         |内容                                                           |
|---|-------------|-------------------------------------------------------------|
|D1 |データソース       |Alpaca（契約済み）: 1h足・日足・News API                                |
|D2 |執行           |サクソバンク OpenAPI（CFD、ショート可）。OAuth は localhost:8765 コールバックフローを流用|
|D3 |ポジション方向      |ロング主体＋ベアレジーム時のみショート（非対称制約あり、§6.4）                            |
|D4 |保有期間         |数日〜3週間のスイング                                                  |
|D5 |ユニバース        |Point-in-Time（四半期リバランス）。固定25銘柄は廃止                            |
|D6 |廃止銘柄データ      |Alpaca のカバレッジ監査で不足なら Polygon 等（月$30程度）を PIT ユニバース構築用のみ追加許容   |
|D7 |チャート外データ     |ニュース／ファンダ／トレンドを「HMM入力・ゲート・サイジング乗数」の3役割に限定。GA特徴量への直接大量投入は禁止   |
|D8 |検証           |チャート単独ベースライン併走＋アブレーション＋WFA×CPCV＋DSR＋ホールドアウト1回                |
|D9 |実行環境         |デスクトップPC（Windows）。ローカル完結、サーバ不要                               |
|D10|Google Trends|真のPITでないため実験扱い。本日からライブ収集を開始し将来検証用に蓄積                         |

-----

## 1. 実行環境（デスクトップPC）

- OS: Windows 11（WSL2 不要。ネイティブ Python で完結させる）
- Python: 3.11 以上
- パッケージ管理: `uv`（venv + lock を一元管理）
- データ格納: ローカルディスクに Parquet（価格・特徴量）＋ SQLite（メタデータ・trial log・bitemporal インデックス）
- 主要ライブラリ: `polars`（大規模バー処理）, `pandas`, `numpy`, `hmmlearn`, `deap`（GA）, `alpaca-py`, `httpx`, `transformers`（FinBERT、CPU推論でよい）, `pyarrow`
- 想定データ量: 1h足 × 10年 × PITユニバース（各時点25〜40銘柄、延べ銘柄数はより多い）→ Parquet 圧縮で数GB程度。デスクトップで問題なし
- シークレット管理: `.env`（`.gitignore` 必須）。Alpaca API キー、Saxo の AppKey/Secret を格納。コミット禁止

-----

## 2. リポジトリ構成

```
trading_core/
├── SPEC.md                  # 本書
├── CLAUDE.md                # Claude Code 用の作業規約（/init 後に本書参照を追記）
├── pyproject.toml
├── .env.example
├── config/
│   ├── universe.yaml        # PITユニバース定義ルール
│   ├── params.yaml          # 探索パラメータ空間
│   └── costs.yaml           # サクソCFDコストモデル
├── data/
│   ├── alpaca_ingest.py     # 1h足・日足取得
│   ├── edgar_ingest.py      # 10-K/10-Q XBRL取得
│   ├── news_ingest.py       # Alpaca News API
│   ├── trends_collector.py  # Google Trends ライブ収集（日次cron/タスクスケジューラ）
│   ├── adjuster.py          # 分割・配当調整
│   ├── bitemporal_store.py  # event_date / available_date 二軸格納
│   ├── pit_universe.py      # 四半期リバランスPITユニバース構築
│   └── quality_check.py     # 欠損・異常値・出来高ゼロ検査
├── altdata/
│   ├── sentiment_scorer.py  # FinBERT。モデルバージョン固定・スコアキャッシュ
│   ├── fundamental_quality.py
│   └── latency_aligner.py   # ニュース→利用可能バー割り付け
├── features/
│   ├── daily_features.py    # 実現ボラ、相対強度(vs QQQ/SMH)、200日RS
│   ├── hourly_features.py   # ATR、Donchian、RSI/ROC、出来高z、MA傾き、価格z
│   └── regime_hmm.py        # Gaussian HMM（市場レベル4〜6次元入力）
├── signals/
│   ├── trend_follow.py      # ブレイクアウト型（ロング）
│   ├── mean_revert.py       # レンジレジーム用
│   ├── short_breakdown.py   # ベアレジーム用ショート
│   ├── regime_gate.py       # レジーム→戦略の割当
│   └── gates.py             # 決算ブラックアウト・ニュースショック・踏み上げ回避
├── risk/
│   ├── vol_target_sizer.py  # size = (資本×リスク率)/(k×ATR)
│   ├── confidence_multiplier.py  # 0.5〜1.5 乗数（§6.3）
│   ├── portfolio_limits.py  # 同時保有数・相関/セクター集中・portfolio heat
│   └── dd_control.py        # DD連動リスク縮小
├── backtest/
│   ├── engine.py            # イベント駆動。シグナル確定→次バー始値約定
│   ├── costs.py             # 手数料＋スプレッド＋スリッページ＋CFD金利
│   └── broker_sim.py
├── validation/
│   ├── walk_forward.py      # アンカー付きWFA
│   ├── cpcv.py
│   ├── dsr.py               # Deflated Sharpe Ratio（Bailey et al. 2014）
│   ├── ablation_runner.py
│   ├── baseline_compare.py  # チャート単独 vs alt-data版
│   └── monte_carlo.py
├── optimize/
│   ├── ga_runner.py
│   ├── param_space.py
│   └── trial_logger.py      # 全試行をSQLiteに記録（DSR補正の分母）
├── execution/
│   └── saxo_client.py       # OAuth (localhost:8765)、CFD発注
└── tests/
```

-----

## 3. データ層仕様

### 3.1 bitemporal 原則（全データ共通・違反は重大バグ扱い）

- すべてのレコードは `event_date`（事象の発生時点）と `available_date`（決定時点で入手可能になった時点）を持つ
- **特徴量・シグナル・ゲート・サイジングの計算は available_date 基準の参照のみ許可**
- バックテストエンジンは `as_of` タイムスタンプを渡し、store は `available_date <= as_of` のレコードのみ返す API とする（それ以外の取得経路を作らない）

### 3.2 ソース別 available_date 定義

|ソース            |event_date|available_date                                                              |
|---------------|----------|----------------------------------------------------------------------------|
|1h/日足バー        |バー期間      |バー確定時刻＋処理レイテンシ（1hバーは次バー開始時点で利用可）                                            |
|Alpaca News    |配信時刻      |配信時刻＋レイテンシ → **次の完全な1hバーから使用可**                                             |
|EDGAR 10-K/10-Q|対象四半期末    |**filing 受理日時**。引け後受理なら翌営業日。10-K/A 等の改訂は新 available_date の別バージョンとして追加（上書き禁止）|
|決算カレンダー        |発表予定日     |判明時点                                                                        |
|Google Trends  |対象週       |収集実行日（ライブ収集分のみ真のPIT）                                                        |

### 3.3 頻度整合

- ファンダ: 四半期 → available_date から前方フィル
- ニュース: 不規則 → latency_aligner でバー割り付け
- トレンド: 日〜週 → 水準ではなく**変化率**のみ使用

### 3.4 PIT ユニバース（pit_universe.py）

- 四半期ごとに「その時点で入手可能な情報」で対象セクター（半導体・AI・宇宙・核融合・メモリ・材料等の米国テック）から時価総額・流動性基準で25〜40銘柄を選定
- 上場廃止・被買収銘柄も当時ユニバースに含まれていれば当時のデータで取引対象とする
- **Phase 1 で Alpaca の廃止銘柄カバレッジを監査**（§9 Phase 1 受入基準参照）。不足時は Polygon 等で補完（D6）

-----

## 4. 特徴量層

- 日足系: HMMレジーム状態、実現ボラ（20日）、相対強度（vs QQQ / SMH）、200日RS
- 1h足系: ATR(14〜28探索)、Donchianブレイク幅、RSI/ROC、出来高z-score、MA傾き、価格z-score
- HMM入力（市場レベルのみ、合計4〜6次元厳守）: 市場リターン・実現ボラ・VIX・ブレッドス＋ユニバース集計ニュースセンチメントz・ネガティブ記事比率
- 個別銘柄ニュースを HMM に入れない。GA の探索特徴量にニュース系を直接入れない（D7）

-----

## 5. シグナル層

### 5.1 レジームゲート

- 強気/低ボラ → trend_follow 有効
- レンジ/中ボラ → mean_revert（または縮小運用、GAで選択探索）
- 高ボラ/危機 → 新規停止＋既存ポジ縮小
- ベア → short_breakdown 有効（ロング新規停止）

### 5.2 trend_follow（ロング）

エントリー = レジームゲート AND Donchian 20〜55期間高値ブレイク AND 相対強度上位 AND 出来高確認。GA探索対象は期間・閾値のみ。

### 5.3 イグジット（全戦略共通の枠組み）

1. シャンデリア型ATRトレーリング（k=2.5〜3.5 探索）
1. 初期損切り: エントリー時ATR基準固定
1. タイムストップ: N=10〜20営業日 進展なしで撤退
1. 固定利確目標なし（部分利確の有無はGA探索対象）

### 5.4 ゲート（gates.py、パラメータは固定・GA探索外）

- 決算ブラックアウト: 発表前 N=2〜3営業日 新規禁止
- ニュースショック: 強ネガティブスコア超過 → 当該銘柄の新規停止＋ストップタイト化
- 踏み上げ回避: 強ポジティブニュース直後のショート新規禁止

-----

## 6. リスク管理層

### 6.1 基本サイジング

`size = (資本 × 1トレードリスク率) / (k × ATR)`。1トレードリスク率は 0.5〜1.0% を探索。

### 6.2 ポートフォリオ制約

最大同時保有数、セクター集中上限、portfolio heat 上限、DD 連動のリスク縮小（dd_control）。

### 6.3 確信度乗数（0.5〜1.5、レンジ固定）

乗数 = ファンダ品質スコア（現金比率・負債トレンド・半導体は在庫回転変化）× センチメント整合性（シグナル方向とセクターセンチメントの一致度）。構成ウェイトのみアブレーション対象。

### 6.4 ショートの非対称制約

- サイズ上限 = ロングの50〜70%
- タイムストップ短縮
- CFD オーバーナイト金利・借株コストをコストモデルに明示計上（サクソ実勢値を costs.yaml に）

-----

## 7. バックテスト

- イベント駆動。シグナル確定 → **次バー始値**で約定
- コスト: 手数料＋スプレッド＋スリッページ 0.05〜0.1%＋CFDファイナンスコスト（日次）
- FinBERT スコアはバージョン固定・キャッシュ済みのものだけを参照（再現性）

-----

## 8. 検証プロトコル

### 8.1 目標レンジ（OOS/WFA後、コスト控除後）

|指標     |目標                     |
|-------|-----------------------|
|Sharpe |1.0〜1.5（IS>2.5 は過学習疑い） |
|Sortino|1.5〜2.2                |
|最大DD   |≤20%（ハード上限25%）         |
|勝率     |TF: 35〜45% / MR: 55〜65%|
|損益レシオ  |TF: ≥2.0 / MR: ≥0.9    |
|DSR    |>0（95%信頼水準）            |
|WF効率   |OOS/IS Sharpe ≥ 0.5    |

### 8.2 目的関数

Calmar比（CAGR/MaxDD）主軸。DD>20% でペナルティ、DSR≤0 で棄却。

### 8.3 過学習対策プロトコル（順序厳守）

1. チャート単独ベースラインを常に同一 WFA/CPCV 設定で併走
1. アブレーション追加順: ファンダ品質 → ニュースゲート → センチメント集計(HMM) → Trends。各段階で採否確定後に次へ
1. 採用条件: OOS で Calmar +15% 以上 かつ DD 悪化なし
1. trial_logger に GA 世代×個体・アブレーション実行・手動閾値調整を**すべて**記録し DSR の試行数補正に使用
1. ホールドアウト（直近1〜2年、全工程で未使用）は全アブレーション完了後に**1回のみ**評価

-----

## 9. 実装フェーズと受入基準

### Phase 1: 基盤＋データ層

- リポジトリ雛形、bitemporal_store、alpaca_ingest、pit_universe
- **受入基準**: (a) `as_of` 指定でリーク不能な取得APIが単体テストで保証される、(b) 廃止銘柄カバレッジ監査レポート（過去10年で対象セクターの主要な廃止・被買収銘柄リストに対する Alpaca の充足率）が出力される、(c) 年ごとの Alpaca News 記事件数監査が出力される
- 監査結果が不足を示す場合 → D6 に従い Polygon 補完を判断

### Phase 2: バックテストエンジン＋コストモデル

- **受入基準**: 既知の単純戦略（例: SPY 単純MAクロス）で手計算と一致する損益、CFD金利が日次で計上されること

### Phase 3: チャート単独ベースライン

- trend_follow / mean_revert / short_breakdown ＋ HMM（市場系のみ、ニュース抜き）＋ vol target ＋ WFA×CPCV×DSR
- **受入基準**: ベースラインの OOS 指標が §8.1 レンジに対して報告されること（達成できなくてもレポートは必須）

### Phase 4: alt-data 層（アブレーション順）

- **受入基準**: 各ファミリーの採否判断レポート（§8.3 条件との照合）

### Phase 5: 執行層（saxo_client）＋ペーパートレード

- 実弾投入前にペーパー/デモ口座で最低1か月の並走検証

-----

## 10. Claude Code での作業規約

- 各 Phase の受入基準を満たすまで次 Phase に進まない
- available_date 違反（先読み）を検出する単体テストを features/signals/risk の全モジュールに必須で付ける
- パラメータのハードコード禁止。config/*.yaml に集約
- 外部APIキーは .env のみ。コミット前に git-secrets 相当のチェック
