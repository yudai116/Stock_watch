# データソース仕様

## 米国株（本番運用中）

### データソース
**Polygon.io Developer プラン**（$79/月）  
APIキー: GitHub Secret `POLYGON_API_KEY`  
取得スクリプト: `backtest/data/fetch_polygon.py`

### SWING モード（1時間足）

| 項目 | 値 |
|---|---|
| 対象銘柄数 | 50銘柄 |
| 足種 | 1時間足 (1Hour) |
| 学習期間 | 10年 (`SWING_LOOKBACK_YEARS = 10`) |
| データ開始 | 約2016年6月〜（Polygon Developer上限） |
| 保存ファイル | `backtest/price_data.json` |
| WF構成 | 4フォールド / 80:20 train:test |

### DAY モード（10分足）

| 項目 | 値 |
|---|---|
| 対象銘柄数 | 50銘柄 |
| 足種 | 10分足 (10Min) |
| 学習期間 | 3年 (`DAY_LOOKBACK_YEARS = 3`) |
| データ開始 | 約2023年〜 |
| 保存ファイル | `backtest/price_data_intraday.json` |
| WF構成 | 4フォールド / 80:20 train:test |

### 対象50銘柄

**半導体・製造装置 (20銘柄)**
NVDA, AMD, AVGO, QCOM, MU, ARM, AMAT, LRCX, KLAC, MRVL,
ASML, TSM, INTC, SMCI, MCHP, SWKS, MPWR, ENTG, ONTO, TXN

**AI / クラウド / ソフトウェア (15銘柄)**
MSFT, GOOGL, META, AAPL, AMZN, PLTR, CRM, DDOG, CRWD, SNOW,
SOUN, IONQ, NOW, PANW, ORCL

**宇宙・防衛テック (8銘柄)**
RKLB, LUNR, KTOS, HII, BA, NOC, LMT, RTX

**核融合・クリーンエネルギー (5銘柄)**
CEG, VST, GEV, OKLO, SMR

**メモリ・ストレージ (2銘柄)**
WDC, STX

> **注意**: RKLB/LUNR/OKLO/SMR/IONQ/SOUN 等の新興株は上場が2020〜2024年のため
> 歴史データが短い（3〜6年）。WF評価時にフォールド数が減少する可能性あり。

---

## マクロデータ（HMMレジーム検出用）

| 項目 | 値 |
|---|---|
| データソース | yfinance（Yahoo Finance） |
| 取得スクリプト | `backtest/data/fetch_macro.py` |
| 銘柄 | ^VIX, HYG, LQD, SOXX, SMH, SPY |
| 足種 | 日足 |
| 学習期間 | 12年 (`MACRO_LOOKBACK_YEARS = 12`) |
| 保存ファイル | `backtest/macro_data.json` |

---

## 日本株（未実装・検討中）

### 現状と課題

日本株の10分足データソースとして **J-Quants API**（日本取引所グループ提供）が存在する。

| 項目 | 内容 |
|---|---|
| データソース | J-Quants API v2 (`https://jpx-jquants.com`) |
| 足種 | 1分足（→10分足に集約可能） |
| 取得可能期間 | **最大2年分**（アドオン契約必要） |
| 無料プラン | 分足データなし（日足のみ） |

### 制約

- 現行の `DAY_LOOKBACK_YEARS = 3` に対して **J-Quantsは2年分のみ**
  → 3年分には不足（WF 4フォールドのうち有効フォールドが減少）
- 分足アドオンの費用については [J-Quants 料金ページ](https://jpx-jquants.com) を参照
- 東証上場銘柄のみ対応（大証・名証の一部銘柄は対象外）

### 実装するには

1. J-Quantsアカウント登録 + 分足アドオン契約
2. `backtest/data/fetch_jquants.py` を新規作成
3. `backtest/config.py` に `TICKERS_JP` リストと `PRICE_DATA_JP` パスを追加
4. `DAY_LOOKBACK_YEARS_JP = 2` を別定数として定義
5. `pipeline.py` に日本株フェッチステップを追加

日本株を追加する場合は J-Quantsアカウントが必要です。

---

## ワークフロー（GitHub Actions）

データ取得は GitHub Actions で自動実行されます。

```
.github/workflows/backtest.yml
```

| 入力パラメータ | 説明 |
|---|---|
| `mode` | all / swing / day |
| `skip_fetch` | true でデータ取得をスキップ（キャッシュ使用） |
| `skip_regime` | true でHMM学習をスキップ |
| `skip_strategy` | true でGA最適化をスキップ（データ取得のみ） |
