"""
config.py — 全設定・銘柄リスト・パス定義
"""
from __future__ import annotations
from pathlib import Path

HERE = Path(__file__).parent

# ── パス ──────────────────────────────────────────────────────────────────────
PRICE_DATA_SWING    = HERE / "price_data.json"           # 日足 10年 (swing用)
PRICE_DATA_DAY      = HERE / "price_data_intraday.json"  # 10min足 2年 (day用)
MACRO_DATA_PATH     = HERE / "macro_data.json"           # マクロ/レジーム用ETF日足
REGIME_SIGNALS_PATH = HERE / "regime_signals.json"       # HMM出力: {ts, label, prob}
RESULTS_SWING_PATH  = HERE / "results_swing.json"
RESULTS_DAY_PATH    = HERE / "results_day.json"
ARTIFACTS_DIR       = HERE / "phase_artifacts"

# ── 取引コスト (往復) ─────────────────────────────────────────────────────────
US_COST = 0.0016   # 米国株: 0.16%
JP_COST = 0.0030   # 日本株: 0.30%

# ── 銘柄リスト ────────────────────────────────────────────────────────────────
# 半導体・製造装置 (20)
TICKERS_SEMIS = [
    "NVDA", "AMD",  "AVGO", "QCOM", "MU",
    "ARM",  "AMAT", "LRCX", "KLAC", "MRVL",
    "ASML", "TSM",  "INTC", "SMCI", "MCHP",
    "SWKS", "MPWR", "ENTG", "ONTO", "TXN",
]
# AI / クラウド / ソフトウェア (15)
TICKERS_AI_CLOUD = [
    "MSFT", "GOOGL", "META",  "AAPL", "AMZN",
    "PLTR", "CRM",   "DDOG",  "CRWD", "SNOW",
    "SOUN", "IONQ",  "NOW",   "PANW", "ORCL",
]
# 宇宙・防衛テック (8)
TICKERS_SPACE = [
    "RKLB", "LUNR", "KTOS", "HII",
    "BA",   "NOC",  "LMT",  "RTX",
]
# 核融合・クリーンエネルギー (5)
TICKERS_NUCLEAR = [
    "CEG", "VST", "GEV", "OKLO", "SMR",
]
# メモリ・ストレージ (2)
TICKERS_MEMORY = ["WDC", "STX"]

# 全銘柄 (50)
TICKERS_ALL: list[str] = (
    TICKERS_SEMIS + TICKERS_AI_CLOUD + TICKERS_SPACE
    + TICKERS_NUCLEAR + TICKERS_MEMORY
)

# デイトレ・スイング共通（同リストを使用）
TICKERS_DAY   = TICKERS_ALL
TICKERS_SWING = TICKERS_ALL

# マクロ/レジーム検出用 ETF・指数
TICKERS_MACRO = [
    "^VIX",   # VIX 恐怖指数
    "HYG",    # ハイイールド社債 ETF
    "LQD",    # 投資適格社債 ETF
    "SOXX",   # 半導体 ETF (iShares)
    "SMH",    # 半導体 ETF (VanEck)
    "SPY",    # S&P500 ETF (ベンチマーク)
]

# ── データ取得設定 ────────────────────────────────────────────────────────────
SWING_LOOKBACK_YEARS = 10       # スイング: 日足 10年
DAY_LOOKBACK_YEARS   = 2        # デイトレ: 10min足 2年
MACRO_LOOKBACK_YEARS = 12       # マクロ:  日足 12年 (HMM学習バッファ込み)

SWING_TIMEFRAME = "1Day"        # Alpaca timeframe
DAY_TIMEFRAME   = "10Min"

# ── HMM 設定 ─────────────────────────────────────────────────────────────────
HMM_MIN_STATES  = 2
HMM_MAX_STATES  = 6
HMM_N_ITER      = 500
HMM_RANDOM_SEED = 42

# ── GA 設定 ──────────────────────────────────────────────────────────────────
# Swing (日足10年 × 50社: 大規模データ)
GA_POP_SWING    = 1000
GA_GENS_SWING   = 300
GA_ELITE_SWING  = 30
# Day (10min 2年 × 50社: 中規模)
GA_POP_DAY      = 300
GA_GENS_DAY     = 100
GA_ELITE_DAY    = 15
# 共通
GA_TOURN_SIZE   = 7
GA_MUT_SIGMA    = 0.10
GA_MUT_PROB     = 0.25
GA_L2_LAMBDA    = 0.5   # L2正則化: 指標への過度な集中を抑制

# ── Sharpe 計算 ───────────────────────────────────────────────────────────────
BARS_PER_YEAR_SWING = 252       # 日足: 年間252営業日
BARS_PER_YEAR_DAY   = 9828      # 10min: 252日 × 39本/日

MIN_TRADES        = 30          # in-sample 最低取引数
MIN_TRADES_OOS    = 30          # OOS 最低取引数
SWING_MIN_TRADES  = 30          # swing OOS 最低取引数

# ── Walk-Forward 設定 ────────────────────────────────────────────────────────
WF_N_FOLDS      = 4
WF_TRAIN_RATIO  = 0.80          # 80% 訓練, 20% テスト (最終ホールドアウト)

# ── 取引コスト・最大保有期間 ──────────────────────────────────────────────────
COST_RATE_SWING = 0.0016        # 往復コスト (米国株スイング)
COST_RATE_DAY   = 0.0008        # 往復コスト (デイトレ: ECN割引考慮)
SWING_MAX_HOLD  = 60            # スイング最大保有バー数 (日足 60日)
DAY_MAX_HOLD    = 8             # デイトレ最大保有バー数 (10min足 8本 = 80分)

# ── リスク管理 ────────────────────────────────────────────────────────────────
RISK_TARGET_VOL       = 0.15    # 年率ボラティリティ目標 15%
RISK_MAX_POSITION     = 0.20    # 1銘柄上限 20%
RISK_STOP_IN_REGIME   = {2, 3}  # HMMレジーム: これらの状態では取引停止 (高ボラ)

# ── スイング取引設定 ──────────────────────────────────────────────────────────
# False: 毎バーでスコア>=閾値なら買い (機会最大化)
# True : スコアが閾値を「下から越えた」初日のみ買い (3.3取引/銘柄/年に制限)
SWING_CROSSOVER_ONLY  = False
