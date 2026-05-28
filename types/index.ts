export interface ScoreComponent {
  score: number;
  max: number;
  value: number | null;
  signal: string;
}

export type MacroRegime = "bullish" | "neutral" | "cautious" | "bearish";

export interface MacroSignal {
  regime: MacroRegime;
  smh_5d_return: number;  // fraction, e.g. 0.023
  multiplier: number;
  label: string;          // human-readable, e.g. "SMH 5日: +2.3%"
}

export interface StockScore {
  ticker: string;
  market: "JP" | "US";
  name: string;
  price: number | null;
  currency: string;
  change_pct: number | null;
  score: number;
  score_raw: number;  // before macro adjustment
  score_components: {
    rsi: ScoreComponent;
    macd: ScoreComponent;
    bollinger: ScoreComponent;
    moving_avg: ScoreComponent;
    analyst: ScoreComponent;
    high52w: { score: number; dist_pct: number | null };
    obv:    { score: number; signal: string };
    aroon:  { score: number; signal: string };
  };
  trailing_pe: number | null;
  forward_pe: number | null;
  pe_label: string | null;
  analyst_score: number | null;
  analyst_signal: string | null;
  analyst_count: number;
  sector: string;
  sector_label: string;
  size: "large" | "mid" | "small";
  last_updated: string;
}

export interface SellSignals {
  bb_target_2sigma: number | null;    // upper BB 2σ — primary sell target (backtest #1)
  bb_target_2_5sigma: number | null;  // upper BB 2.5σ — secondary target
  atr_14d: number | null;             // current 14-day ATR (stop sizing reference)
  stop_loss_5pct: number | null;      // current_price × 0.95 (backtest-derived stop)
  dist_from_52w_high_pct: number | null; // % below 52-week high
  high_52w: number | null;
}

export interface StockScoresResponse {
  results: StockScore[];
  errors: Record<string, string>;
  macro: MacroSignal;
}

export interface OHLCVPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SeriesPoint {
  date: string;
  value: number;
}

export interface MACDPoint {
  date: string;
  macd: number;
  signal: number;
  histogram: number;
}

export interface BBPoint {
  date: string;
  upper: number;
  middle: number;
  lower: number;
  pct_b: number;
}

export interface StockDetail extends StockScore {
  macro: MacroSignal;
  sell_signals: SellSignals;
  history: OHLCVPoint[];
  rsi_series: SeriesPoint[];
  macd_series: MACDPoint[];
  bb_series: BBPoint[];
  ma20_series: SeriesPoint[];
  ma50_series: SeriesPoint[];
}

export interface WatchlistResponse {
  tickers: string[];
}

export interface ValidateResponse {
  valid: boolean;
  name?: string;
  market?: string;
  error?: string;
}

