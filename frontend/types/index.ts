export interface ScoreComponent {
  score: number;
  max: number;
  value: number | null;
  signal: string;
}

export interface StockScore {
  ticker: string;
  market: "JP" | "US";
  name: string;
  price: number | null;
  currency: string;
  change_pct: number | null;
  score: number;
  score_components: {
    rsi: ScoreComponent;
    macd: ScoreComponent;
    bollinger: ScoreComponent;
    moving_avg: ScoreComponent;
  };
  trailing_pe: number | null;
  forward_pe: number | null;
  pe_label: string | null;
  last_updated: string;
}

export interface StockScoresResponse {
  results: StockScore[];
  errors: Record<string, string>;
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
