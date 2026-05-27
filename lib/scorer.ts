import { calcRSI, calcMACD, calcBollinger, calcEMA, calcOBV } from "./indicators";

function interp(value: number, bp: [number, number][]): number {
  if (value <= bp[0][0]) return bp[0][1];
  if (value >= bp[bp.length - 1][0]) return bp[bp.length - 1][1];
  for (let i = 0; i < bp.length - 1; i++) {
    const [x0, y0] = bp[i];
    const [x1, y1] = bp[i + 1];
    if (x0 <= value && value <= x1) {
      const t = (value - x0) / (x1 - x0);
      return y0 + t * (y1 - y0);
    }
  }
  return bp[bp.length - 1][1];
}

export interface ScoreResult {
  score: number;
  max: number;
  value: number | null;
  signal: string;
}

export function scoreRSI(closes: number[]): ScoreResult {
  const rsi = calcRSI(closes);
  const valid = rsi.filter((v) => !isNaN(v));
  if (valid.length < 2) return { score: 12, max: 25, value: null, signal: "insufficient_data" };

  const current = valid[valid.length - 1];
  const prev = valid[valid.length - 2];
  const rising = current > prev;

  let raw: number;
  let signal: string;

  if (current < 25) {
    raw = 23;
    signal = "extreme_oversold";
  } else if (current < 35) {
    const base = interp(current, [[25, 22], [35, 12]]);
    raw = Math.min(base + (rising ? 3 : 0), 25);
    signal = rising ? "oversold_recovery" : "oversold";
  } else if (current < 45) {
    raw = interp(current, [[35, 14], [45, 10]]);
    signal = "approaching_oversold";
  } else if (current < 55) {
    raw = interp(current, [[45, 12], [55, 8]]);
    signal = "neutral";
  } else if (current < 65) {
    raw = interp(current, [[55, 8], [65, 5]]);
    signal = "neutral_bullish";
  } else if (current < 75) {
    raw = interp(current, [[65, 5], [75, 2]]);
    signal = "overbought";
  } else {
    raw = interp(current, [[75, 2], [85, 0]]);
    signal = "extreme_overbought";
  }

  return { score: Math.round(raw), max: 25, value: Math.round(current * 100) / 100, signal };
}

export function scoreMACD(closes: number[]): ScoreResult {
  const { macdLine, signalLine, histogram } = calcMACD(closes);
  const validIdx = macdLine.map((v, i) => (!isNaN(v) ? i : -1)).filter((i) => i >= 0);
  if (validIdx.length < 3) return { score: 12, max: 25, value: null, signal: "insufficient_data" };

  const n = closes.length;
  const mCurr = macdLine[n - 1];
  const mPrev = macdLine[n - 2];
  const sCurr = signalLine[n - 1];
  const sPrev = signalLine[n - 2];
  const hCurr = histogram[n - 1];
  const hPrev = histogram[n - 2];

  const bullishCrossNow = mPrev < sPrev && mCurr >= sCurr;
  let bullishCrossRecent = false;
  for (let i = 2; i <= 3 && i < n; i++) {
    if (macdLine[n - i - 1] < signalLine[n - i - 1] && macdLine[n - i] >= signalLine[n - i] && mCurr > sCurr) {
      bullishCrossRecent = true;
      break;
    }
  }
  const bearishCrossNow = mPrev > sPrev && mCurr <= sCurr;

  let score: number;
  let signal: string;

  if (bullishCrossNow) { score = 24; signal = "bullish_crossover"; }
  else if (bullishCrossRecent && hCurr > 0) { score = 20; signal = "recent_bullish_crossover"; }
  else if (mCurr > sCurr && hCurr > hPrev) { score = 15; signal = "bullish_momentum"; }
  else if (mCurr > sCurr && hCurr <= hPrev) { score = 10; signal = "bullish_fading"; }
  else if (bearishCrossNow) { score = 2; signal = "bearish_crossover"; }
  else if (mCurr < sCurr && hCurr > hPrev) { score = 6; signal = "bearish_weakening"; }
  else { score = 3; signal = "bearish"; }

  return { score, max: 25, value: Math.round(mCurr * 10000) / 10000, signal };
}

export function scoreBollinger(closes: number[]): ScoreResult {
  const { upper, middle, lower, pctB } = calcBollinger(closes);
  const validPctB = pctB.filter((v) => !isNaN(v));
  if (validPctB.length < 2) return { score: 12, max: 25, value: null, signal: "insufficient_data" };

  const curr = validPctB[validPctB.length - 1];
  const prev = validPctB[validPctB.length - 2];
  const rising = curr > prev;

  // Squeeze bonus
  const bw = upper.map((u, i) => isNaN(u) ? NaN : (u - lower[i]) / middle[i]);
  const validBw = bw.filter((v) => !isNaN(v));
  let squeezeBonus = 0;
  if (validBw.length >= 5) {
    const bwNow = validBw[validBw.length - 1];
    const bwAvg = validBw.slice(-5).reduce((a, b) => a + b, 0) / 5;
    if (bwNow < bwAvg * 0.85 && curr < 0.3) squeezeBonus = 3;
  }

  // Direction bonus: %B rising = contrarian bounce (lower half) or momentum continuation (upper half)
  let dirBonus = 0;
  if (rising) dirBonus = curr < 0.5 ? 4 : curr <= 0.85 ? 3 : 1;

  let raw: number;
  let signal: string;

  if (curr < 0) {
    raw = interp(curr, [[-0.2, 18], [0, 15]]);
    signal = "below_lower_band";
  } else if (curr < 0.1) {
    raw = interp(curr, [[0, 15], [0.1, 12]]);
    signal = "near_lower_band";
  } else if (curr < 0.3) {
    raw = interp(curr, [[0.1, 12], [0.3, 9]]);
    signal = "lower_zone";
  } else if (curr < 0.5) {
    raw = interp(curr, [[0.3, 9], [0.5, 7]]);
    signal = "lower_half";
  } else if (curr < 0.7) {
    raw = interp(curr, [[0.5, 7], [0.7, 5]]);
    signal = rising ? "bullish_momentum" : "upper_half";
  } else if (curr < 0.9) {
    raw = interp(curr, [[0.7, 5], [0.9, 2]]);
    signal = rising ? "bullish_momentum" : "near_upper_band";
  } else {
    raw = interp(curr, [[0.9, 2], [1.2, 0]]);
    signal = "above_upper_band";
  }

  return { score: Math.min(Math.round(raw + dirBonus + squeezeBonus), 25), max: 25, value: Math.round(curr * 1000) / 1000, signal };
}

export function scoreMovingAverage(closes: number[]): ScoreResult {
  const ma20 = calcEMA(closes, 20);
  const ma50 = calcEMA(closes, 50);

  const n = closes.length;
  if (n < 51) return { score: 12, max: 25, value: null, signal: "insufficient_data" };

  const priceNow = closes[n - 1];
  const pricePrev = closes[n - 2];
  const ma20Now = ma20[n - 1];
  const ma20Prev = ma20[n - 2];
  const ma50Now = ma50[n - 1];
  const ma50Prev = ma50[n - 2];

  const crossAboveToday = pricePrev < ma20Prev && priceNow >= ma20Now;
  let crossAboveRecent = false;
  for (let i = 2; i <= 3 && i < n; i++) {
    if (closes[n - i - 1] < ma20[n - i - 1] && closes[n - i] >= ma20[n - i] && priceNow > ma20Now) {
      crossAboveRecent = true;
      break;
    }
  }

  let priceScore: number;
  let priceSignal: string;

  if (crossAboveToday) { priceScore = 15; priceSignal = "cross_above_ma20"; }
  else if (crossAboveRecent) { priceScore = 12; priceSignal = "recent_cross_above_ma20"; }
  else if (priceNow > ma20Now) {
    priceScore = Math.round(interp(priceNow / ma20Now, [[1.0, 9], [1.05, 6], [1.1, 4]]));
    priceSignal = "above_ma20";
  } else if (Math.abs(priceNow - ma20Now) / ma20Now < 0.005) {
    priceScore = 5; priceSignal = "at_ma20";
  } else {
    priceScore = Math.round(interp(priceNow / ma20Now, [[0.9, 4], [0.95, 3], [1.0, 0]]));
    priceSignal = "below_ma20";
  }

  const gap = ma20Now - ma50Now;
  const gapPrev = ma20Prev - ma50Prev;
  const widening = Math.abs(gap) > Math.abs(gapPrev);

  let maScore: number;
  if (gap > 0) { maScore = widening ? 9 : 6; }
  else if (Math.abs(gap / ma50Now) < 0.005) { maScore = 4; }
  else { maScore = widening ? 1 : 3; }

  return {
    score: Math.min(priceScore + maScore, 25),
    max: 25,
    value: Math.round((priceNow / ma20Now) * 10000) / 10000,
    signal: priceSignal,
  };
}

export interface AnalystTrend {
  strongBuy: number;
  buy: number;
  hold: number;
  sell: number;
  strongSell: number;
}

export function scoreAnalyst(trend: AnalystTrend | null): { score: number; signal: string; count: number } {
  if (!trend) return { score: 0, signal: "no_data", count: 0 };
  const { strongBuy, buy, hold, sell, strongSell } = trend;
  const total = strongBuy + buy + hold + sell + strongSell;
  if (total === 0) return { score: 0, signal: "no_data", count: 0 };

  // Weighted: strongBuy=2, buy=1, hold=0, sell=-1, strongSell=-2
  const weighted = strongBuy * 2 + buy - sell - strongSell * 2;
  const ratio = weighted / (total * 2); // [-1, 1]
  const score = Math.round(((ratio + 1) / 2) * 30); // [0, 30]

  let signal: string;
  if (score >= 24) signal = "strong_buy";
  else if (score >= 18) signal = "buy";
  else if (score >= 12) signal = "hold";
  else if (score >= 6) signal = "sell";
  else signal = "strong_sell";

  return { score, signal, count: total };
}

// ── Additional confirmation indicators (additive bonus, not in MULTS system) ──
// Backtest result: far_from_52w_high ΔSharpe +5.98, OBV_rising ΔSharpe +2.77
// These are scored separately and added as bonus points (max 12 + max 8 = 20).

// Distance below 52-week high: sweet spot is 5-15% below (room to recover, not broken)
export function score52wHigh(currentPrice: number, high52w: number | null): { score: number; dist_pct: number | null } {
  if (!high52w || high52w <= 0) return { score: 6, dist_pct: null };  // neutral if unknown
  const dist = (high52w - currentPrice) / high52w * 100;  // % below 52w high
  const score = Math.round(interp(dist, [
    [0,  9],   // near ATH — limited upside
    [5,  12],  // sweet spot start
    [15, 12],  // sweet spot end
    [25, 10],  // reasonable recovery potential
    [40, 7],   // large drawdown
    [60, 4],   // caution territory
    [80, 1],   // likely structural break
  ]));
  return { score, dist_pct: Math.round(dist * 10) / 10 };
}

// OBV trend: rising OBV = institutional accumulation = bullish confirmation
export function scoreOBV(closes: number[], volumes: number[]): { score: number; signal: string } {
  if (closes.length < 32 || volumes.length < 32) return { score: 4, signal: "insufficient_data" };
  const obv = calcOBV(closes, volumes);
  const n = obv.length;
  const valid = obv.filter((v) => !isNaN(v));
  if (valid.length < 11) return { score: 4, signal: "insufficient_data" };

  const obvNow    = obv[n - 1];
  const obvMinus10 = obv[n - 11];  // 10 bars ago
  const avgVol    = volumes.slice(n - 30, n).reduce((a, b) => a + b, 0) / 30;
  if (avgVol === 0) return { score: 4, signal: "no_volume" };

  const normalizedChange = (obvNow - obvMinus10) / avgVol;  // in units of avg daily volume

  let score: number;
  let signal: string;
  if (normalizedChange > 1.5)       { score = 8; signal = "strong_accumulation"; }
  else if (normalizedChange > 0.5)  { score = 6; signal = "accumulation"; }
  else if (normalizedChange > 0.0)  { score = 4; signal = "slight_accumulation"; }
  else if (normalizedChange > -0.5) { score = 3; signal = "neutral_obv"; }
  else                              { score = 1; signal = "distribution"; }

  return { score, signal };
}

export function computeScore(closes: number[], size: "large" | "mid" | "small" = "mid", extras?: { volumes?: number[]; high52w?: number | null }) {
  const rsi = scoreRSI(closes);
  const macd = scoreMACD(closes);
  const bollinger = scoreBollinger(closes);
  const movingAvg = scoreMovingAverage(closes);
  // Size-stratified weights from group-specific Monte Carlo walk-forward backtests.
  // Multipliers are Sharpe-proportional per group, sum=4.0 (preserves 0-100 scale).
  // Updated by run_backtest.py — see backtest/backtest_results.json for details.
  // Derived from size-stratified 10yr Monte Carlo walk-forward (thr=65, hold=5d):
  //   large: MACD(0.537)>MA(0.509)>BB(0.335)>RSI(0.304) → trend-following
  //   mid:   MA(0.498)>MACD(0.430)>BB(0.403)>RSI(0.355) → MA dominant
  //   small: RSI(0.565)>BB(0.521)>MA(0.520)>MACD(0.475) → mean-reversion
  const MULTS: Record<"large" | "mid" | "small", [number, number, number, number]> = {
    //          [RSI,   MACD,   BB,    MA  ]
    large: [0.722, 1.274, 0.795, 1.209],  // MACD/MA dominate (persistent trends)
    mid:   [0.842, 1.021, 0.956, 1.182],  // MA dominant, balanced otherwise
    small: [1.086, 0.913, 1.001, 1.000],  // RSI/BB effective (high-vol mean-reversion)
  };
  const [mRsi, mMacd, mBb, mMa] = MULTS[size];
  const base = Math.round(
    rsi.score       * mRsi  +
    macd.score      * mMacd +
    bollinger.score * mBb   +
    movingAvg.score * mMa
  );

  // Additive confirmation bonuses: only large cap benefits (backtest: large +27% Sharpe;
  // mid/small are hurt — the 4-indicator base system already captures their dynamics).
  const highResult = size === "large" ? score52wHigh(closes[closes.length - 1], extras?.high52w ?? null) : { score: 0, dist_pct: null as null };
  const obvResult  = size === "large" ? scoreOBV(closes, extras?.volumes ?? []) : { score: 0, signal: "n/a" };

  const total = Math.min(100, base + highResult.score + obvResult.score);
  return { total, rsi, macd, bollinger, moving_avg: movingAvg, high52w: highResult, obv: obvResult };
}

// ── Day trade scoring functions ───────────────────────────────────────────────

// V-shaped RSI: rewards both oversold bounces AND upward momentum (RSI 55-70)
export function scoreRSIDay(closes: number[]): ScoreResult {
  const rsi = calcRSI(closes);
  const valid = rsi.filter((v) => !isNaN(v));
  if (valid.length < 2) return { score: 12, max: 25, value: null, signal: "insufficient_data" };

  const current = valid[valid.length - 1];
  const prev = valid[valid.length - 2];
  const rising = current > prev;

  let raw: number;
  let signal: string;

  if (current < 25) {
    raw = rising ? 22 : 17;
    signal = rising ? "oversold_bounce" : "extreme_oversold";
  } else if (current < 40) {
    const base = interp(current, [[25, 19], [40, 12]]);
    raw = base + (rising ? 2 : 0);
    signal = rising ? "recovering" : "oversold";
  } else if (current < 50) {
    raw = interp(current, [[40, 12], [50, 9]]);
    signal = "neutral_low";
  } else if (current < 55) {
    raw = interp(current, [[50, 9], [55, 11]]);
    signal = "neutral";
  } else if (current < 65) {
    raw = Math.max(interp(current, [[55, 11], [65, 17]]) + (rising ? 2 : -2), 8);
    signal = rising ? "momentum_up" : "momentum_fading";
  } else if (current < 75) {
    raw = interp(current, [[65, 15], [75, 10]]) + (rising ? 1 : 0);
    signal = (current > 70 && rising) ? "overbought_momentum" : "strong_momentum";
  } else {
    raw = interp(current, [[75, 8], [85, 3]]);
    signal = "extreme_overbought";
  }

  return { score: Math.round(Math.min(raw, 25)), max: 25, value: Math.round(current * 100) / 100, signal };
}

// EMA5/EMA10 crossovers for intraday sensitivity (vs EMA20/50 for swing)
export function scoreMADay(closes: number[]): ScoreResult {
  const ma5  = calcEMA(closes, 5);
  const ma10 = calcEMA(closes, 10);

  const n = closes.length;
  if (n < 11) return { score: 12, max: 25, value: null, signal: "insufficient_data" };

  const priceNow  = closes[n - 1];
  const pricePrev = closes[n - 2];
  const ma5Now    = ma5[n - 1];
  const ma5Prev   = ma5[n - 2];
  const ma10Now   = ma10[n - 1];
  const ma10Prev  = ma10[n - 2];

  const crossAboveToday = pricePrev < ma5Prev && priceNow >= ma5Now;
  let crossAboveRecent = false;
  if (n >= 4 && closes[n - 3] < ma5[n - 3] && closes[n - 2] >= ma5[n - 2] && priceNow > ma5Now) {
    crossAboveRecent = true;
  }

  let priceScore: number;
  let priceSignal: string;

  if (crossAboveToday) { priceScore = 15; priceSignal = "cross_above_ma5"; }
  else if (crossAboveRecent) { priceScore = 13; priceSignal = "recent_cross_above_ma5"; }
  else if (priceNow > ma5Now) {
    priceScore = Math.round(interp(priceNow / ma5Now, [[1.0, 10], [1.03, 7], [1.06, 4]]));
    priceSignal = "above_ma5";
  } else if (Math.abs(priceNow - ma5Now) / ma5Now < 0.003) {
    priceScore = 6; priceSignal = "at_ma5";
  } else {
    priceScore = Math.round(interp(priceNow / ma5Now, [[0.94, 4], [0.97, 3], [1.0, 0]]));
    priceSignal = "below_ma5";
  }

  const gap      = ma5Now  - ma10Now;
  const gapPrev  = ma5Prev - ma10Prev;
  const widening = Math.abs(gap) > Math.abs(gapPrev);

  let maScore: number;
  if (gap > 0)                              { maScore = widening ? 9 : 6; }
  else if (Math.abs(gap / ma10Now) < 0.003) { maScore = 4; }
  else                                       { maScore = widening ? 1 : 3; }

  return {
    score: Math.min(priceScore + maScore, 25),
    max: 25,
    value: Math.round((priceNow / ma5Now) * 10000) / 10000,
    signal: priceSignal,
  };
}

export function computeScoreDay(closes: number[], size: "large" | "mid" | "small" = "mid", extras?: { volumes?: number[]; high52w?: number | null }) {
  const rsi       = scoreRSIDay(closes);
  const macd      = scoreMACD(closes);
  const bollinger = scoreBollinger(closes);
  const movingAvg = scoreMADay(closes);
  // Day trade multipliers from hold=1d size-stratified Monte Carlo walk-forward backtest.
  // Updated by run_backtest.py (day mode) — see backtest/backtest_results.json.
  // Derived from 50-stock hold=1d MC walk-forward (thr=65):
  //   large: MA(0.603)>MACD(0.529)>RSI(0.422)>BB(0.355) → MA/MACD dominant
  //   mid:   MA(0.480)>MACD(0.442)>RSI(0.391)>BB(0.307) → MA/MACD dominant
  //   small: MA(0.597)>MACD(0.548)>BB(0.524)>RSI(0.492) → more balanced
  const MULTS_DAY: Record<"large" | "mid" | "small", [number, number, number, number]> = {
    //          [RSI,   MACD,   BB,    MA  ]
    large: [0.885, 1.109, 0.743, 1.264],  // MA strongest, BB weakest
    mid:   [0.966, 1.092, 0.757, 1.185],  // MA/MACD lead, BB weak
    small: [0.911, 1.015, 0.969, 1.105],  // more balanced, MA still leads
  };
  const [mRsi, mMacd, mBb, mMa] = MULTS_DAY[size];
  const base = Math.round(
    rsi.score       * mRsi  +
    macd.score      * mMacd +
    bollinger.score * mBb   +
    movingAvg.score * mMa
  );
  const highResult = size === "large" ? score52wHigh(closes[closes.length - 1], extras?.high52w ?? null) : { score: 0, dist_pct: null as null };
  const obvResult  = size === "large" ? scoreOBV(closes, extras?.volumes ?? []) : { score: 0, signal: "n/a" };
  const total = Math.min(100, base + highResult.score + obvResult.score);
  return { total, rsi, macd, bollinger, moving_avg: movingAvg, high52w: highResult, obv: obvResult };
}

export function getPeLabel(pe: number | null | undefined, market: string): string | null {
  if (!pe || pe <= 0 || pe > 5000) return null;
  if (market === "JP") return pe < 15 ? "割安" : pe <= 25 ? "適正" : "割高";
  return pe < 20 ? "割安" : pe <= 35 ? "適正" : "割高";
}
