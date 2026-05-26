import { calcRSI, calcMACD, calcBollinger, calcEMA } from "./indicators";

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

  // Squeeze bonus
  const bw = upper.map((u, i) => isNaN(u) ? NaN : (u - lower[i]) / middle[i]);
  const validBw = bw.filter((v) => !isNaN(v));
  let squeezeBonus = 0;
  if (validBw.length >= 5) {
    const bwNow = validBw[validBw.length - 1];
    const bwAvg = validBw.slice(-5).reduce((a, b) => a + b, 0) / 5;
    if (bwNow < bwAvg * 0.85 && curr < 0.3) squeezeBonus = 3;
  }

  let raw: number;
  let signal: string;

  if (curr < 0) { raw = interp(curr, [[-0.2, 25], [0, 21]]); signal = "below_lower_band"; }
  else if (curr < 0.1) { raw = interp(curr, [[0, 21], [0.1, 17]]); signal = "near_lower_band"; }
  else if (curr < 0.25) { raw = interp(curr, [[0.1, 17], [0.25, 12]]); signal = "lower_zone"; }
  else if (curr < 0.5) { raw = interp(curr, [[0.25, 12], [0.5, 8]]); signal = "lower_half"; }
  else if (curr < 0.75) { raw = interp(curr, [[0.5, 8], [0.75, 4]]); signal = "upper_half"; }
  else if (curr < 0.9) { raw = interp(curr, [[0.75, 4], [0.9, 1]]); signal = "near_upper_band"; }
  else { raw = interp(curr, [[0.9, 1], [1.2, 0]]); signal = "above_upper_band"; }

  return { score: Math.min(Math.round(raw + squeezeBonus), 25), max: 25, value: Math.round(curr * 1000) / 1000, signal };
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

export function computeScore(closes: number[]) {
  const rsi = scoreRSI(closes);
  const macd = scoreMACD(closes);
  const bollinger = scoreBollinger(closes);
  const movingAvg = scoreMovingAverage(closes);
  return {
    total: rsi.score + macd.score + bollinger.score + movingAvg.score,
    rsi,
    macd,
    bollinger,
    moving_avg: movingAvg,
  };
}

export function getPeLabel(pe: number | null | undefined, market: string): string | null {
  if (!pe || pe <= 0 || pe > 5000) return null;
  if (market === "JP") return pe < 15 ? "割安" : pe <= 25 ? "適正" : "割高";
  return pe < 20 ? "割安" : pe <= 35 ? "適正" : "割高";
}
