// Technical indicator calculations (pure TypeScript, no external deps)

export function calcRSI(closes: number[], period = 14): number[] {
  const result: number[] = new Array(closes.length).fill(NaN);
  if (closes.length < period + 1) return result;

  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff > 0) avgGain += diff;
    else avgLoss += -diff;
  }
  avgGain /= period;
  avgLoss /= period;

  for (let i = period; i < closes.length; i++) {
    if (i > period) {
      const diff = closes[i] - closes[i - 1];
      const gain = diff > 0 ? diff : 0;
      const loss = diff < 0 ? -diff : 0;
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
    }
    const rs = avgLoss === 0 ? Infinity : avgGain / avgLoss;
    result[i] = 100 - 100 / (1 + rs);
  }
  return result;
}

export function calcEMA(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const result: number[] = new Array(values.length).fill(NaN);
  let ema = values[0];
  result[0] = ema;
  for (let i = 1; i < values.length; i++) {
    ema = values[i] * k + ema * (1 - k);
    result[i] = ema;
  }
  return result;
}

export function calcMACD(closes: number[], fast = 12, slow = 26, signal = 9) {
  const emaFast = calcEMA(closes, fast);
  const emaSlow = calcEMA(closes, slow);
  const macdLine = closes.map((_, i) =>
    isNaN(emaFast[i]) || isNaN(emaSlow[i]) ? NaN : emaFast[i] - emaSlow[i]
  );
  const validMacd = macdLine.filter((v) => !isNaN(v));
  const signalTemp = calcEMA(validMacd, signal);
  const signalLine: number[] = new Array(closes.length).fill(NaN);
  let si = 0;
  for (let i = 0; i < closes.length; i++) {
    if (!isNaN(macdLine[i])) {
      signalLine[i] = signalTemp[si++] ?? NaN;
    }
  }
  const histogram = closes.map((_, i) =>
    isNaN(macdLine[i]) || isNaN(signalLine[i]) ? NaN : macdLine[i] - signalLine[i]
  );
  return { macdLine, signalLine, histogram };
}

export function calcBollinger(closes: number[], period = 20, stdDev = 2) {
  const upper: number[] = new Array(closes.length).fill(NaN);
  const middle: number[] = new Array(closes.length).fill(NaN);
  const lower: number[] = new Array(closes.length).fill(NaN);
  const pctB: number[] = new Array(closes.length).fill(NaN);

  for (let i = period - 1; i < closes.length; i++) {
    const slice = closes.slice(i - period + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / period;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / period;
    const std = Math.sqrt(variance);
    middle[i] = mean;
    upper[i] = mean + stdDev * std;
    lower[i] = mean - stdDev * std;
    const bw = upper[i] - lower[i];
    pctB[i] = bw === 0 ? 0.5 : (closes[i] - lower[i]) / bw;
  }
  return { upper, middle, lower, pctB };
}

// 14-day ATR (Average True Range) for position sizing / stop-loss estimation
export function calcATR(highs: number[], lows: number[], closes: number[], period = 14): number[] {
  const n = closes.length;
  const atr: number[] = new Array(n).fill(NaN);
  if (n < period + 1) return atr;

  const trueRanges: number[] = [NaN];
  for (let i = 1; i < n; i++) {
    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1])
    );
    trueRanges.push(tr);
  }

  // Wilder's smoothing
  let sum = 0;
  for (let i = 1; i <= period; i++) sum += trueRanges[i];
  atr[period] = sum / period;
  for (let i = period + 1; i < n; i++) {
    atr[i] = (atr[i - 1] * (period - 1) + trueRanges[i]) / period;
  }
  return atr;
}

// Aroon (25-period): measures how recently the highest high / lowest low occurred.
// AroonUp=100 → new 25-bar high just made; AroonDown=100 → new 25-bar low just made.
export function calcAroon(highs: number[], lows: number[], period = 25): { up: number[]; down: number[] } {
  const n = highs.length;
  const up   = new Array(n).fill(NaN);
  const down = new Array(n).fill(NaN);
  for (let i = period; i < n; i++) {
    let maxH = -Infinity, barsFromHigh = 0;
    let minL =  Infinity, barsFromLow  = 0;
    for (let j = 0; j <= period; j++) {
      if (highs[i - j] > maxH) { maxH = highs[i - j]; barsFromHigh = j; }
      if (lows[i  - j] < minL) { minL = lows[i  - j]; barsFromLow  = j; }
    }
    up[i]   = ((period - barsFromHigh) / period) * 100;
    down[i] = ((period - barsFromLow)  / period) * 100;
  }
  return { up, down };
}

export function calcStoch(
  closes: number[], highs: number[], lows: number[],
  kPeriod = 14, dPeriod = 3
): { k: number[]; d: number[] } {
  const n = closes.length;
  const k: number[] = new Array(n).fill(NaN);
  for (let i = kPeriod - 1; i < n; i++) {
    let hi = -Infinity, lo = Infinity;
    for (let j = i - kPeriod + 1; j <= i; j++) {
      if (highs[j] > hi) hi = highs[j];
      if (lows[j] < lo) lo = lows[j];
    }
    k[i] = (hi - lo) > 1e-10 ? ((closes[i] - lo) / (hi - lo)) * 100 : 50;
  }
  const d: number[] = new Array(n).fill(NaN);
  for (let i = kPeriod + dPeriod - 2; i < n; i++) {
    let sum = 0, cnt = 0;
    for (let j = i - dPeriod + 1; j <= i; j++) {
      if (!isNaN(k[j])) { sum += k[j]; cnt++; }
    }
    if (cnt === dPeriod) d[i] = sum / cnt;
  }
  return { k, d };
}

export function calcCCI(closes: number[], highs: number[], lows: number[], period = 20): number[] {
  const n = closes.length;
  const out: number[] = new Array(n).fill(NaN);
  for (let i = period - 1; i < n; i++) {
    const tps: number[] = [];
    for (let j = i - period + 1; j <= i; j++) tps.push((closes[j] + highs[j] + lows[j]) / 3);
    const mean = tps.reduce((a, b) => a + b, 0) / period;
    const md = tps.reduce((a, b) => a + Math.abs(b - mean), 0) / period;
    out[i] = md > 1e-10 ? (tps[period - 1] - mean) / (0.015 * md) : 0;
  }
  return out;
}

export function calcROC(closes: number[], period = 10): number[] {
  const n = closes.length;
  const out: number[] = new Array(n).fill(NaN);
  for (let i = period; i < n; i++) {
    if (closes[i - period] > 0) out[i] = (closes[i] / closes[i - period] - 1) * 100;
  }
  return out;
}

// OBV (On-Balance Volume) — cumulative signed volume
export function calcOBV(closes: number[], volumes: number[]): number[] {
  const n = closes.length;
  const obv: number[] = new Array(n).fill(NaN);
  if (n < 2) return obv;

  obv[0] = 0;
  for (let i = 1; i < n; i++) {
    const sign = closes[i] > closes[i - 1] ? 1 : closes[i] < closes[i - 1] ? -1 : 0;
    obv[i] = obv[i - 1] + sign * volumes[i];
  }
  return obv;
}
