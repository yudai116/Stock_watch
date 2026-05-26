// yahoo-finance2 v3 requires instantiation
// eslint-disable-next-line @typescript-eslint/no-require-imports
const YFClass = require("yahoo-finance2").default;
const yf = new YFClass() as InstanceType<typeof YFClass>;
import { computeScore, getPeLabel, scoreAnalyst } from "./scorer";
import { calcRSI, calcMACD, calcBollinger, calcEMA } from "./indicators";
import { getSector, DEFAULT_SECTOR_WATCHLIST } from "./sectors";
import type { StockScore, StockDetail, OHLCVPoint, SeriesPoint, MACDPoint, BBPoint } from "@/types";

export const DEFAULT_WATCHLIST = DEFAULT_SECTOR_WATCHLIST;

function detectMarket(ticker: string) {
  return ticker.toUpperCase().endsWith(".T") ? "JP" : "US";
}

function getCurrency(market: string) {
  return market === "JP" ? "JPY" : "USD";
}

type HistoricalRow = { date: Date; open: number; high: number; low: number; close: number; volume: number };

async function fetchHistory(ticker: string): Promise<HistoricalRow[]> {
  const endDate = new Date();
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - 120);

  const rows = await yf.historical(ticker, {
    period1: startDate.toISOString().split("T")[0],
    period2: endDate.toISOString().split("T")[0],
    interval: "1d",
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (rows as any[])
    .filter((r) => r.close != null)
    .map((r) => ({
      date: r.date as Date,
      open: (r.open ?? r.close) as number,
      high: (r.high ?? r.close) as number,
      low: (r.low ?? r.close) as number,
      close: r.close as number,
      volume: (r.volume ?? 0) as number,
    }));
}

async function fetchQuote(ticker: string) {
  try {
    const q = await yf.quoteSummary(ticker, {
      modules: ["price", "summaryDetail", "recommendationTrend"],
    });
    return q;
  } catch {
    return null;
  }
}

export async function buildStockScore(ticker: string): Promise<StockScore> {
  const upper = ticker.toUpperCase();
  const market = detectMarket(upper);
  const currency = getCurrency(market);
  const sectorInfo = getSector(upper);

  const [rows, quote] = await Promise.all([fetchHistory(upper), fetchQuote(upper)]);
  if (rows.length < 30) throw new Error(`Not enough data for ${upper}`);

  const closes = rows.map((r) => r.close);
  const technicalScores = computeScore(closes);
  const technicalTotal = technicalScores.total; // 0-100

  const price = closes[closes.length - 1];
  const prevClose = closes[closes.length - 2];
  const changePct = prevClose ? Math.round(((price - prevClose) / prevClose) * 10000) / 100 : null;

  const name = quote?.price?.longName ?? quote?.price?.shortName ?? upper;
  const trailingPe = (quote?.summaryDetail?.trailingPE as number | undefined) ?? null;
  const forwardPe = (quote?.summaryDetail?.forwardPE as number | undefined) ?? null;
  const cleanPe = (pe: number | null) => (pe && pe > 0 && pe < 5000 ? Math.round(pe * 100) / 100 : null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const trendData = (quote as any)?.recommendationTrend?.trend?.[0] ?? null;
  const analystResult = scoreAnalyst(trendData);
  const hasAnalyst = analystResult.count > 0;

  // Blend: 70% technical + 30% analyst (when analyst data available)
  const blendedScore = hasAnalyst
    ? Math.min(100, Math.round(technicalTotal * 0.7 + analystResult.score))
    : technicalTotal;

  return {
    ticker: upper,
    market: market as "JP" | "US",
    name: name ?? upper,
    price: Math.round(price * 100) / 100,
    currency,
    change_pct: changePct,
    score: blendedScore,
    score_components: {
      rsi: technicalScores.rsi,
      macd: technicalScores.macd,
      bollinger: technicalScores.bollinger,
      moving_avg: technicalScores.moving_avg,
      analyst: {
        score: hasAnalyst ? analystResult.score : 0,
        max: 30,
        value: null,
        signal: analystResult.signal,
      },
    },
    trailing_pe: cleanPe(trailingPe),
    forward_pe: cleanPe(forwardPe),
    pe_label: getPeLabel(trailingPe, market),
    analyst_score: hasAnalyst ? analystResult.score : null,
    analyst_signal: analystResult.signal,
    analyst_count: analystResult.count,
    sector: sectorInfo.key,
    sector_label: sectorInfo.label,
    last_updated: new Date().toISOString(),
  };
}

export async function buildStockDetail(ticker: string): Promise<StockDetail> {
  const upper = ticker.toUpperCase();
  const market = detectMarket(upper);
  const currency = getCurrency(market);
  const sectorInfo = getSector(upper);

  const [rows, quote] = await Promise.all([fetchHistory(upper), fetchQuote(upper)]);
  if (rows.length < 30) throw new Error(`Not enough data for ${upper}`);

  const closes = rows.map((r) => r.close);
  const technicalScores = computeScore(closes);
  const technicalTotal = technicalScores.total;

  const price = closes[closes.length - 1];
  const prevClose = closes[closes.length - 2];
  const changePct = prevClose ? Math.round(((price - prevClose) / prevClose) * 10000) / 100 : null;
  const name = quote?.price?.longName ?? quote?.price?.shortName ?? upper;
  const trailingPe = (quote?.summaryDetail?.trailingPE as number | undefined) ?? null;
  const forwardPe = (quote?.summaryDetail?.forwardPE as number | undefined) ?? null;
  const cleanPe = (pe: number | null | undefined) => (pe && pe > 0 && pe < 5000 ? Math.round(pe * 100) / 100 : null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const trendData = (quote as any)?.recommendationTrend?.trend?.[0] ?? null;
  const analystResult = scoreAnalyst(trendData);
  const hasAnalyst = analystResult.count > 0;

  const blendedScore = hasAnalyst
    ? Math.min(100, Math.round(technicalTotal * 0.7 + analystResult.score))
    : technicalTotal;

  const rsiArr = calcRSI(closes);
  const { macdLine, signalLine, histogram } = calcMACD(closes);
  const { upper: bbUpper, middle: bbMiddle, lower: bbLower, pctB } = calcBollinger(closes);
  const ma20Arr = calcEMA(closes, 20);
  const ma50Arr = calcEMA(closes, 50);

  const fmt = (d: Date) => d.toISOString().split("T")[0];

  const history: OHLCVPoint[] = rows.map((r) => ({
    date: fmt(r.date),
    open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume,
  }));

  const rsiSeries: SeriesPoint[] = rows
    .map((r, i) => ({ date: fmt(r.date), value: rsiArr[i] }))
    .filter((p) => !isNaN(p.value));

  const macdSeries: MACDPoint[] = rows
    .map((r, i) => ({ date: fmt(r.date), macd: macdLine[i], signal: signalLine[i], histogram: histogram[i] }))
    .filter((p) => !isNaN(p.macd) && !isNaN(p.signal));

  const bbSeries: BBPoint[] = rows
    .map((r, i) => ({ date: fmt(r.date), upper: bbUpper[i], middle: bbMiddle[i], lower: bbLower[i], pct_b: pctB[i] }))
    .filter((p) => !isNaN(p.upper));

  const ma20Series: SeriesPoint[] = rows.map((r, i) => ({ date: fmt(r.date), value: ma20Arr[i] })).filter((p) => !isNaN(p.value));
  const ma50Series: SeriesPoint[] = rows.map((r, i) => ({ date: fmt(r.date), value: ma50Arr[i] })).filter((p) => !isNaN(p.value));

  return {
    ticker: upper,
    market: market as "JP" | "US",
    name: name ?? upper,
    price: Math.round(price * 100) / 100,
    currency,
    change_pct: changePct,
    score: blendedScore,
    score_components: {
      rsi: technicalScores.rsi,
      macd: technicalScores.macd,
      bollinger: technicalScores.bollinger,
      moving_avg: technicalScores.moving_avg,
      analyst: {
        score: hasAnalyst ? analystResult.score : 0,
        max: 30,
        value: null,
        signal: analystResult.signal,
      },
    },
    trailing_pe: cleanPe(trailingPe),
    forward_pe: cleanPe(forwardPe),
    pe_label: getPeLabel(trailingPe, market),
    analyst_score: hasAnalyst ? analystResult.score : null,
    analyst_signal: analystResult.signal,
    analyst_count: analystResult.count,
    sector: sectorInfo.key,
    sector_label: sectorInfo.label,
    last_updated: new Date().toISOString(),
    history,
    rsi_series: rsiSeries,
    macd_series: macdSeries,
    bb_series: bbSeries,
    ma20_series: ma20Series,
    ma50_series: ma50Series,
  };
}

export async function validateTicker(ticker: string): Promise<{ valid: boolean; name?: string; market?: string }> {
  try {
    const rows = await yf.historical(ticker.toUpperCase(), {
      period1: new Date(Date.now() - 7 * 86400_000).toISOString().split("T")[0],
      period2: new Date().toISOString().split("T")[0],
      interval: "1d",
    });
    if (rows.length === 0) return { valid: false };
    const q = await fetchQuote(ticker.toUpperCase()).catch(() => null);
    const name = q?.price?.longName ?? q?.price?.shortName ?? ticker.toUpperCase();
    return { valid: true, name, market: detectMarket(ticker) };
  } catch {
    return { valid: false };
  }
}
