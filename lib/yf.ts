// yahoo-finance2 v3 requires instantiation
// eslint-disable-next-line @typescript-eslint/no-require-imports
const YFClass = require("yahoo-finance2").default;
const yf = new YFClass() as InstanceType<typeof YFClass>;
import { computeScore, computeScoreDay, getPeLabel, scoreAnalyst } from "./scorer";
import { calcRSI, calcMACD, calcBollinger, calcEMA } from "./indicators";
import { getSector, getSize, DEFAULT_SECTOR_WATCHLIST } from "./sectors";
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
  }, { validateResult: false });

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

// ── Macro regime signal (SMH 5-day momentum) ─────────────────────────────────
// Research result: SMH 5d momentum is the best macro filter for semis/AI stocks.
// Favorable vs unfavorable Sharpe delta: +0.414 (5d hold), +0.355 (1d hold).
// VIX was tested and found to *hurt* returns (-0.264 Δ Sharpe) — not used.
import type { MacroSignal, MacroRegime } from "@/types";

function macroMultiplier(ret: number): { regime: MacroRegime; multiplier: number } {
  if (ret >  0.02) return { regime: "bullish",  multiplier: 1.05 };
  if (ret >= 0)    return { regime: "neutral",  multiplier: 1.00 };
  if (ret >= -0.03) return { regime: "cautious", multiplier: 0.87 };
  return                   { regime: "bearish",  multiplier: 0.75 };
}

export async function fetchSMHSignal(): Promise<MacroSignal> {
  try {
    const rows = await fetchHistory("SMH");
    if (rows.length < 7) return { regime: "neutral", smh_5d_return: 0, multiplier: 1.0, label: "SMH データ不足" };
    const now = rows[rows.length - 1].close;
    const ago = rows[rows.length - 6].close;  // 5 trading days ago
    const ret = (now - ago) / ago;
    const { regime, multiplier } = macroMultiplier(ret);
    const sign = ret >= 0 ? "+" : "";
    return { regime, smh_5d_return: ret, multiplier, label: `SMH 5日: ${sign}${(ret * 100).toFixed(1)}%` };
  } catch {
    return { regime: "neutral", smh_5d_return: 0, multiplier: 1.0, label: "SMH 取得失敗" };
  }
}

export async function buildStockScore(ticker: string, mode: "swing" | "day" = "swing"): Promise<StockScore> {
  const upper = ticker.toUpperCase();
  const market = detectMarket(upper);
  const currency = getCurrency(market);
  const sectorInfo = getSector(upper);
  const sizeKey    = getSize(upper);

  const [rows, quote] = await Promise.all([fetchHistory(upper), fetchQuote(upper)]);
  if (rows.length < 30) throw new Error(`Not enough data for ${upper}`);

  const closes = rows.map((r) => r.close);
  const technicalScores = mode === "day" ? computeScoreDay(closes, sizeKey) : computeScore(closes, sizeKey);
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
    score_raw: blendedScore,  // populated; macro adjustment applied in API route
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
    size: sizeKey,
    last_updated: new Date().toISOString(),
  };
}

export async function buildStockDetail(ticker: string, mode: "swing" | "day" = "swing"): Promise<StockDetail> {
  const upper = ticker.toUpperCase();
  const market = detectMarket(upper);
  const currency = getCurrency(market);
  const sectorInfo = getSector(upper);
  const sizeKey2   = getSize(upper);

  const [rows, quote] = await Promise.all([fetchHistory(upper), fetchQuote(upper)]);
  if (rows.length < 30) throw new Error(`Not enough data for ${upper}`);

  const closes = rows.map((r) => r.close);
  const technicalScores = mode === "day" ? computeScoreDay(closes, sizeKey2) : computeScore(closes, sizeKey2);
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
    score_raw: blendedScore,  // macro adjustment applied in API route
    macro: { regime: "neutral", smh_5d_return: 0, multiplier: 1.0, label: "" },  // filled in API route
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
    size: sizeKey2,
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
