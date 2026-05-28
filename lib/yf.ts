// yahoo-finance2 v3 requires instantiation
// eslint-disable-next-line @typescript-eslint/no-require-imports
const YFClass = require("yahoo-finance2").default;
const yf = new YFClass({ suppressNotices: ["ripHistorical"] }) as InstanceType<typeof YFClass>;
import { computeScore, computeScoreDay, getPeLabel, scoreAnalyst } from "./scorer";
import { calcRSI, calcMACD, calcBollinger, calcEMA, calcATR } from "./indicators";
import { getSector, getSize, DEFAULT_SECTOR_WATCHLIST } from "./sectors";
import type { StockScore, StockDetail, OHLCVPoint, SeriesPoint, MACDPoint, BBPoint, SellSignals } from "@/types";

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
  startDate.setDate(startDate.getDate() - 250);  // wider window ensures enough valid rows for JP stocks

  // Use chart() directly: historical() delegates to chart() but throws on rows where
  // only SOME OHLCV fields are null. chart()'s own schema allows null values, so it
  // returns all rows without throwing, and we filter out null closes ourselves.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const result = await (yf as any).chart(ticker, {
    period1: startDate.toISOString().split("T")[0],
    period2: endDate.toISOString().split("T")[0],
    interval: "1d",
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const quotes: any[] = result?.quotes ?? [];
  return quotes
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    .filter((r: any) => r.close != null)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    .map((r: any) => ({
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
  if (rows.length < 20) throw new Error(`Not enough data for ${upper}`);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const qp = quote?.price as any;
  // Append today's live price so indicators reflect intraday movement, not just yesterday's close.
  const livePrice  = (qp?.regularMarketPrice  as number | null) ?? null;
  const liveVolume = (qp?.regularMarketVolume as number | null) ?? 0;

  const closes  = rows.map((r) => r.close);
  const volumes = rows.map((r) => r.volume);
  const highs   = rows.map((r) => r.high);
  const lows    = rows.map((r) => r.low);
  const liveDayHigh = (qp?.regularMarketDayHigh as number | null) ?? null;
  const liveDayLow  = (qp?.regularMarketDayLow  as number | null) ?? null;
  const closesForScore  = livePrice ? [...closes, livePrice]  : closes;
  const volumesForScore = livePrice ? [...volumes, liveVolume] : volumes;
  const highsForScore   = livePrice ? [...highs, liveDayHigh ?? livePrice] : highs;
  const lowsForScore    = livePrice ? [...lows,  liveDayLow  ?? livePrice] : lows;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const high52w  = (qp?.fiftyTwoWeekHigh as number | null) ?? null;
  const extras   = { volumes: volumesForScore, high52w, highs: highsForScore, lows: lowsForScore };
  const technicalScores = mode === "day" ? computeScoreDay(closesForScore, sizeKey, extras) : computeScore(closesForScore, sizeKey, extras);
  const technicalTotal = technicalScores.total; // 0-100

  // Displayed price = live market price; change % vs yesterday's close
  const price     = livePrice ?? closes[closes.length - 1];
  const prevClose = closes[closes.length - 1];  // last historical close = yesterday
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
      high52w: technicalScores.high52w,
      obv: technicalScores.obv,
      aroon: technicalScores.aroon,
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
  if (rows.length < 20) throw new Error(`Not enough data for ${upper}`);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const qp2 = quote?.price as any;
  const livePrice2  = (qp2?.regularMarketPrice  as number | null) ?? null;
  const liveVolume2 = (qp2?.regularMarketVolume as number | null) ?? 0;
  const liveDayHigh = (qp2?.regularMarketDayHigh as number | null) ?? null;
  const liveDayLow  = (qp2?.regularMarketDayLow  as number | null) ?? null;

  const closes  = rows.map((r) => r.close);
  const volumes = rows.map((r) => r.volume);
  const highs   = rows.map((r) => r.high);
  const lows    = rows.map((r) => r.low);

  // Append today's live data for up-to-date indicator calculation
  const closesForScore = livePrice2 ? [...closes, livePrice2]  : closes;
  const volumesForScore = livePrice2 ? [...volumes, liveVolume2] : volumes;
  const highsForScore  = livePrice2 ? [...highs, liveDayHigh ?? livePrice2] : highs;
  const lowsForScore   = livePrice2 ? [...lows,  liveDayLow  ?? livePrice2] : lows;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const high52w  = (qp2?.fiftyTwoWeekHigh as number | null) ?? null;
  const extras2  = { volumes: volumesForScore, high52w, highs: highsForScore, lows: lowsForScore };
  const technicalScores = mode === "day" ? computeScoreDay(closesForScore, sizeKey2, extras2) : computeScore(closesForScore, sizeKey2, extras2);
  const technicalTotal = technicalScores.total;

  const price     = livePrice2 ?? closes[closes.length - 1];
  const prevClose = closes[closes.length - 1];  // yesterday's close
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

  // Use live-appended arrays for indicator series so charts also show today's movement
  const rsiArr = calcRSI(closesForScore);
  const { macdLine, signalLine, histogram } = calcMACD(closesForScore);
  const { upper: bbUpper, middle: bbMiddle, lower: bbLower, pctB } = calcBollinger(closesForScore);
  const ma20Arr = calcEMA(closesForScore, 20);
  const ma50Arr = calcEMA(closesForScore, 50);
  const atrArr  = calcATR(highsForScore, lowsForScore, closesForScore);

  // Sell signals based on latest BB/ATR values (including today's live price)
  const n = closesForScore.length;
  const lastBbUpper   = bbUpper[n - 1] ?? null;
  const lastBbMiddle  = bbMiddle[n - 1] ?? null;
  const lastAtr       = atrArr[n - 1] ?? null;
  const distFrom52w   = high52w && price > 0 ? Math.round((high52w - price) / high52w * 1000) / 10 : null;
  const bb25 = lastBbUpper && lastBbMiddle
    ? Math.round((lastBbMiddle + (lastBbUpper - lastBbMiddle) * 1.25) * 100) / 100
    : null;
  const sellSignals: SellSignals = {
    bb_target_2sigma:    lastBbUpper ? Math.round(lastBbUpper * 100) / 100 : null,
    bb_target_2_5sigma:  bb25,
    atr_14d:             lastAtr ? Math.round(lastAtr * 100) / 100 : null,
    stop_loss_5pct:      price ? Math.round(price * 0.95 * 100) / 100 : null,
    dist_from_52w_high_pct: distFrom52w,
    high_52w:            high52w,
  };

  const fmt = (d: Date) => d.toISOString().split("T")[0];
  const today = new Date().toISOString().split("T")[0];

  // Historical rows for chart; append synthetic "today" bar if live price is available
  const histRows = livePrice2
    ? [...rows, { date: new Date(), open: livePrice2, high: liveDayHigh ?? livePrice2, low: liveDayLow ?? livePrice2, close: livePrice2, volume: liveVolume2 }]
    : rows;

  const history: OHLCVPoint[] = histRows.map((r) => ({
    date: r.date instanceof Date ? (r.date.toISOString().split("T")[0] === "1970-01-01" ? today : r.date.toISOString().split("T")[0]) : today,
    open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume,
  }));

  const rsiSeries: SeriesPoint[] = histRows
    .map((r, i) => ({ date: history[i].date, value: rsiArr[i] }))
    .filter((p) => !isNaN(p.value));

  const macdSeries: MACDPoint[] = histRows
    .map((r, i) => ({ date: history[i].date, macd: macdLine[i], signal: signalLine[i], histogram: histogram[i] }))
    .filter((p) => !isNaN(p.macd) && !isNaN(p.signal));

  const bbSeries: BBPoint[] = histRows
    .map((r, i) => ({ date: history[i].date, upper: bbUpper[i], middle: bbMiddle[i], lower: bbLower[i], pct_b: pctB[i] }))
    .filter((p) => !isNaN(p.upper));

  const ma20Series: SeriesPoint[] = histRows.map((r, i) => ({ date: history[i].date, value: ma20Arr[i] })).filter((p) => !isNaN(p.value));
  const ma50Series: SeriesPoint[] = histRows.map((r, i) => ({ date: history[i].date, value: ma50Arr[i] })).filter((p) => !isNaN(p.value));

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
      high52w: technicalScores.high52w,
      obv: technicalScores.obv,
      aroon: technicalScores.aroon,
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
    sell_signals: sellSignals,
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
