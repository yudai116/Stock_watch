/**
 * Fetch real OHLCV data for backtest using yahoo-finance2 (Node.js).
 *
 * Usage:
 *   node backtest/fetch_data.mjs              # daily (2015-2025) → price_data.json
 *   node backtest/fetch_data.mjs --intraday   # 1h bars (2 years) → price_data_intraday.json
 */
import { createRequire } from "module";
import { writeFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const require = createRequire(import.meta.url);
const YFClass = require("yahoo-finance2").default;
const yf = new YFClass();

const INTRADAY = process.argv.includes("--intraday");

// Full list: 9 stocks used in scoring + backtest
const TICKERS = [
  "NVDA",    // large
  "ASML",    // large
  "MSFT",    // large
  "8035.T",  // large (Tokyo Electron)
  "MRVL",    // mid
  "SMCI",    // mid
  "CRWD",    // mid (IPO 2019)
  "SOUN",    // small (SPAC 2022)
  "IONQ",    // small (SPAC 2021)
  // Additional large-cap for weight validation
  "AMD", "TSM", "AMAT", "LRCX", "KLAC",
  "6857.T",  // Advantest (JP large)
];

// Daily: 10 years of history
const DAILY_START = "2015-01-01";
const DAILY_END   = "2025-05-01";

// Intraday 1h: Yahoo Finance allows up to 730 days back
// Use 2 years ago from today minus 1 week safety margin
function getIntradayRange() {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 720); // ~2 years, safely within 730-day limit
  return {
    start: start.toISOString().split("T")[0],
    end:   end.toISOString().split("T")[0],
  };
}

async function fetchTickerDaily(ticker) {
  console.log(`  Fetching ${ticker} (daily) ...`);
  try {
    const rows = await yf.historical(ticker, {
      period1:  DAILY_START,
      period2:  DAILY_END,
      interval: "1d",
    });
    const filtered = rows
      .filter(r => r.close != null && r.close > 0)
      .map(r => ({
        date:   r.date.toISOString().split("T")[0],
        close:  Math.round(r.close  * 10000) / 10000,
        open:   Math.round((r.open  ?? r.close) * 10000) / 10000,
        high:   Math.round((r.high  ?? r.close) * 10000) / 10000,
        low:    Math.round((r.low   ?? r.close) * 10000) / 10000,
        volume: r.volume ?? 0,
      }));
    console.log(`    ${ticker}: ${filtered.length} bars (${filtered[0]?.date} → ${filtered[filtered.length-1]?.date})`);
    return filtered;
  } catch (e) {
    console.error(`    ${ticker}: FAILED — ${e.message}`);
    return null;
  }
}

async function fetchTickerIntraday(ticker, start, end) {
  console.log(`  Fetching ${ticker} (1h) ...`);
  try {
    // yahoo-finance2 chart() supports intraday intervals
    const result = await yf.chart(ticker, {
      period1:  start,
      period2:  end,
      interval: "1h",
    });
    const quotes = result?.quotes ?? [];
    const filtered = quotes
      .filter(r => r.close != null && r.close > 0)
      .map(r => {
        // chart() returns Date objects; format as ISO datetime
        const dt = r.date instanceof Date ? r.date : new Date(r.date);
        return {
          date:   dt.toISOString(),  // keep full timestamp for intraday
          close:  Math.round(r.close  * 10000) / 10000,
          open:   Math.round((r.open  ?? r.close) * 10000) / 10000,
          high:   Math.round((r.high  ?? r.close) * 10000) / 10000,
          low:    Math.round((r.low   ?? r.close) * 10000) / 10000,
          volume: r.volume ?? 0,
        };
      });
    console.log(`    ${ticker}: ${filtered.length} 1h bars (${filtered[0]?.date?.slice(0,10)} → ${filtered[filtered.length-1]?.date?.slice(0,10)})`);
    return filtered;
  } catch (e) {
    console.error(`    ${ticker}: FAILED — ${e.message}`);
    return null;
  }
}

async function main() {
  const dir = path.dirname(fileURLToPath(import.meta.url));

  if (INTRADAY) {
    const { start, end } = getIntradayRange();
    console.log(`Fetching ${TICKERS.length} tickers — 1h bars (${start} → ${end}) ...`);
    const result = {};
    for (const ticker of TICKERS) {
      const data = await fetchTickerIntraday(ticker, start, end);
      if (data && data.length >= 100) {
        result[ticker] = data;
      } else {
        console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 100)`);
      }
    }
    const out = path.join(dir, "price_data_intraday.json");
    writeFileSync(out, JSON.stringify(result, null, 0));
    console.log(`\nSaved ${Object.keys(result).length} tickers → price_data_intraday.json`);
    for (const [t, d] of Object.entries(result)) {
      console.log(`  ${t}: ${d.length} 1h bars`);
    }
  } else {
    console.log(`Fetching ${TICKERS.length} tickers (${DAILY_START} → ${DAILY_END}) ...`);
    const result = {};
    for (const ticker of TICKERS) {
      const data = await fetchTickerDaily(ticker);
      if (data && data.length >= 100) {
        result[ticker] = data;
      } else {
        console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 100)`);
      }
    }
    const out = path.join(dir, "price_data.json");
    writeFileSync(out, JSON.stringify(result, null, 0));
    console.log(`\nSaved ${Object.keys(result).length} tickers → price_data.json`);
    for (const [t, d] of Object.entries(result)) {
      console.log(`  ${t}: ${d.length} bars`);
    }
  }
}

main().catch(console.error);
