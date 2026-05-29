/**
 * Fetch real OHLCV data for backtest using yahoo-finance2 (Node.js).
 * Saves to price_data.json for use by the Python backtest.
 */
import { createRequire } from "module";
import { writeFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const require = createRequire(import.meta.url);
const YFClass = require("yahoo-finance2").default;
const yf = new YFClass();

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

const START = "2015-01-01";
const END   = "2025-05-01";

async function fetchTicker(ticker) {
  console.log(`  Fetching ${ticker} ...`);
  try {
    const rows = await yf.historical(ticker, {
      period1: START,
      period2: END,
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

async function main() {
  console.log(`Fetching ${TICKERS.length} tickers (${START} → ${END}) ...`);
  const result = {};
  for (const ticker of TICKERS) {
    const data = await fetchTicker(ticker);
    if (data && data.length >= 100) {
      result[ticker] = data;
    } else {
      console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 100)`);
    }
  }
  const out = path.join(path.dirname(fileURLToPath(import.meta.url)), "price_data.json");
  writeFileSync(out, JSON.stringify(result, null, 0));
  console.log(`\nSaved ${Object.keys(result).length} tickers → price_data.json`);
  for (const [t, d] of Object.entries(result)) {
    console.log(`  ${t}: ${d.length} bars`);
  }
}

main().catch(console.error);
