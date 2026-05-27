/**
 * Fetch 10yr historical prices for backtest using yahoo-finance2 (Node.js).
 * Saves to price_data.json for use by the Python backtest.
 */
import { createRequire } from "module";
import { writeFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const require = createRequire(import.meta.url);
const YFClass = require("yahoo-finance2").default;
const yf = new YFClass();

const TICKERS = [
  "NVDA", "AMD", "ASML", "TSM", "MU",
  "MSFT", "GOOGL", "IBM",
  "8035.T", "6857.T",
];

const START = "2015-01-01";
const END   = "2025-05-01";

async function fetchTicker(ticker) {
  const endDate   = new Date(END);
  const startDate = new Date(START);
  console.log(`  Fetching ${ticker} ...`);
  try {
    const rows = await yf.historical(ticker, {
      period1: START,
      period2: END,
      interval: "1d",
    });
    const filtered = rows
      .filter(r => r.close != null)
      .map(r => ({
        date:   r.date.toISOString().split("T")[0],
        close:  Math.round(r.close * 10000) / 10000,
        open:   Math.round((r.open  ?? r.close) * 10000) / 10000,
        high:   Math.round((r.high  ?? r.close) * 10000) / 10000,
        low:    Math.round((r.low   ?? r.close) * 10000) / 10000,
        volume: r.volume ?? 0,
      }));
    console.log(`    ${ticker}: ${filtered.length} days`);
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
    if (data && data.length >= 200) {
      result[ticker] = data;
    }
  }
  const out = path.join(path.dirname(fileURLToPath(import.meta.url)), "price_data.json");
  writeFileSync(out, JSON.stringify(result, null, 0));
  console.log(`\nSaved ${Object.keys(result).length} tickers → price_data.json`);
}

main().catch(console.error);
