// Server-side watchlist storage using a module-level variable (in-memory, persists per instance)
// On Vercel, each serverless function invocation may be cold — data persists within warm instances.
// For durable cross-device storage, set WATCHLIST env var with a comma-separated list of tickers.

import { DEFAULT_WATCHLIST } from "./yf";

const ENV_WATCHLIST = process.env.WATCHLIST;

let _tickers: string[] = ENV_WATCHLIST
  ? ENV_WATCHLIST.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean)
  : [...DEFAULT_WATCHLIST];

export function load(): string[] {
  return [..._tickers];
}

export function add(ticker: string): string[] {
  const upper = ticker.toUpperCase();
  if (!_tickers.includes(upper)) _tickers = [..._tickers, upper];
  return load();
}

export function remove(ticker: string): string[] {
  const upper = ticker.toUpperCase();
  _tickers = _tickers.filter((t) => t !== upper);
  return load();
}
