import type { StockScoresResponse, StockDetail, WatchlistResponse, ValidateResponse } from "@/types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getWatchlist: () => request<WatchlistResponse>("/api/watchlist"),

  addTicker: (ticker: string) =>
    request<WatchlistResponse>("/api/watchlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker }),
    }),

  removeTicker: (ticker: string) =>
    request<WatchlistResponse>(`/api/watchlist/${ticker}`, { method: "DELETE" }),

  getScores: (tickers: string[]) =>
    request<StockScoresResponse>(`/api/stocks/scores?tickers=${tickers.join(",")}`),

  getDetail: (ticker: string) =>
    request<StockDetail>(`/api/stocks/${ticker}/detail`),

  validate: (ticker: string) =>
    request<ValidateResponse>(`/api/stocks/${ticker}/validate`),
};
