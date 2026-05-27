import type { StockScoresResponse, StockDetail, WatchlistResponse, ValidateResponse } from "@/types";

// Always use relative paths — API routes are in the same Next.js app (no separate backend needed)
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const base = typeof window === "undefined" ? "http://localhost:3000" : "";
  const res = await fetch(`${base}${path}`, init);
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
    request<WatchlistResponse>(`/api/watchlist?ticker=${ticker}`, { method: "DELETE" }),

  getScores: (tickers: string[], mode: "swing" | "day" = "swing") =>
    request<StockScoresResponse>(`/api/stocks/scores?tickers=${tickers.join(",")}&mode=${mode}`),

  getDetail: (ticker: string, mode: "swing" | "day" = "swing") =>
    request<StockDetail>(`/api/stocks/${ticker}/detail?mode=${mode}`),

  validate: (ticker: string) =>
    request<ValidateResponse>(`/api/stocks/${ticker}/validate`),
};
