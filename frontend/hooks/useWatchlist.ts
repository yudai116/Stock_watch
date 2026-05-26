"use client";
import useSWR from "swr";
import { api } from "@/lib/api";
import type { StockScore, StockScoresResponse } from "@/types";

export function useWatchlist(refreshInterval = 300_000) {
  const { data: wl, mutate: mutateWl } = useSWR(
    "/api/watchlist",
    () => api.getWatchlist(),
    { revalidateOnFocus: true }
  );

  const tickers = wl?.tickers ?? [];

  const { data: scoresData, isLoading, mutate: mutateScores } = useSWR<StockScoresResponse>(
    tickers.length > 0 ? ["/api/stocks/scores", tickers.join(",")] : null,
    () => api.getScores(tickers),
    { refreshInterval, revalidateOnFocus: true }
  );

  const stocks: StockScore[] = scoresData?.results ?? [];
  const errors = scoresData?.errors ?? {};

  async function addTicker(ticker: string) {
    const result = await api.addTicker(ticker);
    await mutateWl({ tickers: result.tickers });
    await mutateScores();
    return result;
  }

  async function removeTicker(ticker: string) {
    const result = await api.removeTicker(ticker);
    await mutateWl({ tickers: result.tickers });
    await mutateScores();
    return result;
  }

  return { stocks, tickers, errors, isLoading, addTicker, removeTicker, mutate: mutateScores };
}
