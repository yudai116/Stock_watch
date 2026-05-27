"use client";
import useSWR from "swr";
import { api } from "@/lib/api";
import type { StockDetail } from "@/types";

export function useStockDetail(ticker: string, mode: "swing" | "day" = "swing") {
  const { data, isLoading, error, mutate } = useSWR<StockDetail>(
    ticker ? [`/api/stocks/${ticker}/detail`, mode] : null,
    () => api.getDetail(ticker, mode),
    { revalidateOnFocus: true, refreshInterval: 300_000 }
  );
  return { detail: data, isLoading, error, mutate };
}
