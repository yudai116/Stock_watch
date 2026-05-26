import { NextRequest, NextResponse } from "next/server";
import { buildStockScore } from "@/lib/yf";
import type { StockScore } from "@/types";

export async function GET(req: NextRequest) {
  const url = new URL(req.url);
  const tickersParam = url.searchParams.get("tickers") ?? "";
  const tickers = tickersParam.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean).slice(0, 50);

  if (tickers.length === 0) {
    return NextResponse.json({ results: [], errors: {} });
  }

  const results: StockScore[] = [];
  const errors: Record<string, string> = {};

  await Promise.all(
    tickers.map(async (ticker) => {
      try {
        results.push(await buildStockScore(ticker));
      } catch (e: unknown) {
        errors[ticker] = e instanceof Error ? e.message : String(e);
      }
    })
  );

  results.sort((a, b) => b.score - a.score);
  return NextResponse.json({ results, errors });
}
