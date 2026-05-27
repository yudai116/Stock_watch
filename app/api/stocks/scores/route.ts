import { NextRequest, NextResponse } from "next/server";
import { buildStockScore, fetchSMHSignal } from "@/lib/yf";

export const dynamic = "force-dynamic";
import type { StockScore } from "@/types";

export async function GET(req: NextRequest) {
  const url = new URL(req.url);
  const tickersParam = url.searchParams.get("tickers") ?? "";
  const tickers = tickersParam.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean).slice(0, 50);
  const mode = (url.searchParams.get("mode") ?? "swing") as "swing" | "day";

  if (tickers.length === 0) {
    const emptyMacro = { regime: "neutral" as const, smh_5d_return: 0, multiplier: 1.0, label: "" };
    return NextResponse.json({ results: [], errors: {}, macro: emptyMacro });
  }

  // Fetch macro signal (SMH 5d momentum) in parallel with stock scores
  const [macro, ...stockResults] = await Promise.all([
    fetchSMHSignal(),
    ...tickers.map((ticker) =>
      buildStockScore(ticker, mode).catch((e: unknown) => ({
        _error: true as const,
        ticker,
        msg: e instanceof Error ? e.message : String(e),
      }))
    ),
  ]);

  const results: StockScore[] = [];
  const errors: Record<string, string> = {};

  for (const r of stockResults) {
    if ("_error" in r) {
      errors[r.ticker] = r.msg;
    } else {
      // Apply macro multiplier to final score
      r.score = Math.min(100, Math.round(r.score_raw * macro.multiplier));
      results.push(r);
    }
  }

  results.sort((a, b) => b.score - a.score);
  return NextResponse.json({ results, errors, macro });
}
