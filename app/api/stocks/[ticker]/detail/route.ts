import { NextRequest, NextResponse } from "next/server";
import { buildStockDetail, fetchSMHSignal } from "@/lib/yf";

export async function GET(req: NextRequest, { params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = await params;
  const mode = (new URL(req.url).searchParams.get("mode") ?? "swing") as "swing" | "day";
  try {
    const [detail, macro] = await Promise.all([
      buildStockDetail(ticker.toUpperCase(), mode),
      fetchSMHSignal(),
    ]);
    detail.score = Math.min(100, Math.round(detail.score_raw * macro.multiplier));
    detail.macro = macro;
    return NextResponse.json(detail);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ detail: msg }, { status: 404 });
  }
}
