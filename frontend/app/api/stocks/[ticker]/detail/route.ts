import { NextRequest, NextResponse } from "next/server";
import { buildStockDetail } from "@/lib/yf";

export async function GET(_req: NextRequest, { params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = await params;
  try {
    const detail = await buildStockDetail(ticker.toUpperCase());
    return NextResponse.json(detail);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ detail: msg }, { status: 404 });
  }
}
