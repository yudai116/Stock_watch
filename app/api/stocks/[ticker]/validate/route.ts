import { NextRequest, NextResponse } from "next/server";
import { validateTicker } from "@/lib/yf";

export async function GET(_req: NextRequest, { params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = await params;
  const result = await validateTicker(ticker.toUpperCase());
  return NextResponse.json(result);
}
