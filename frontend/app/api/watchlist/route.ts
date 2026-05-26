import { NextRequest, NextResponse } from "next/server";
import { load, add, remove } from "@/lib/watchlistStore";
import { validateTicker } from "@/lib/yf";

export async function GET() {
  return NextResponse.json({ tickers: load() });
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const ticker = (body.ticker ?? "").trim().toUpperCase();

  if (!ticker || !/^[A-Z0-9]{1,6}(\.[A-Z]{1,2})?$/.test(ticker)) {
    return NextResponse.json({ detail: `無効なティッカー形式: ${ticker}` }, { status: 422 });
  }

  const { valid, name, market } = await validateTicker(ticker);
  if (!valid) {
    return NextResponse.json({ detail: `${ticker} はYahoo Financeで見つかりません` }, { status: 404 });
  }

  const tickers = add(ticker);
  return NextResponse.json({ tickers, name, market });
}

export async function DELETE(req: NextRequest) {
  const url = new URL(req.url);
  const ticker = url.searchParams.get("ticker")?.toUpperCase() ?? "";
  const tickers = remove(ticker);
  return NextResponse.json({ tickers });
}
