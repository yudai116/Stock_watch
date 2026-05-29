import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import path from "path";

export async function GET() {
  const filePath = path.join(process.cwd(), "backtest", "strategy_results.json");
  if (!existsSync(filePath)) {
    return NextResponse.json({ available: false }, { status: 404 });
  }
  try {
    const raw = readFileSync(filePath, "utf-8");
    const data = JSON.parse(raw);
    return NextResponse.json({ available: true, ...data });
  } catch {
    return NextResponse.json({ available: false, error: "parse error" }, { status: 500 });
  }
}
