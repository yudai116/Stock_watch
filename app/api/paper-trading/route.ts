import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import path from "path";

export async function GET() {
  const tradingDir = path.join(process.cwd(), "trading");
  const posPath = path.join(tradingDir, "positions.json");
  const logPath = path.join(tradingDir, "trade_log.json");
  const algoPath = path.join(tradingDir, "algorithms.json");

  if (!existsSync(posPath) || !existsSync(algoPath)) {
    return NextResponse.json({ available: false }, { status: 404 });
  }
  try {
    const positions = JSON.parse(readFileSync(posPath, "utf-8"));
    const logs = existsSync(logPath) ? JSON.parse(readFileSync(logPath, "utf-8")) : [];
    const algorithms = JSON.parse(readFileSync(algoPath, "utf-8"));
    return NextResponse.json({ available: true, positions, logs, algorithms });
  } catch {
    return NextResponse.json({ available: false, error: "parse error" }, { status: 500 });
  }
}
