import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import path from "path";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const mode = searchParams.get("mode") === "day" ? "day" : "swing";

  // v3: mode-specific files; v2 fallback for swing
  const v3Path = path.join(process.cwd(), "backtest", `strategy_results_${mode}.json`);
  const v2Path = path.join(process.cwd(), "backtest", "strategy_results.json");
  const filePath = existsSync(v3Path) ? v3Path : (mode === "swing" ? v2Path : null);

  if (!filePath || !existsSync(filePath)) {
    return NextResponse.json({ available: false }, { status: 404 });
  }
  try {
    const data = JSON.parse(readFileSync(filePath, "utf-8"));
    return NextResponse.json({ available: true, ...data });
  } catch {
    return NextResponse.json({ available: false, error: "parse error" }, { status: 500 });
  }
}
