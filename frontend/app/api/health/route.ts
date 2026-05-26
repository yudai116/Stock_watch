import { NextResponse } from "next/server";

const startTime = Date.now();

export async function GET() {
  return NextResponse.json({
    status: "ok",
    uptime_seconds: Math.round((Date.now() - startTime) / 1000),
  });
}
