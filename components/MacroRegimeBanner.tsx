"use client";
import type { MacroSignal } from "@/types";

interface Props {
  macro?: MacroSignal;
}

const REGIME_CONFIG = {
  bullish: {
    bg:   "bg-emerald-950/60 border-emerald-800",
    text: "text-emerald-300",
    badge: "bg-emerald-800 text-emerald-200",
    icon: "↑",
    label: "強気",
    desc: "半導体セクター上昇 — シグナル信頼度 高 (×1.05)",
  },
  neutral: {
    bg:   "bg-gray-800/60 border-gray-700",
    text: "text-gray-300",
    badge: "bg-gray-700 text-gray-300",
    icon: "→",
    label: "中立",
    desc: "半導体セクター横ばい — 通常スコア適用",
  },
  cautious: {
    bg:   "bg-yellow-950/60 border-yellow-800",
    text: "text-yellow-300",
    badge: "bg-yellow-800 text-yellow-200",
    icon: "↓",
    label: "注意",
    desc: "半導体セクター下落 — スコアを13%割引 (×0.87)",
  },
  bearish: {
    bg:   "bg-red-950/60 border-red-800",
    text: "text-red-300",
    badge: "bg-red-800 text-red-200",
    icon: "↓↓",
    label: "弱気",
    desc: "半導体セクター下落中 — スコアを25%割引 (×0.75)",
  },
} as const;

export default function MacroRegimeBanner({ macro }: Props) {
  if (!macro) return null;

  const cfg = REGIME_CONFIG[macro.regime];
  const sign = macro.smh_5d_return >= 0 ? "+" : "";
  const pct  = (macro.smh_5d_return * 100).toFixed(1);

  return (
    <div className={`${cfg.bg} border rounded-xl px-3 py-2.5 flex items-center gap-3 text-xs`}>
      {/* Regime badge */}
      <span className={`${cfg.badge} px-2 py-0.5 rounded font-bold flex-shrink-0`}>
        {cfg.icon} {cfg.label}
      </span>

      {/* Description */}
      <div className="flex-1 min-w-0">
        <span className="text-gray-400">{cfg.desc}</span>
      </div>

      {/* SMH value */}
      <div className="flex-shrink-0 text-right">
        <div className={`${cfg.text} font-mono font-medium tabular-nums`}>
          {sign}{pct}%
        </div>
        <div className="text-gray-600 text-xs">SMH 5日</div>
      </div>
    </div>
  );
}
