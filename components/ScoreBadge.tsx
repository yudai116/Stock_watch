"use client";
import { scoreBg, scoreLabel } from "@/lib/formatters";

interface Props {
  score: number;
  showLabel?: boolean;
  size?: "sm" | "md" | "lg";
}

export default function ScoreBadge({ score, showLabel = false, size = "md" }: Props) {
  const bg = scoreBg(score);
  const sizeClass = size === "sm" ? "text-xs px-2 py-0.5" : size === "lg" ? "text-xl px-4 py-2 font-bold" : "text-sm px-3 py-1";

  return (
    <div className="flex flex-col items-center gap-1">
      <span className={`${bg} text-white rounded-full font-semibold tabular-nums ${sizeClass}`}>
        {score}
      </span>
      {showLabel && (
        <span className="text-xs text-gray-400">{scoreLabel(score)}</span>
      )}
    </div>
  );
}
