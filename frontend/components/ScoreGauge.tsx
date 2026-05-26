"use client";
import { scoreLabel } from "@/lib/formatters";

interface Props {
  score: number;
}

export default function ScoreGauge({ score }: Props) {
  const angle = (score / 100) * 180 - 90;

  const getColor = () => {
    if (score >= 70) return "#10b981";
    if (score >= 50) return "#eab308";
    if (score >= 30) return "#f97316";
    return "#ef4444";
  };

  const color = getColor();

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-48 h-24 overflow-hidden">
        <svg viewBox="0 0 200 100" className="w-full h-full">
          <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="#374151" strokeWidth="16" strokeLinecap="round" />
          <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke={color} strokeWidth="16" strokeLinecap="round"
            strokeDasharray={`${(score / 100) * 283} 283`} />
          <line
            x1="100" y1="100"
            x2={100 + 70 * Math.cos(((angle - 90) * Math.PI) / 180)}
            y2={100 + 70 * Math.sin(((angle - 90) * Math.PI) / 180)}
            stroke="white" strokeWidth="2" strokeLinecap="round"
          />
          <circle cx="100" cy="100" r="5" fill="white" />
          <text x="10" y="98" fill="#6b7280" fontSize="10" textAnchor="middle">0</text>
          <text x="190" y="98" fill="#6b7280" fontSize="10" textAnchor="middle">100</text>
        </svg>
      </div>
      <div className="text-center mt-1">
        <span className="text-4xl font-bold" style={{ color }}>{score}</span>
        <span className="text-gray-400 ml-1 text-sm">/100</span>
        <p className="text-sm mt-1" style={{ color }}>{scoreLabel(score)}</p>
      </div>
    </div>
  );
}
