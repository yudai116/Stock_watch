export function formatPrice(price: number | null, currency: string): string {
  if (price === null) return "—";
  return new Intl.NumberFormat("ja-JP", {
    style: "currency",
    currency,
    minimumFractionDigits: currency === "JPY" ? 0 : 2,
    maximumFractionDigits: currency === "JPY" ? 0 : 2,
  }).format(price);
}

export function formatChangePct(pct: number | null): string {
  if (pct === null) return "—";
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}

export function scoreColor(score: number): string {
  if (score >= 70) return "text-emerald-400";
  if (score >= 50) return "text-yellow-400";
  if (score >= 30) return "text-orange-400";
  return "text-red-400";
}

export function scoreBg(score: number): string {
  if (score >= 70) return "bg-emerald-500";
  if (score >= 50) return "bg-yellow-500";
  if (score >= 30) return "bg-orange-500";
  return "bg-red-500";
}

export function scoreLabel(score: number): string {
  if (score >= 70) return "強い買いシグナル";
  if (score >= 50) return "中立/要観察";
  if (score >= 30) return "弱い";
  return "回避/弱気";
}

export function peLabelColor(label: string | null): string {
  if (!label) return "text-gray-400";
  if (label === "割安") return "text-emerald-400";
  if (label === "適正") return "text-yellow-400";
  return "text-red-400";
}

export function signalToJa(signal: string): string {
  const map: Record<string, string> = {
    extreme_oversold: "極度売られすぎ",
    oversold_recovery: "売られすぎ回復中",
    oversold: "売られすぎ",
    approaching_oversold: "売られすぎ接近",
    neutral: "中立",
    neutral_bullish: "やや強気",
    overbought: "買われすぎ",
    extreme_overbought: "極度買われすぎ",
    bullish_crossover: "ゴールデンクロス",
    recent_bullish_crossover: "直近クロス",
    bullish_momentum: "上昇モメンタム",
    bullish_fading: "強気弱まり",
    bearish_crossover: "デッドクロス",
    bearish_weakening: "弱気弱まり",
    bearish: "下落",
    below_lower_band: "下限帯突破",
    near_lower_band: "下限帯付近",
    lower_zone: "下方域",
    lower_half: "下半分",
    upper_half: "上半分",
    near_upper_band: "上限帯付近",
    above_upper_band: "上限帯突破",
    cross_above_ma20: "MA20上抜け",
    recent_cross_above_ma20: "直近上抜け",
    above_ma20: "MA20上方",
    at_ma20: "MA20付近",
    below_ma20: "MA20下方",
    golden_cross_zone: "ゴールデンクロス帯",
    at_crossover: "クロス付近",
    death_cross_zone: "デッドクロス帯",
    insufficient_data: "データ不足",
  };
  return map[signal] ?? signal;
}
