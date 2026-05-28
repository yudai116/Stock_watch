export function yahooFinanceUrls(ticker: string): { label: string; href: string }[] {
  if (ticker.endsWith(".T")) {
    return [
      { label: "Yahoo Finance JP", href: `https://finance.yahoo.co.jp/quote/${ticker}` },
      { label: "Yahoo Finance US", href: `https://finance.yahoo.com/quote/${ticker}` },
    ];
  }
  return [{ label: "Yahoo Finance", href: `https://finance.yahoo.com/quote/${ticker}` }];
}

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

export function analystSignalToJa(signal: string | null): string {
  if (!signal) return "—";
  const map: Record<string, string> = {
    strong_buy: "強い買い",
    buy: "買い",
    hold: "中立",
    sell: "売り",
    strong_sell: "強い売り",
    no_data: "—",
  };
  return map[signal] ?? signal;
}

export function analystSignalColor(signal: string | null): string {
  if (!signal || signal === "no_data") return "text-gray-400";
  if (signal === "strong_buy" || signal === "buy") return "text-emerald-400";
  if (signal === "hold") return "text-yellow-400";
  return "text-red-400";
}

export function sectorBadgeColor(color: string): string {
  const map: Record<string, string> = {
    violet: "bg-violet-900/60 text-violet-300",
    blue: "bg-blue-900/60 text-blue-300",
    cyan: "bg-cyan-900/60 text-cyan-300",
    purple: "bg-purple-900/60 text-purple-300",
    amber: "bg-amber-900/60 text-amber-300",
    gray: "bg-gray-800 text-gray-400",
  };
  return map[color] ?? "bg-gray-800 text-gray-400";
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
    // OBV signals
    strong_accumulation: "強い買い集め",
    accumulation: "買い集め",
    slight_accumulation: "やや買い集め",
    neutral_obv: "OBV中立",
    distribution: "売り圧力",
    no_volume: "出来高なし",
    "n/a": "—",
    // day trade signals
    oversold_bounce: "売られすぎ反発",
    recovering: "回復中",
    neutral_low: "中立（やや低め）",
    momentum_up: "上昇モメンタム",
    momentum_fading: "モメンタム減衰",
    strong_momentum: "強いモメンタム",
    overbought_momentum: "過熱モメンタム",
    cross_above_ma5: "EMA5上抜け",
    recent_cross_above_ma5: "EMA5上抜け（直近）",
    above_ma5: "EMA5上方",
    at_ma5: "EMA5付近",
    below_ma5: "EMA5下方",
    // Aroon signals
    aroon_strong_up: "強い上昇トレンド",
    aroon_uptrend_building: "上昇トレンド形成中",
    aroon_uptrend: "上昇トレンド",
    aroon_sideways: "横ばい",
    aroon_downtrend: "下降トレンド",
    aroon_strong_down: "強い下降トレンド",
  };
  return map[signal] ?? signal;
}
