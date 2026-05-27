export type SectorKey = "semiconductor" | "memory" | "ai" | "quantum" | "nuclear";

export interface SectorInfo {
  key: SectorKey;
  label: string;
  color: string;
}

const SECTOR_MAP: Record<string, SectorInfo> = {
  // ── AI半導体 / GPU ────────────────────────────────────────────────────────
  "NVDA":   { key: "semiconductor", label: "AI半導体",   color: "violet" },
  "AMD":    { key: "semiconductor", label: "AI半導体",   color: "violet" },
  // ── 半導体装置・EUV ───────────────────────────────────────────────────────
  "ASML":   { key: "semiconductor", label: "半導体装置", color: "violet" },
  "LRCX":   { key: "semiconductor", label: "半導体装置", color: "violet" },
  "KLAC":   { key: "semiconductor", label: "半導体装置", color: "violet" },
  "AMAT":   { key: "semiconductor", label: "半導体装置", color: "violet" },
  "TER":    { key: "semiconductor", label: "半導体装置", color: "violet" },
  "8035.T": { key: "semiconductor", label: "半導体装置", color: "violet" },
  "6857.T": { key: "semiconductor", label: "半導体装置", color: "violet" },
  "6920.T": { key: "semiconductor", label: "半導体装置", color: "violet" },
  // ── ファウンドリ・設計 ────────────────────────────────────────────────────
  "TSM":    { key: "semiconductor", label: "半導体",     color: "violet" },
  "AVGO":   { key: "semiconductor", label: "半導体",     color: "violet" },
  "QCOM":   { key: "semiconductor", label: "半導体",     color: "violet" },
  "INTC":   { key: "semiconductor", label: "半導体",     color: "violet" },
  "MRVL":   { key: "semiconductor", label: "半導体",     color: "violet" },
  "ARM":    { key: "semiconductor", label: "半導体",     color: "violet" },
  "CDNS":   { key: "semiconductor", label: "EDA",        color: "violet" },
  "SNPS":   { key: "semiconductor", label: "EDA",        color: "violet" },
  "AMBA":   { key: "semiconductor", label: "エッジAI",   color: "violet" },
  "4063.T": { key: "semiconductor", label: "半導体材料", color: "violet" },
  "6762.T": { key: "semiconductor", label: "電子部品",   color: "violet" },
  "6503.T": { key: "semiconductor", label: "電機",       color: "violet" },
  // ── 特殊半導体 ────────────────────────────────────────────────────────────
  "WOLF":   { key: "semiconductor", label: "SiC半導体",  color: "violet" },
  "ONTO":   { key: "semiconductor", label: "検査装置",   color: "violet" },
  // ── メモリ ────────────────────────────────────────────────────────────────
  "MU":     { key: "memory",        label: "メモリ",     color: "blue"   },
  "WDC":    { key: "memory",        label: "メモリ",     color: "blue"   },
  // ── AI / クラウド (大企業) ────────────────────────────────────────────────
  "MSFT":   { key: "ai",            label: "AI",         color: "cyan"   },
  "GOOGL":  { key: "ai",            label: "AI",         color: "cyan"   },
  "META":   { key: "ai",            label: "AI",         color: "cyan"   },
  "AMZN":   { key: "ai",            label: "AI",         color: "cyan"   },
  "ORCL":   { key: "ai",            label: "AI",         color: "cyan"   },
  "CRM":    { key: "ai",            label: "AI",         color: "cyan"   },
  "CRWD":   { key: "ai",            label: "AIセキュリティ", color: "cyan" },
  // ── AI / クラウド (中小) ──────────────────────────────────────────────────
  "SMCI":   { key: "ai",            label: "AIサーバ",   color: "cyan"   },
  "SNOW":   { key: "ai",            label: "データAI",   color: "cyan"   },
  "PLTR":   { key: "ai",            label: "データAI",   color: "cyan"   },
  "PATH":   { key: "ai",            label: "AI自動化",   color: "cyan"   },
  "SOUN":   { key: "ai",            label: "音声AI",     color: "cyan"   },
  "BBAI":   { key: "ai",            label: "AI分析",     color: "cyan"   },
  "IREN":   { key: "ai",            label: "AIデータセンタ", color: "cyan" },
  "AI":     { key: "ai",            label: "AI",         color: "cyan"   },
  "6501.T": { key: "ai",            label: "ITサービス", color: "cyan"   },
  "6702.T": { key: "ai",            label: "ITサービス", color: "cyan"   },
  // ── 量子コンピュータ ──────────────────────────────────────────────────────
  "IONQ":   { key: "quantum",       label: "量子",       color: "purple" },
  "RGTI":   { key: "quantum",       label: "量子",       color: "purple" },
  "QBTS":   { key: "quantum",       label: "量子",       color: "purple" },
  "IBM":    { key: "quantum",       label: "量子/AI",    color: "purple" },
  // ── 核融合 / SMR ─────────────────────────────────────────────────────────
  "OKLO":   { key: "nuclear",       label: "SMR",        color: "amber"  },
  "SMR":    { key: "nuclear",       label: "SMR",        color: "amber"  },
  "BWXT":   { key: "nuclear",       label: "核融合/SMR", color: "amber"  },
};

export const DEFAULT_SECTOR_WATCHLIST: string[] = [
  // AI半導体
  "NVDA", "AMD",
  // 半導体装置
  "ASML", "LRCX", "KLAC", "AMAT", "TER", "8035.T", "6857.T", "6920.T",
  // ファウンドリ・設計
  "TSM", "AVGO", "QCOM", "INTC", "MRVL", "ARM", "CDNS", "SNPS", "AMBA",
  "4063.T", "6762.T", "6503.T",
  // 特殊半導体
  "WOLF", "ONTO",
  // メモリ
  "MU", "WDC",
  // AI大企業
  "MSFT", "GOOGL", "META", "AMZN", "ORCL", "CRM", "CRWD",
  // AI中小
  "SMCI", "SNOW", "PLTR", "PATH", "SOUN", "BBAI", "IREN", "AI",
  "6501.T", "6702.T",
  // 量子コンピュータ
  "IONQ", "RGTI", "QBTS", "IBM",
  // 核融合/SMR
  "OKLO", "SMR", "BWXT",
];

export const SECTOR_LABELS: Record<SectorKey | "ALL", string> = {
  ALL: "全て",
  semiconductor: "半導体",
  memory: "メモリ",
  ai: "AI",
  quantum: "量子コンピュータ",
  nuclear: "核融合/SMR",
};

export function getSector(ticker: string): SectorInfo {
  return SECTOR_MAP[ticker.toUpperCase()] ?? { key: "semiconductor", label: "テクノロジー", color: "gray" };
}
