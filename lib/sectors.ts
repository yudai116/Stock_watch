export type SectorKey = "semiconductor" | "memory" | "ai" | "quantum" | "nuclear";

export interface SectorInfo {
  key: SectorKey;
  label: string;
  color: string;
}

const SECTOR_MAP: Record<string, SectorInfo> = {
  "NVDA":   { key: "semiconductor", label: "AI半導体", color: "violet" },
  "AMD":    { key: "semiconductor", label: "半導体",   color: "violet" },
  "ASML":   { key: "semiconductor", label: "半導体",   color: "violet" },
  "TSM":    { key: "semiconductor", label: "半導体",   color: "violet" },
  "8035.T": { key: "semiconductor", label: "半導体",   color: "violet" },
  "6857.T": { key: "semiconductor", label: "半導体",   color: "violet" },
  "6920.T": { key: "semiconductor", label: "半導体",   color: "violet" },
  "MU":     { key: "memory",        label: "メモリ",   color: "blue" },
  "MSFT":   { key: "ai",            label: "AI",       color: "cyan" },
  "GOOGL":  { key: "ai",            label: "AI",       color: "cyan" },
  "PLTR":   { key: "ai",            label: "AI",       color: "cyan" },
  "IONQ":   { key: "quantum",       label: "量子",     color: "purple" },
  "IBM":    { key: "quantum",       label: "量子/AI",  color: "purple" },
  "OKLO":   { key: "nuclear",       label: "SMR",      color: "amber" },
  "SMR":    { key: "nuclear",       label: "SMR",      color: "amber" },
};

export const DEFAULT_SECTOR_WATCHLIST = [
  "NVDA", "AMD", "ASML", "TSM", "8035.T", "6857.T",
  "MU",
  "MSFT", "GOOGL", "PLTR",
  "IONQ", "IBM",
  "OKLO", "SMR",
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
