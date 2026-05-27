"use client";
import { useState } from "react";
import type { TradeMode } from "@/components/ModeToggle";

interface Props {
  mode?: TradeMode;
}

const SWING_WEIGHTS = {
  large: ["MACD×1.274", "MA×1.209", "BB×0.795", "RSI×0.722"],
  mid:   ["MA×1.182", "MACD×1.021", "BB×0.956", "RSI×0.842"],
  small: ["RSI×1.086", "BB×1.001", "MA×1.000", "MACD×0.913"],
};

const DAY_WEIGHTS = {
  large: ["MA×1.264", "MACD×1.109", "RSI×0.885", "BB×0.743"],
  mid:   ["MA×1.185", "MACD×1.092", "RSI×0.966", "BB×0.757"],
  small: ["MA×1.105", "MACD×1.015", "BB×0.969", "RSI×0.911"],
};

export default function ScoreExplainButton({ mode = "swing" }: Props) {
  const [open, setOpen] = useState(false);
  const weights = mode === "day" ? DAY_WEIGHTS : SWING_WEIGHTS;

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="w-5 h-5 rounded-full bg-gray-700 hover:bg-gray-600 text-gray-400 hover:text-white text-xs font-bold leading-none flex items-center justify-center transition-colors flex-shrink-0"
        title="スコアの算出方法"
        aria-label="スコアの算出方法を表示"
      >
        ?
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-0 sm:p-4"
          onClick={() => setOpen(false)}
        >
          {/* backdrop */}
          <div className="absolute inset-0 bg-black/70" />

          {/* modal */}
          <div
            className="relative w-full sm:max-w-lg bg-gray-900 border border-gray-700 rounded-t-2xl sm:rounded-2xl overflow-y-auto max-h-[90dvh] shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* header */}
            <div className="sticky top-0 bg-gray-900 border-b border-gray-800 px-5 py-4 flex items-center justify-between">
              <h2 className="text-white font-bold text-base">スコアの算出方法</h2>
              <button
                onClick={() => setOpen(false)}
                className="text-gray-500 hover:text-white text-xl leading-none px-1"
              >
                ×
              </button>
            </div>

            <div className="px-5 py-4 space-y-5 text-sm">

              {/* Overview */}
              <section>
                <h3 className="text-gray-300 font-semibold mb-2">スコアの概要</h3>
                <p className="text-gray-400 leading-relaxed">
                  0〜100点のスコアで「今買いやすい状態か」を数値化しています。
                  4つのテクニカル指標に、アナリスト評価と半導体セクターのマクロ環境を加えて算出します。
                </p>
                <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-xs">
                  {[
                    { dot: "bg-emerald-500", text: "70以上 — 強い買いシグナル" },
                    { dot: "bg-yellow-500",  text: "50〜69 — 中立・要観察" },
                    { dot: "bg-orange-500",  text: "30〜49 — 弱い" },
                    { dot: "bg-red-500",     text: "〜29 — 回避" },
                  ].map((item) => (
                    <div key={item.text} className="flex items-center gap-1.5">
                      <span className={`${item.dot} w-2 h-2 rounded-full flex-shrink-0`} />
                      <span className="text-gray-400">{item.text}</span>
                    </div>
                  ))}
                </div>
              </section>

              {/* Formula */}
              <section>
                <h3 className="text-gray-300 font-semibold mb-2">計算式</h3>
                <div className="bg-gray-800 rounded-xl p-3 text-xs font-mono text-gray-300 leading-relaxed space-y-1">
                  <p>テクニカル = RSI + MACD + BB + MA &nbsp;&nbsp;(各最大25点)</p>
                  <p>合計 = テクニカル×70% + アナリスト評価×30%</p>
                  <p className={mode === "day" ? "text-orange-400" : "text-blue-400"}>
                    表示スコア = 合計 × SMHマクロ補正
                  </p>
                </div>
              </section>

              {/* Indicators */}
              <section>
                <h3 className="text-gray-300 font-semibold mb-3">
                  {mode === "day" ? "デイトレ指標 (各最大25点)" : "スイング指標 (各最大25点)"}
                </h3>
                <div className="space-y-3">
                  {mode === "swing" ? (
                    <>
                      <IndicatorRow
                        color="text-emerald-400"
                        name="RSI (相対力指数)"
                        detail="14日間の騰落率を0〜100で表示。30以下で売られすぎ（買いシグナル）。逆張り向き。小型株で特に有効。"
                      />
                      <IndicatorRow
                        color="text-blue-400"
                        name="MACD"
                        detail="12日・26日EMAの差分。シグナル線を上抜けるとゴールデンクロス（買いシグナル）。トレンド追従。大型株で特に有効。"
                      />
                      <IndicatorRow
                        color="text-purple-400"
                        name="ボリンジャーバンド (BB)"
                        detail="価格が平均±2σのどこにいるかを%Bで表示。下限バンド突破で高得点。スクイーズ（バンド収縮）でボーナス加算。"
                      />
                      <IndicatorRow
                        color="text-orange-400"
                        name="移動平均線 (EMA20/50)"
                        detail="EMA20上抜けで高得点。EMA20がEMA50の上でゴールデンクロス帯なら追加ポイント。大型・中型株で重視。"
                      />
                    </>
                  ) : (
                    <>
                      <IndicatorRow
                        color="text-emerald-400"
                        name="RSI — V字型 (デイ)"
                        detail="売られすぎ（RSI<30）の反発と、上昇モメンタム（RSI 55〜70）の両方を評価するV字型カーブ。逆張りとモメンタムを両立。"
                      />
                      <IndicatorRow
                        color="text-blue-400"
                        name="MACD (共通)"
                        detail="スイングと同じシグナル。ゴールデンクロスとヒストグラムの拡大・縮小でモメンタム方向を判定。"
                      />
                      <IndicatorRow
                        color="text-purple-400"
                        name="ボリンジャーバンド (共通)"
                        detail="%B絶対値に加え、%Bが上昇中なら方向ボーナス（下半分+4点、上半分+3点）。逆張りとモメンタム両対応。"
                      />
                      <IndicatorRow
                        color="text-orange-400"
                        name="移動平均線 (EMA5/10)"
                        detail="EMA5を使った短期反応。EMA5上抜けで即高得点。EMA5がEMA10の上かつ乖離拡大で上昇トレンド判定。"
                      />
                    </>
                  )}
                </div>
              </section>

              {/* Analyst */}
              <section>
                <h3 className="text-gray-300 font-semibold mb-2">アナリスト評価 (最大30点)</h3>
                <p className="text-gray-400 leading-relaxed">
                  機関投資家アナリストの推奨（強い買い/買い/中立/売り/強い売り）を集計し、
                  加重スコアに変換。データがない銘柄はテクニカル100%で計算。
                </p>
              </section>

              {/* Macro */}
              <section>
                <h3 className="text-gray-300 font-semibold mb-2">マクロ環境補正 (SMH 5日モメンタム)</h3>
                <p className="text-gray-400 leading-relaxed mb-2">
                  半導体ETF（SMH）の直近5営業日リターンで、セクター全体の地合いを判定します。
                  バックテスト（36銘柄・5年間）でVIXより予測精度が高いことを確認済み。
                </p>
                <div className="space-y-1 text-xs">
                  {[
                    { color: "text-emerald-400", cond: "SMH 5日 > +2%", mult: "×1.05", label: "強気" },
                    { color: "text-gray-400",    cond: "SMH 5日 0〜+2%", mult: "×1.00", label: "中立" },
                    { color: "text-yellow-400",  cond: "SMH 5日 −3〜0%", mult: "×0.87", label: "注意" },
                    { color: "text-red-400",     cond: "SMH 5日 < −3%",  mult: "×0.75", label: "弱気" },
                  ].map((r) => (
                    <div key={r.cond} className="flex items-center gap-2">
                      <span className={`${r.color} w-12 flex-shrink-0 font-medium`}>{r.label}</span>
                      <span className="text-gray-500 flex-1">{r.cond}</span>
                      <span className={`${r.color} font-mono`}>{r.mult}</span>
                    </div>
                  ))}
                </div>
              </section>

              {/* Weights by size */}
              <section>
                <h3 className="text-gray-300 font-semibold mb-2">時価総額別の重み付け</h3>
                <p className="text-gray-400 text-xs mb-2 leading-relaxed">
                  50銘柄・10年分のMCバックテスト（ウォークフォワード）でシャープレシオを最大化する
                  重みを規模別に算出しています。
                </p>
                <div className="space-y-2 text-xs">
                  {(["large", "mid", "small"] as const).map((sz) => (
                    <div key={sz} className="bg-gray-800 rounded-lg px-3 py-2">
                      <span className="text-gray-400 font-medium mr-2">
                        {sz === "large" ? "大型株" : sz === "mid" ? "中型株" : "小型株"}
                      </span>
                      <span className="text-gray-500">{weights[sz].join(" + ")}</span>
                    </div>
                  ))}
                </div>
              </section>

              {/* Disclaimer */}
              <p className="text-gray-600 text-xs border-t border-gray-800 pt-3">
                ※ このスコアは情報提供のみを目的としています。投資判断は自己責任でお願いします。
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function IndicatorRow({ color, name, detail }: { color: string; name: string; detail: string }) {
  return (
    <div className="flex gap-2.5">
      <span className={`${color} font-bold text-base leading-none mt-0.5 flex-shrink-0`}>▍</span>
      <div>
        <p className={`${color} font-medium`}>{name}</p>
        <p className="text-gray-500 text-xs mt-0.5 leading-relaxed">{detail}</p>
      </div>
    </div>
  );
}
