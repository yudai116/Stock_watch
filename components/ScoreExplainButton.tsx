"use client";
import { useState } from "react";
import type { TradeMode } from "@/components/ModeToggle";

interface Props {
  mode?: TradeMode;
}

const SWING_WEIGHTS = {
  large: ["MACD×1.283", "MA×1.222", "RSI×0.776", "BB×0.719"],
  mid:   ["MA×1.204", "MACD×1.057", "BB×0.858", "RSI×0.881"],
  small: ["RSI×1.122", "BB×1.001", "MA×1.021", "MACD×0.856"],
};

const DAY_WEIGHTS = {
  large: ["MA×1.303", "MACD×1.100", "RSI×0.897", "BB×0.700"],
  mid:   ["MA×1.418", "RSI×1.104", "MACD×1.041", "BB×0.437"],
  small: ["MA×1.072", "MACD×1.026", "BB×0.986", "RSI×0.917"],
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
                        detail="14日間の騰落率を0〜100で表示。逆張り向き。小型株で特に有効。
点数目安: RSI<25→23点 / 25-35(上昇中)→19-22点 / 35-45→10-14点 / 45-55→8-12点 / 55-65→5-8点 / >65→0-5点"
                      />
                      <IndicatorRow
                        color="text-blue-400"
                        name="MACD (EMA12−EMA26)"
                        detail="シグナル線(EMA9)を上抜けるとゴールデンクロス。トレンド追従。大型株で特に有効。
点数目安: GC当日→24点 / GC後1-3日+ヒスト正→20点 / MACD>Signal+ヒスト拡大→15点 / MACD>Signal→10点 / ヒスト上昇のみ→6点 / DC当日→2点"
                      />
                      <IndicatorRow
                        color="text-purple-400"
                        name="ボリンジャーバンド (BB)"
                        detail="%B = (価格−下限)÷(上限−下限)。下限突破で最高点。スクイーズ（バンド幅縮小）+3点ボーナス。小型株で重視。
点数目安: %B<0→15-18点 / 0-0.1→12-15点 / 0.1-0.3→9-12点 / 0.5-0.7→5-7点 / >0.9→0-2点"
                      />
                      <IndicatorRow
                        color="text-orange-400"
                        name="移動平均線 (EMA20/50)"
                        detail="価格とEMA20の関係 + EMA20とEMA50のクロス判定。大型・中型株で重視。
点数目安: EMA20上抜け当日→+15点 / 直近3日上抜け→+12点 / EMA20上方→+4-9点 / EMA20>EMA50拡大→+8-9点加算 / EMA20<EMA50→+1点のみ加算"
                      />
                    </>
                  ) : (
                    <>
                      <IndicatorRow
                        color="text-emerald-400"
                        name="RSI — V字型 (デイ)"
                        detail="売られすぎ反発（RSI<30）と上昇モメンタム（RSI 55-70）の両方に高得点を与えるV字型カーブ。逆張りとモメンタムを両立。
点数目安: RSI<25→22点(上昇中) / 25-40→12-19点 / 40-50→9-12点 / 55-70→11-17点(モメンタム帯) / >75→3-8点"
                      />
                      <IndicatorRow
                        color="text-blue-400"
                        name="MACD (共通)"
                        detail="スイングと同じシグナル・同じ点数テーブル。ゴールデンクロスとヒストグラムの拡大・縮小でモメンタム方向を判定。
点数目安: GC当日→24点 / GC後1-3日+ヒスト正→20点 / MACD>Signal+ヒスト拡大→15点 / MACD>Signal→10点 / DC当日→2点"
                      />
                      <IndicatorRow
                        color="text-purple-400"
                        name="ボリンジャーバンド (共通)"
                        detail="%B絶対値に加え、%Bが上昇中なら方向ボーナス（%B<0.5→+4点、%B≥0.5→+3点）。逆張りとモメンタム両対応。
点数目安: %B<0→15-18点 / 0-0.1→12-15点 / >0.9→0-2点 (+ボーナス最大+4点)"
                      />
                      <IndicatorRow
                        color="text-orange-400"
                        name="移動平均線 (EMA5/10)"
                        detail="EMA5（短期5日）を使った素早い反応。EMA5>EMA10でゴールデンクロス帯判定。
点数目安: EMA5上抜け当日→+15点 / 直近3日上抜け→+13点 / EMA5上方+GC帯→+10点+加算 / EMA5上方→+7点"
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

              {/* Large cap bonus */}
              <section>
                <h3 className="text-gray-300 font-semibold mb-2">大型株ボーナス指標</h3>
                <p className="text-gray-400 text-xs leading-relaxed mb-2">
                  大型株のみ、追加の確認指標でボーナス点を加算します（バックテストでシャープレシオ +27%）。
                  中型・小型株では4指標モデルの方が優れているためボーナスなし。
                </p>
                <div className="space-y-1 text-xs text-gray-500">
                  <p><span className="text-cyan-400">52週高値距離 (最大+12点)</span>: 高値から5〜15%下が最高点。直近高値付近は上値余地が限られ低得点。</p>
                  <p><span className="text-cyan-400">OBV出来高圧力 (最大+8点)</span>: 10日間のOBV変化を平均出来高で正規化。機関投資家の買い集めを検出。</p>
                  <p><span className="text-indigo-400">Aroon (最大+10点)</span>: 25本足の高値/安値タイミングからトレンド方向を判定。AroonUp&gt;70かつAroonDown&lt;30で最高点。デイトレでは全規模に適用 (ΔSharpe +5.80)。</p>
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
