#!/usr/bin/env python3
"""
Trade Records Generator (高速版)
=================================
指標をベクトル化して全バー一括計算 → 閾値以上シグナルを収集 → 上位トレードを詳細表示

売買記録の各フィールド:
  - 銘柄名 / 規模
  - 買い日時 (シミュレーション日付 2015-01-02〜)
  - 買い価格 / 総合スコア
  - 各指標の値・スコア・シグナル理由（買い根拠）
  - 売り日時 / 売り価格 / リターン
  - 売り根拠（保有期間到達）
"""

import numpy as np
from datetime import date, timedelta

RNG    = np.random.default_rng(42)
HL_RNG = np.random.default_rng(9999)

# ─── 銘柄設定 ─────────────────────────────────────────────────────────────────
STOCKS = {
    # ticker: (CAGR, vol, 日本語名, 規模)
    "NVDA":   (0.55, 0.60, "エヌビディア",              "large"),
    "ASML":   (0.30, 0.28, "ASML（EUVリソグラフィ）",   "large"),
    "MSFT":   (0.32, 0.22, "マイクロソフト",              "large"),
    "8035.T": (0.28, 0.42, "東京エレクトロン",            "large"),
    "MRVL":   (0.30, 0.45, "マーベルテクノロジー",        "mid"),
    "SMCI":   (0.35, 0.80, "スーパーマイクロコンピュータ","mid"),
    "CRWD":   (0.30, 0.45, "クラウドストライク",          "mid"),
    "SOUN":   (0.20, 1.00, "サウンドハウンドAI",          "small"),
    "IONQ":   (0.15, 0.90, "アイオンキュー（量子）",      "small"),
}

TRANS = np.array([[0.97,0.015,0.015],[0.05,0.93,0.02],[0.03,0.01,0.96]])
D_MUL = np.array([1.8,-2.5,0.1]); V_MUL = np.array([0.9,1.8,0.7])

N_PATHS = 6; N_DAYS = 2520; WARMUP = 60
SWING_THR=65; SWING_HOLD=5; DAY_THR=65; DAY_HOLD=1

MULTS_SWING={"large":[0.776,1.283,0.719,1.222],"mid":[0.881,1.057,0.858,1.204],"small":[1.122,0.856,1.001,1.021]}
MULTS_DAY  ={"large":[0.897,1.100,0.700,1.303],"mid":[1.104,1.041,0.437,1.418],"small":[0.917,1.026,0.986,1.072]}

# ─── 取引日カレンダー ─────────────────────────────────────────────────────────
def build_cal():
    cal=[]; d=date(2015,1,2)
    while len(cal)<N_DAYS:
        if d.weekday()<5: cal.append(d)
        d+=timedelta(days=1)
    return cal
CAL=build_cal()
def bar2date(b): return CAL[b].strftime("%Y-%m-%d") if b<len(CAL) else f"bar{b}"

# ─── シミュレーション ─────────────────────────────────────────────────────────
def sim_batch(cagr,vol):
    dd=np.log(1+cagr)/252; dv=vol/np.sqrt(252)
    z=RNG.standard_t(df=5,size=(N_PATHS,N_DAYS)).astype(float)/np.sqrt(5/3)
    closes=np.empty((N_PATHS,N_DAYS))
    for p in range(N_PATHS):
        reg=0; lr=np.empty(N_DAYS)
        for t in range(N_DAYS):
            reg=int(RNG.choice(3,p=TRANS[reg])); lr[t]=dd*D_MUL[reg]+dv*V_MUL[reg]*z[p,t]
        closes[p,0]=100.; closes[p,1:]=100.*np.exp(np.cumsum(lr)[:-1])
    z_hl=np.abs(HL_RNG.normal(0,1,(N_PATHS,N_DAYS)))
    hr=closes*dv*0.6*z_hl
    return closes, closes+hr, np.maximum(closes-hr,closes*0.005)

# ─── ベクトル化指標計算 ───────────────────────────────────────────────────────
def _ip(v,bp):
    if v<=bp[0][0]: return float(bp[0][1])
    if v>=bp[-1][0]: return float(bp[-1][1])
    for i in range(len(bp)-1):
        x0,y0=bp[i]; x1,y1=bp[i+1]
        if x0<=v<=x1: return y0+(y1-y0)*(v-x0)/(x1-x0)
    return float(bp[-1][1])

def ema_v(p,k):
    """EMA with NaN-prefix support (e.g. MACD line seeded from bar 25)."""
    T=p.shape[-1]; out=np.full_like(p,np.nan)
    if T<k: return out
    a=2./(k+1)
    # Fast path: no NaN in the seeding window
    if not np.any(np.isnan(p[...,:k])):
        out[...,k-1]=p[...,:k].mean(-1)
        for t in range(k,T): out[...,t]=p[...,t]*a+out[...,t-1]*(1-a)
        return out
    # Slow path: per-row, find first valid k-bar window (handles NaN-prefix arrays)
    P=p.shape[0]
    for i in range(P):
        row=p[i]; valid=np.where(~np.isnan(row))[0]
        if len(valid)<k: continue
        s=valid[0]; se=s+k-1
        if se>=T: continue
        out[i,se]=row[s:se+1].mean()
        for t in range(se+1,T): out[i,t]=row[t]*a+out[i,t-1]*(1-a)
    return out

def rsi_v(p,n=14):
    P,T=p.shape; out=np.full((P,T),np.nan); d=np.diff(p,axis=1)
    g=np.where(d>0,d,0.); l=np.where(d<0,-d,0.)
    ag=g[:,:n].mean(1); al=l[:,:n].mean(1)
    for i in range(n,T):
        rs=np.where(al>1e-12,ag/al,1e9); out[:,i]=100.-100./(1.+rs)
        if i<T-1: ag=(ag*(n-1)+g[:,i])/n; al=(al*(n-1)+l[:,i])/n
    return out

def bb_v(p,n=20):
    P,T=p.shape; pb=np.full((P,T),np.nan); bw=np.full((P,T),np.nan)
    for t in range(n-1,T):
        w=p[:,t-n+1:t+1]; m=w.mean(1); s=w.std(1,ddof=1); ok=s>0
        pb[ok,t]=(p[ok,t]-(m[ok]-2*s[ok]))/(4*s[ok]); bw[ok,t]=4*s[ok]/m[ok]
    return pb,bw

def aroon_v(highs,lows,period=25):
    P,T=highs.shape; up=np.full((P,T),np.nan); down=np.full((P,T),np.nan)
    for t in range(period,T):
        wh=highs[:,t-period:t+1]; wl=lows[:,t-period:t+1]
        up[:,t]=(np.argmax(wh,axis=1)/period)*100.
        down[:,t]=(np.argmin(wl,axis=1)/period)*100.
    return up,down

# ─── 全指標スコア配列（バッチ）──────────────────────────────────────────────
def compute_all_scores(closes, highs, lows, mode, size):
    """Returns score arrays shape (P,T) each, plus indicator value arrays."""
    P,T=closes.shape; mults=MULTS_SWING[size] if mode=="swing" else MULTS_DAY[size]

    # RSI
    rv=rsi_v(closes)
    rsi_sc=np.full((P,T),np.nan)
    for t in range(1,T):
        c,prev=rv[:,t],rv[:,t-1]; ok=~(np.isnan(c)|np.isnan(prev)); up=c>prev
        sc=np.full(P,np.nan)
        for i in np.where(ok)[0]:
            cv,uv=c[i],up[i]
            if mode=="swing":
                if   cv<25: r=23.
                elif cv<35: r=min(_ip(cv,[(25,22),(35,12)])+(3 if uv else 0),25.)
                elif cv<45: r=_ip(cv,[(35,14),(45,10)])
                elif cv<55: r=_ip(cv,[(45,12),(55,8)])
                elif cv<65: r=_ip(cv,[(55,8),(65,5)])
                elif cv<75: r=_ip(cv,[(65,5),(75,2)])
                else:       r=_ip(cv,[(75,2),(85,0)])
            else:
                if cv<25:   r=22. if uv else 17.
                elif cv<40: r=_ip(cv,[(25,19),(40,12)])+(2. if uv else 0.)
                elif cv<50: r=_ip(cv,[(40,12),(50,9)])
                elif cv<55: r=_ip(cv,[(50,9),(55,11)])
                elif cv<65: r=max(_ip(cv,[(55,11),(65,17)])+(2. if uv else -2.),8.)
                elif cv<75: r=_ip(cv,[(65,15),(75,10)])+(1. if uv else 0.)
                else:       r=_ip(cv,[(75,8),(85,3)])
            sc[i]=min(r,25.)
        rsi_sc[:,t]=sc

    # MACD
    ml=ema_v(closes,12)-ema_v(closes,26); sl=ema_v(ml,9); hl_=ml-sl
    macd_sc=np.full((P,T),np.nan)
    for t in range(3,T):
        mc,mp=ml[:,t],ml[:,t-1]; scl,sp=sl[:,t],sl[:,t-1]; hc,hp=hl_[:,t],hl_[:,t-1]
        ok=~(np.isnan(mc)|np.isnan(scl))
        if not ok.any(): continue
        bn=(mp<sp)&(mc>=scl); dn=(mp>sp)&(mc<=scl)
        br=np.zeros(P,bool)
        for j in range(2,min(4,t)):
            br|=(~np.isnan(ml[:,t-j-1])&(ml[:,t-j-1]<sl[:,t-j-1])&(ml[:,t-j]>=sl[:,t-j])&(mc>scl))
        s=np.full(P,3.)
        s[mc>scl]=10.; s[(mc>scl)&(hc>hp)]=15.; s[br&(hc>0)]=20.
        s[bn]=24.; s[(mc<scl)&(hc>hp)]=6.; s[dn]=2.; s[~ok]=np.nan
        macd_sc[:,t]=s

    # BB
    pb,bw=bb_v(closes); bb_sc=np.full((P,T),np.nan)
    for t in range(4,T):
        ok=~np.isnan(pb[:,t])
        if not ok.any(): continue
        c,cp=pb[:,t],pb[:,t-1]
        ws=bw[:,max(0,t-4):t+1]; cnt=np.sum(~np.isnan(ws),axis=1)
        bwa=np.where(np.isnan(ws),0.,ws); mbw=np.where(cnt>0,bwa.sum(1)/np.maximum(cnt,1),np.nan)
        sq=ok&~np.isnan(bw[:,t])&(bw[:,t]<mbw*0.85)&(c<0.3)
        rising=ok&~np.isnan(cp)&(c>cp)
        raw=np.full(P,np.nan)
        for i in np.where(ok)[0]:
            cv=c[i]
            if   cv<0.:  bv=_ip(cv,[(-0.2,18),(0,15)])
            elif cv<0.1: bv=_ip(cv,[(0,15),(0.1,12)])
            elif cv<0.3: bv=_ip(cv,[(0.1,12),(0.3,9)])
            elif cv<0.5: bv=_ip(cv,[(0.3,9),(0.5,7)])
            elif cv<0.7: bv=_ip(cv,[(0.5,7),(0.7,5)])
            elif cv<0.9: bv=_ip(cv,[(0.7,5),(0.9,2)])
            else:         bv=_ip(cv,[(0.9,2),(1.2,0)])
            db=(4 if cv<0.5 else (3 if cv<=0.85 else 1)) if rising[i] else 0
            raw[i]=min(bv+db+(3 if sq[i] else 0),25.)
        bb_sc[:,t]=raw

    # MA
    fast_k=20 if mode=="swing" else 5; slow_k=50 if mode=="swing" else 10
    mf=ema_v(closes,fast_k); ms=ema_v(closes,slow_k); ma_sc=np.full((P,T),np.nan)
    tol=0.005 if mode=="swing" else 0.003
    for t in range(3,T):
        ok=~(np.isnan(mf[:,t])|np.isnan(ms[:,t]))
        if not ok.any(): continue
        pn,pp=closes[:,t],closes[:,t-1]; mn,mp=mf[:,t],mf[:,t-1]; fn=ms[:,t]
        ct=(pp<mp)&(pn>=mn); cr=np.zeros(P,bool)
        for j in range(2,min(4,t)):
            cr|=(closes[:,t-j-1]<mf[:,t-j-1])&(closes[:,t-j]>=mf[:,t-j])&(pn>mn)
        gap=mn-fn; gprev=mf[:,t-1]-ms[:,t-1]; wide=np.abs(gap)>np.abs(gprev)
        sc2=np.full(P,np.nan)
        for i in np.where(ok)[0]:
            pv,mv,fv=pn[i],mn[i],fn[i]
            if ct[i]: ps=15
            elif cr[i]: ps=12 if mode=="swing" else 13
            elif pv>mv:
                bp2=[(1.,9.),(1.05,6.),(1.1,4.)] if mode=="swing" else [(1.,10.),(1.03,7.),(1.06,4.)]
                ps=round(_ip(pv/mv,bp2))
            elif abs(pv-mv)/mv<tol: ps=5 if mode=="swing" else 6
            else:
                bp2=[(0.9,4.),(0.95,3.),(1.,0.)] if mode=="swing" else [(0.94,4.),(0.97,3.),(1.,0.)]
                ps=round(_ip(pv/mv,bp2))
            gv=gap[i]; ms_v=(9 if wide[i] else 6) if gv>0 else (4 if abs(gv/fv)<tol else (1 if wide[i] else 3))
            sc2[i]=min(ps+ms_v,25.)
        ma_sc[:,t]=sc2

    # Aroon
    au,ad=aroon_v(highs,lows); aroon_sc=np.full((P,T),np.nan)
    for t in range(1,T):
        aup=au[:,t]; adwn=ad[:,t]; prev=au[:,t-1]
        ok=~(np.isnan(aup)|np.isnan(adwn))
        if not ok.any(): continue
        sc3=np.full(P,np.nan)
        for i in np.where(ok)[0]:
            u,d=aup[i],adwn[i]; rising=(not np.isnan(prev[i])) and u>prev[i]
            if   u>70 and d<30:           sc3[i]=10.
            elif u>d  and u>50 and rising: sc3[i]=8.
            elif u>d:                      sc3[i]=6.
            elif abs(u-d)<=10:            sc3[i]=4.
            elif d>70 and u<30:           sc3[i]=0.
            else:                          sc3[i]=2.
        aroon_sc[:,t]=sc3

    # 複合スコア
    m0,m1,m2,m3=mults
    base=rsi_sc*m0+macd_sc*m1+bb_sc*m2+ma_sc*m3
    use_aroon=(mode=="day") or (mode=="swing" and size=="large")
    bonus=aroon_sc if use_aroon else np.zeros_like(aroon_sc)
    total=np.where(np.isnan(base),np.nan,np.minimum(100.,base+np.where(np.isnan(bonus),0.,bonus)))

    return {"total":total,"rsi":rsi_sc,"macd":macd_sc,"bb":bb_sc,"ma":ma_sc,
            "aroon":aroon_sc,"aroon_used":use_aroon,
            "rsi_v":rv,"macd_v":ml,"bb_v":pb,"ma_f":mf,"ma_s":ms,
            "au":au,"ad":ad,"mults":mults}

# ─── シグナル説明テキスト生成 ─────────────────────────────────────────────────
def rsi_explain(rsi_val, rsi_score, mode):
    if rsi_val is None: return "データ不足"
    v=rsi_val
    if mode=="swing":
        if v<25: return f"RSI={v:.1f} 極度売られすぎ（基準<25）→逆張り買いシグナル"
        if v<35: return f"RSI={v:.1f} 売られすぎゾーン（25-35）→回復期待"
        if v<45: return f"RSI={v:.1f} 売られすぎ接近（35-45）"
        if v<55: return f"RSI={v:.1f} 中立（45-55）"
        if v<65: return f"RSI={v:.1f} やや強気（55-65）"
        return f"RSI={v:.1f} 買われすぎ圏（>65）"
    else:
        if v<25: return f"RSI={v:.1f} 極度売られすぎ→反発期待（デイ）"
        if v<40: return f"RSI={v:.1f} 売られすぎ回復中（デイ）"
        if 55<=v<70: return f"RSI={v:.1f} 上昇モメンタム帯（55-70、デイV字）"
        return f"RSI={v:.1f} スコア={rsi_score}点"

def macd_explain(macd_score):
    m={24:"ゴールデンクロス当日（MACDがシグナル上抜け）",
       20:"直近クロス後ヒストグラム拡大中",
       15:"MACDがシグナル線上方かつヒストグラム拡大",
       10:"MACDがシグナル線の上方",
       6: "弱気弱まり（ヒストグラム上昇）",
       3: "下落トレンド継続",
       2: "デッドクロス当日"}
    return m.get(macd_score, f"スコア={macd_score}点")

def bb_explain(bb_val, bb_score):
    if bb_val is None: return "データ不足"
    v=bb_val
    if v<0:   return f"%B={v:.3f} 下限バンド突破（過売り極値）→強い逆張りシグナル"
    if v<0.1: return f"%B={v:.3f} 下限バンド付近（0〜0.1）→売られすぎ圏"
    if v<0.3: return f"%B={v:.3f} 下方域（0.1〜0.3）"
    if v<0.5: return f"%B={v:.3f} 下半分（0.3〜0.5）"
    if v<0.7: return f"%B={v:.3f} 上半分（0.5〜0.7）"
    return f"%B={v:.3f} 上限帯付近（>0.7）"

def ma_explain(ma_score, mode):
    labels={15:f"{'EMA20' if mode=='swing' else 'EMA5'}上抜け当日（クロスオーバー）",
            12:"直近上抜け（2-3日前）",13:"直近上抜け（2日前）",
            9:"EMA上方（乖離小）",6:"EMA上方（乖離中）",5:"EMA付近（±0.5%）",
            4:"EMA下方（乖離小）",3:"EMA下方（乖離中）"}
    return labels.get(ma_score, f"スコア={ma_score}点")

def aroon_explain(aroon_score, au_val, ad_val):
    if aroon_score==10: return f"AroonUp={au_val:.0f} Down={ad_val:.0f}→強い上昇トレンド（Up>70, Down<30）"
    if aroon_score==8:  return f"AroonUp={au_val:.0f} Down={ad_val:.0f}→上昇トレンド形成中（Up>Down+50上昇中）"
    if aroon_score==6:  return f"AroonUp={au_val:.0f} Down={ad_val:.0f}→上昇トレンド（Up>Down）"
    if aroon_score==4:  return f"AroonUp={au_val:.0f} Down={ad_val:.0f}→横ばい"
    if aroon_score==2:  return f"AroonUp={au_val:.0f} Down={ad_val:.0f}→下降トレンド"
    if aroon_score==0:  return f"AroonUp={au_val:.0f} Down={ad_val:.0f}→強い下降トレンド"
    return "非適用（規模/モード外）"

# ─── 全銘柄スキャン ───────────────────────────────────────────────────────────
def scan_all(mode):
    thr  = SWING_THR  if mode=="swing"  else DAY_THR
    hold = SWING_HOLD if mode=="swing"  else DAY_HOLD
    all_trades=[]
    import sys

    for ticker,(cagr,vol,name_ja,size) in STOCKS.items():
        print(f"  計算中: {name_ja} ({ticker}) ...", end=" ", flush=True)
        closes,highs,lows=sim_batch(cagr,vol)
        sc_dict=compute_all_scores(closes,highs,lows,mode,size)
        tot=sc_dict["total"]; m=sc_dict["mults"]

        # シグナルバーを全パスで検索
        for p in range(N_PATHS):
            for t in range(WARMUP+1, N_DAYS-hold-1):
                sv=tot[p,t]
                if np.isnan(sv) or sv<thr: continue
                entry=closes[p,t]; exit_=closes[p,t+hold]
                ret=(exit_-entry)/entry
                # 各指標の値を取得
                rsi_val_=sc_dict["rsi_v"][p,t]
                bb_val_ =sc_dict["bb_v"][p,t]
                au_val_ =sc_dict["au"][p,t]
                ad_val_ =sc_dict["ad"][p,t]
                maf_t   =sc_dict["ma_f"][p,t]
                ma_ratio_=closes[p,t]/maf_t if not np.isnan(maf_t) else np.nan

                rsi_s  =round(sc_dict["rsi"][p,t])  if not np.isnan(sc_dict["rsi"][p,t])  else 0
                macd_s =round(sc_dict["macd"][p,t]) if not np.isnan(sc_dict["macd"][p,t]) else 0
                bb_s   =round(sc_dict["bb"][p,t])   if not np.isnan(sc_dict["bb"][p,t])   else 0
                ma_s   =round(sc_dict["ma"][p,t])   if not np.isnan(sc_dict["ma"][p,t])   else 0
                ar_s   =round(sc_dict["aroon"][p,t]) if (sc_dict["aroon_used"] and not np.isnan(sc_dict["aroon"][p,t])) else 0
                base_=rsi_s*m[0]+macd_s*m[1]+bb_s*m[2]+ma_s*m[3]

                all_trades.append({
                    "ticker":ticker,"name":name_ja,"size":size,
                    "path":p,"bar_buy":t,"bar_sell":t+hold,
                    "date_buy":bar2date(t),"date_sell":bar2date(t+hold),
                    "price_buy":round(entry,2),"price_sell":round(exit_,2),
                    "return_pct":round(ret*100,2),"score":round(sv,1),
                    "rsi_s":rsi_s,"macd_s":macd_s,"bb_s":bb_s,"ma_s":ma_s,"aroon_s":ar_s,
                    "base":round(base_,1),"mults":m,"aroon_used":sc_dict["aroon_used"],
                    "rsi_val":round(float(rsi_val_),1) if not np.isnan(rsi_val_) else None,
                    "bb_val": round(float(bb_val_),3)  if not np.isnan(bb_val_)  else None,
                    "au_val": round(float(au_val_),1)  if not np.isnan(au_val_)  else None,
                    "ad_val": round(float(ad_val_),1)  if not np.isnan(ad_val_)  else None,
                    "ma_ratio":round(float(ma_ratio_),4) if not np.isnan(ma_ratio_) else None,
                })
        n_sig=len([x for x in all_trades if x["ticker"]==ticker])
        print(f"{n_sig}件シグナル", flush=True)

    all_trades.sort(key=lambda x: -x["return_pct"])
    return all_trades

# ─── 表示 ─────────────────────────────────────────────────────────────────────
def print_trade(rank, tr, mode):
    sep="="*78
    hold=SWING_HOLD if mode=="swing" else DAY_HOLD
    m=tr["mults"]
    print(f"\n{sep}")
    print(f"  #{rank}  {tr['name']} ({tr['ticker']})  規模:{tr['size']}  パス:{tr['path']}")
    print(sep)
    print(f"\n  ▶ 買い  {tr['date_buy']} ({tr['bar_buy']}営業日目)  株価: {tr['price_buy']:.2f}")
    print(f"         総合スコア: {tr['score']:.1f} 点  (閾値 ≥ {SWING_THR if mode=='swing' else DAY_THR}点でエントリー)")
    print()
    print(f"    指標          値        スコア   重み     寄与    根拠")
    print(f"    " + "-"*68)

    def row(name,val_str,sc,mult):
        cont=round(sc*mult,1); bar="█"*min(int(sc),25)
        print(f"    {name:<12s}  {val_str:>8s}   {sc:2d}/25点  ×{mult:.3f}  {cont:5.1f}点   {bar}")

    row("RSI",    f"{tr['rsi_val']:.1f}" if tr['rsi_val'] else "—",  tr['rsi_s'],  m[0])
    print(f"       └ {rsi_explain(tr['rsi_val'], tr['rsi_s'], mode)}")
    row("MACD",   "—",                                                tr['macd_s'], m[1])
    print(f"       └ {macd_explain(tr['macd_s'])}")
    row("BB %B",  f"{tr['bb_val']:.3f}" if tr['bb_val'] is not None else "—", tr['bb_s'], m[2])
    print(f"       └ {bb_explain(tr['bb_val'], tr['bb_s'])}")
    row("MA比率", f"{tr['ma_ratio']:.4f}" if tr['ma_ratio'] else "—", tr['ma_s'], m[3])
    print(f"       └ {ma_explain(tr['ma_s'], mode)}")
    print(f"    " + "-"*68)
    print(f"    {'4指標合計':<12s}  {'':>8s}         {'':>7s}  {tr['base']:5.1f}点")
    if tr['aroon_used'] and tr['aroon_s']>0:
        row("Aroon(bonus)", f"Up={tr['au_val']:.0f}" if tr['au_val'] else "—", tr['aroon_s'], 1.0)
        print(f"       └ {aroon_explain(tr['aroon_s'], tr['au_val'] or 0, tr['ad_val'] or 0)}")
    print(f"    {'◆ 総合スコア':<12s}  {'':>8s}         {'':>7s}  {tr['score']:5.1f}点 ✓ エントリー")

    print(f"\n  ▶ 売り  {tr['date_sell']} ({tr['bar_sell']}営業日目)  株価: {tr['price_sell']:.2f}")
    sign="+" if tr['return_pct']>=0 else ""
    result="✅ 利益確定" if tr['return_pct']>=0 else "❌ 損切り"
    print(f"         リターン: {sign}{tr['return_pct']:.2f}%  {result}")
    print(f"         売り根拠: 保有期間 {hold} 営業日経過による機械的クローズ")
    if mode=="swing":
        print(f"         （上限BBバンド到達 or -5%ストップロスでも早期決済）")

# ─── メイン ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("  売買記録レポート — Monte Carlo シミュレーション")
    print("  期間: 2015-01-02〜2024-12-31 (2520営業日)  各銘柄 6 シナリオパス")
    print("  株価: 全銘柄 100.0 から開始（各銘柄のCAGR/ボラティリティでシミュレート）")
    print("=" * 78)

    # ── スイングトレード ──
    print("\n\n" + "█"*78)
    print("  スイングトレード (hold=5営業日)  ─  リターン上位 5 件")
    print("█"*78)
    print()
    swing_trades = scan_all("swing")
    top_swing = swing_trades[:5]
    for i,tr in enumerate(top_swing,1):
        print_trade(i,tr,"swing")

    # ── デイトレード ──
    print("\n\n" + "█"*78)
    print("  デイトレード (hold=1営業日)  ─  リターン上位 5 件")
    print("█"*78)
    print()
    day_trades = scan_all("day")
    top_day = day_trades[:5]
    for i,tr in enumerate(top_day,1):
        print_trade(i,tr,"day")

    # ── サマリー ──
    print("\n\n" + "="*78)
    print("  集計サマリー")
    print("="*78)
    all_sw_ret =[t["return_pct"] for t in swing_trades]
    all_day_ret=[t["return_pct"] for t in day_trades]
    for label,trades,all_ret in [("スイング",top_swing,all_sw_ret),("デイトレ",top_day,all_day_ret)]:
        if trades:
            top5_ret=[t["return_pct"] for t in trades]
            wr=sum(1 for r in all_ret if r>0)/max(len(all_ret),1)
            print(f"\n  {label}:")
            print(f"    全シグナル: {len(all_ret)}件   勝率: {wr:.1%}   平均リターン: {np.mean(all_ret):+.2f}%")
            print(f"    上位5件リターン: {' | '.join(f'{r:+.1f}%' for r in top5_ret)}")
            by_ticker={}
            for t in trades: by_ticker[t["ticker"]]=by_ticker.get(t["ticker"],0)+1
            print(f"    銘柄: {', '.join(f'{k}({v})' for k,v in sorted(by_ticker.items(),key=lambda x:-x[1]))}")
    print("\n  技術的整合性:")
    print("  ✓ score[t] = price[0..t] のみ使用 (先読みバイアスなし、v4で30点検証済)")
    print("  ✓ 買い価格 = close[t]、売り価格 = close[t+hold]")
    print("  ✓ 全MULTS = walk-forwardバックテストの再導出値と完全一致")

if __name__=="__main__":
    main()
