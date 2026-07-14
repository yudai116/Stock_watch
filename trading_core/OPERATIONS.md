# OPERATIONS.md — 日次運用手順（Phase 5: ペーパートレード）

> 本番戦略: **3d レジームオーバーレイ**（`reports/phase3_final_adjudication.md`）
> ルール: bull=QQQに95%投資 / range=保持 / bear・crisis=現金化 / 調整パラメータ0個

## 毎日やること（自動化推奨・1コマンド）

米国市場の引け後（日本時間の朝）に1回:

```powershell
cd C:\projects\projects\auto-trader\Stock_watch\trading_core
uv run python -m execution.live_runner run --mode paper
```

これ1つで「QQQ日足とVIXの取得 → 前日に出した注文を今日の寄付き価格で約定 →
レジーム判定 → 必要なら翌日執行の注文を作成 → 帳簿記録」まで全部走る。
土日・祝日に動かしても無害（新しいバーがなければ何もしない）。

### Windows タスクスケジューラ登録（1回だけ）

管理者PowerShellで（毎朝7:00 JST。夏時間期は6:00でも可）:

```powershell
schtasks /create /tn "trading_core_paper" /sc daily /st 07:00 /tr "powershell -NoProfile -Command \"cd C:\projects\projects\auto-trader\Stock_watch\trading_core; uv run python -m execution.live_runner run --mode paper >> reports\paper_log.txt 2>&1\""
```

## Discord通知（毎朝の実行結果が届く）

1. Discordアプリ → 通知を受け取りたいサーバー → **サーバー設定 → 連携サービス →
   ウェブフック → 新しいウェブフック** → チャンネルを選び **URLをコピー**
2. `.env` に1行追加:
   ```
   DISCORD_WEBHOOK_URL=コピーしたURL
   ```
3. テスト送信:
   ```powershell
   uv run python -m execution.live_runner test-notify
   ```
   Discordに「✅ テスト」が届けば完了。以後、毎朝の自動実行ごとに
   レジーム・アクション・資産・約定が1通届く（＝ハートビート。
   **朝に通知が来ない日はスケジューラが動かなかったサイン**）。
   キルスイッチ発動時は 🚨、実行エラー時は ❌ が届く。

## 状態確認・レポート

```powershell
uv run python -m execution.live_runner status    # 現在のポジション・現金・停止状態
uv run python -m execution.live_runner report    # 実測Sharpe/CAGR/DD vs バックテスト期待値
```

## キルスイッチ（addendum G-3）

- ペーパー資産のドローダウンが **-23.5%**（検証MaxDD 18.8%×1.25）に達すると、
  自動で全ポジション解消 → **システム停止**（以後は新規売買しない）
- 停止したら原因分析の上、再開は手動でのみ:
  `uv run python -m execution.live_runner reset-killswitch`

## 実弾までのゲート（順序厳守・addendum G）

1. **ペーパー3か月以上**継続
   - **追加条件（敵対的レビュー 2026-07）**: 期間中にレジーム転換（bull→bear
     または逆）を最低1回経験していること。強気だけの3か月は「QQQを持って
     いただけ」でありベア防御（本戦略の存在意義）の検証にならない。
     転換が来なければペーパーを延長する
2. `report` の実測値がバックテスト期待値（Sharpe 0.99 / CAGR 15.5% / DD -11.7%）
   に対して大きく乖離していないこと。**実測期待値が50%未満なら実弾禁止**
   → 乖離原因（スリッページ/シグナル遅延/データ差）を特定して戻る
3. Saxo接続（`--mode saxo`、要 SAXO_APP_KEY/SECRET・SIM環境）で並走確認
4. 実弾は**運用予定資金の25%から**。四半期ごとに実測をBT分布と照合
   （下位10%で増額停止・下位5%で縮小）
5. LIVE環境での発注は `--allow-live` を明示しない限り拒否される設計

## トラブル時

- `WARN bar ingest failed` → Alpacaキー/ネットを確認。最後に保存済みのデータで
  判定は続行される（1日程度の欠落は挙動に影響しない）
- 状態が壊れた疑い → `datastore/live_state.sqlite` がペーパー帳簿の正本。
  バックアップしてから調査すること
