"""Discord notification via webhook (daily heartbeat + urgent alerts).

Setup: Discord server -> サーバー設定 -> 連携サービス -> ウェブフック ->
新しいウェブフック -> URLをコピー -> .env に DISCORD_WEBHOOK_URL=... を追記。

Design rules:
  * notification failure must NEVER break the trading run (best-effort)
  * no webhook configured -> silent no-op (returns False)
  * messages truncated to Discord's 2000-char limit
"""

from __future__ import annotations

import os

import httpx

from data.config_loader import load_dotenv_if_present

DISCORD_LIMIT = 2000


def send_discord(message: str, timeout: float = 10.0) -> bool:
    """Post ``message`` to the configured webhook. Returns True on success."""
    load_dotenv_if_present()
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not url:
        return False
    try:
        r = httpx.post(url, json={"content": message[:DISCORD_LIMIT]},
                       timeout=timeout)
        return r.status_code in (200, 204)
    except Exception as e:  # noqa: BLE001 - never break the caller
        print(f"WARN discord notify failed: {e}")
        return False


def format_run_summary(out: dict) -> str:
    """Human-readable daily summary for a live_runner run_once() result."""
    status = out.get("status")
    if status == "warmup":
        return (f"⏳ trading_core: ウォームアップ中 "
                f"({out.get('bars')}/{out.get('needed')}本)")
    if status == "already_decided":
        return (f"➖ trading_core {out.get('bar_date')}: 判定済み（重複実行・"
                f"操作不要）")

    icon = {"buy": "📈", "sell": "📉", "hold": "➖", "halted": "🛑",
            "KILL_SWITCH_LIQUIDATE": "🚨"}.get(out.get("action", ""), "ℹ️")
    lines = [
        f"{icon} trading_core {out.get('bar_date')}",
        f"レジーム: {out.get('regime')} / アクション: {out.get('action')}",
        f"資産: ${out.get('equity', 0):,.0f} / 保有QQQ: {out.get('qty', 0):,.1f}株"
        f" / DD: {out.get('dd', 0):.1%}",
    ]
    for f in out.get("fills", []):
        side = "買" if f["side"] > 0 else "売"
        lines.append(f"約定: {side} {f['qty']:.1f}株 @ ${f['price']:,.2f}"
                     f" (手数料 ${f['fee']:.2f})")
    if out.get("halted"):
        lines.append("🚨 **キルスイッチ発動中 — システム停止。要確認** 🚨")
    return "\n".join(lines)
