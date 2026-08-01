"""Discord notifier: formatting, no-op without webhook, never-break guarantee."""

import httpx
import pytest

import execution.notifier as nt
from execution.notifier import format_run_summary, send_discord


def test_noop_without_webhook(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(nt, "load_dotenv_if_present", lambda: None)
    assert send_discord("hello") is False


def test_send_posts_truncated(monkeypatch):
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent["url"] = url
        sent["content"] = json["content"]
        return httpx.Response(204, request=httpx.Request("POST", url))

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/hook")
    monkeypatch.setattr(nt, "load_dotenv_if_present", lambda: None)
    monkeypatch.setattr(nt.httpx, "post", fake_post)
    assert send_discord("x" * 5000) is True
    assert len(sent["content"]) == 2000            # Discord limit respected


def test_send_never_raises(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/hook")
    monkeypatch.setattr(nt, "load_dotenv_if_present", lambda: None)
    monkeypatch.setattr(nt.httpx, "post", boom)
    assert send_discord("hello") is False          # swallowed, not raised


def test_format_buy_with_fill():
    msg = format_run_summary({
        "status": "ok", "bar_date": "2026-07-09", "regime": "bull",
        "action": "buy", "equity": 100000.0, "qty": 0.0, "dd": 0.0,
        "halted": False,
        "fills": [{"side": 1, "qty": 170.2, "price": 558.12, "fee": 76.0}],
    })
    assert "📈" in msg and "bull" in msg and "買 170.2株" in msg
    assert "キルスイッチ" not in msg


def test_format_kill_switch_alert():
    msg = format_run_summary({
        "status": "ok", "bar_date": "2026-07-09", "regime": "bear",
        "action": "KILL_SWITCH_LIQUIDATE", "equity": 76000.0, "qty": 150.0,
        "dd": 0.24, "halted": True, "fills": [],
    })
    assert "🚨" in msg and "キルスイッチ発動中" in msg


def test_format_warmup_and_duplicate():
    assert "ウォームアップ" in format_run_summary(
        {"status": "warmup", "bars": 50, "needed": 210})
    assert "判定済み" in format_run_summary(
        {"status": "already_decided", "bar_date": "2026-07-09", "fills": []})
