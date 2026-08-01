"""Saxo Bank OpenAPI client (D2, revised by SPEC_ADDENDUM_v2 R1):
OAuth code flow via localhost:8765 callback, token refresh, and order
placement for BOTH asset types:

  * ``Stock``      — cash equity (all long stock positions; no financing)
  * ``CfdOnIndex`` — index-tracker CFD (bear-regime QQQ/SMH hedge shorts)

Phase 5 runs this against the SIM environment for >= 3 months of paper
trading (addendum G) before any live order.

Environment: SAXO_APP_KEY / SAXO_APP_SECRET / SAXO_ENV / SAXO_REDIRECT_URI in .env
"""

from __future__ import annotations

import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

from data.config_loader import data_root, load_dotenv_if_present

ENDPOINTS = {
    "sim": {
        "auth": "https://sim.logonvalidation.net/authorize",
        "token": "https://sim.logonvalidation.net/token",
        "api": "https://gateway.saxobank.com/sim/openapi",
    },
    "live": {
        "auth": "https://live.logonvalidation.net/authorize",
        "token": "https://live.logonvalidation.net/token",
        "api": "https://gateway.saxobank.com/openapi",
    },
}


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    code: str | None = None
    expected_state: str = ""

    def do_GET(self):  # noqa: N802
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if q.get("state", [""])[0] != self.expected_state:
            self.send_response(400)
            self.end_headers()
            return
        _CallbackHandler.code = q.get("code", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Login complete. You can close this tab.")

    def log_message(self, *a):  # silence
        pass


class SaxoClient:
    def __init__(self, env: str | None = None):
        load_dotenv_if_present()
        self.env = env or os.environ.get("SAXO_ENV", "sim")
        self.app_key = os.environ["SAXO_APP_KEY"]
        self.app_secret = os.environ["SAXO_APP_SECRET"]
        self.redirect_uri = os.environ.get("SAXO_REDIRECT_URI", "http://localhost:8765/callback")
        self.eps = ENDPOINTS[self.env]
        self.token_path = Path(data_root()) / f"saxo_token_{self.env}.json"
        self._token: dict | None = None
        self._http = httpx.Client(timeout=30)

    # ------------------------------------------------------------- oauth

    def authorize_url(self, state: str) -> str:
        params = {
            "response_type": "code",
            "client_id": self.app_key,
            "redirect_uri": self.redirect_uri,
            "state": state,
        }
        return f"{self.eps['auth']}?{urllib.parse.urlencode(params)}"

    def login(self) -> None:
        """Interactive OAuth: opens browser, catches code on localhost:8765."""
        state = secrets.token_urlsafe(16)
        _CallbackHandler.expected_state = state
        _CallbackHandler.code = None
        port = int(urllib.parse.urlparse(self.redirect_uri).port or 8765)
        server = http.server.HTTPServer(("localhost", port), _CallbackHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            webbrowser.open(self.authorize_url(state))
            deadline = time.time() + 300
            while _CallbackHandler.code is None and time.time() < deadline:
                time.sleep(0.25)
        finally:
            server.shutdown()
        if _CallbackHandler.code is None:
            raise TimeoutError("OAuth callback not received within 5 minutes")
        self._exchange_token({"grant_type": "authorization_code",
                              "code": _CallbackHandler.code,
                              "redirect_uri": self.redirect_uri})

    def _exchange_token(self, payload: dict) -> None:
        r = self._http.post(self.eps["token"], data=payload,
                            auth=(self.app_key, self.app_secret))
        r.raise_for_status()
        tok = r.json()
        tok["expires_at"] = time.time() + int(tok.get("expires_in", 1200)) - 60
        self._token = tok
        self.token_path.write_text(json.dumps(tok))

    def _access_token(self) -> str:
        if self._token is None and self.token_path.exists():
            self._token = json.loads(self.token_path.read_text())
        if self._token is None:
            raise RuntimeError("not logged in: call login()")
        if time.time() >= self._token["expires_at"]:
            self._exchange_token({"grant_type": "refresh_token",
                                  "refresh_token": self._token["refresh_token"],
                                  "redirect_uri": self.redirect_uri})
        return self._token["access_token"]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._access_token()}"}

    # --------------------------------------------------------------- api

    def get(self, path: str, params: dict | None = None) -> dict:
        r = self._http.get(f"{self.eps['api']}{path}", params=params, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def account_key(self) -> str:
        return self.get("/port/v1/accounts/me")["Data"][0]["AccountKey"]

    def find_uic(self, symbol: str, asset_type: str = "Stock") -> int:
        """asset_type: "Stock" (cash equity) or "CfdOnIndex"/"CfdOnEtf"
        (hedge). QQQ/SMH trade as ETFs, so the CFD lookup tries CfdOnEtf
        as a fallback for CfdOnIndex."""
        tries = [asset_type]
        if asset_type == "CfdOnIndex":
            tries.append("CfdOnEtf")
        for at in tries:
            data = self.get("/ref/v1/instruments",
                            {"Keywords": symbol, "AssetTypes": at})
            for item in data.get("Data", []):
                if item.get("Symbol", "").split(":")[0].upper() == symbol.upper():
                    return int(item["Identifier"])
        raise LookupError(f"no {asset_type} instrument for {symbol}")

    def place_market_order(self, symbol: str, side: int, qty: float,
                           asset_type: str = "Stock",
                           stop_price: float | None = None) -> dict:
        """side: +1 buy, -1 sell. asset_type "Stock" for cash longs,
        "CfdOnIndex" for the hedge. Optionally attaches a stop-loss order."""
        uic = self.find_uic(symbol, asset_type)
        order: dict = {
            "AccountKey": self.account_key(),
            "Uic": uic,
            "AssetType": asset_type,
            "BuySell": "Buy" if side > 0 else "Sell",
            "Amount": qty,
            "OrderType": "Market",
            "OrderDuration": {"DurationType": "DayOrder"},
            "ManualOrder": False,
        }
        if stop_price is not None:
            order["Orders"] = [{
                "AccountKey": order["AccountKey"], "Uic": uic,
                "AssetType": asset_type,
                "BuySell": "Sell" if side > 0 else "Buy",
                "Amount": qty, "OrderType": "Stop",
                "OrderPrice": stop_price,
                "OrderDuration": {"DurationType": "GoodTillCancel"},
                "ManualOrder": False,
            }]
        r = self._http.post(f"{self.eps['api']}/trade/v2/orders",
                            json=order, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def positions(self) -> list[dict]:
        return self.get("/port/v1/positions/me").get("Data", [])
