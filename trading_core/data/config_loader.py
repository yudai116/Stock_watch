"""Load config/*.yaml and .env. No parameter may be hardcoded elsewhere."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load config/<name>.yaml (name without extension)."""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    """Local datastore root (Parquet + SQLite)."""
    env = os.environ.get("TRADING_CORE_DATA_ROOT", "").strip()
    root = Path(env) if env else REPO_ROOT / "datastore"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(REPO_ROOT / ".env")


def alpaca_credentials() -> tuple[str, str]:
    """Return (api_key, secret_key), tolerant of the several naming schemes
    seen in the wild (this repo's canonical names, the old Stock_watch names,
    and Alpaca's own APCA_* names). Raises a clear error if unset."""
    load_dotenv_if_present()
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = (os.environ.get("ALPACA_SECRET_KEY")
              or os.environ.get("ALPACA_API_SECRET")      # old Stock_watch name
              or os.environ.get("APCA_API_SECRET_KEY"))   # Alpaca SDK name
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials not found. Set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY (in .env or as environment variables)."
        )
    return key, secret
