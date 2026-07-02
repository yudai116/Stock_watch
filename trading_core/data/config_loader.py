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
