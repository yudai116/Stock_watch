"""FinBERT sentiment scoring with pinned model version and score cache.

Reproducibility rule (SPEC §7): backtests may ONLY read cached scores keyed
by (article id, model_name, model_revision). Scoring is an offline batch job:

    python -m altdata.sentiment_scorer

Scores in [-1, 1] = P(positive) - P(negative).
Requires optional extra: uv sync --extra altdata
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.config_loader import data_root, load_config

UTC = timezone.utc


class SentimentCache:
    def __init__(self, root: Path | None = None):
        cfg = load_config("params")["sentiment"]
        self.model_name = cfg["model_name"]
        self.model_revision = cfg["model_revision"]
        self.db = sqlite3.connect((root or data_root()) / "sentiment_cache.sqlite")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS scores (
                article_id TEXT, model TEXT, revision TEXT,
                score REAL, scored_at TEXT,
                PRIMARY KEY (article_id, model, revision))"""
        )
        self.db.commit()

    def get(self, article_id: str) -> float | None:
        row = self.db.execute(
            "SELECT score FROM scores WHERE article_id=? AND model=? AND revision=?",
            (article_id, self.model_name, self.model_revision),
        ).fetchone()
        return row[0] if row else None

    def put(self, article_id: str, score: float) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?)",
            (article_id, self.model_name, self.model_revision, float(score),
             datetime.now(UTC).isoformat()),
        )
        self.db.commit()

    def get_many(self, article_ids: list[str]) -> dict[str, float]:
        out = {}
        for aid in article_ids:
            s = self.get(aid)
            if s is not None:
                out[aid] = s
        return out


def article_id(payload: dict) -> str:
    """Stable id: Alpaca news id when present, else hash of headline+ts."""
    if payload.get("id"):
        return str(payload["id"])
    return hashlib.sha1(payload.get("headline", "").encode()).hexdigest()


class FinBertScorer:
    """Lazy-loaded FinBERT. CPU inference is acceptable (SPEC §1)."""

    def __init__(self):
        cfg = load_config("params")["sentiment"]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tok = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["model_revision"])
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], revision=cfg["model_revision"]
        )
        self.pipe = pipeline("text-classification", model=model, tokenizer=tok,
                             top_k=None, truncation=True, max_length=512)

    def score(self, texts: list[str]) -> list[float]:
        out = []
        for res in self.pipe(texts, batch_size=16):
            probs = {r["label"].lower(): r["score"] for r in res}
            out.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
        return out


def score_all_pending(batch: int = 256) -> int:
    """Score every cached-miss news article in the store."""
    from data.bitemporal_store import BitemporalStore

    store = BitemporalStore(data_root())
    cache = SentimentCache()
    news = store.get_records_asof("news", datetime.now(UTC))
    if news.empty:
        print("no news in store")
        return 0
    scorer = FinBertScorer()
    n_scored = 0
    pending: list[tuple[str, str]] = []
    for _, r in news.iterrows():
        aid = article_id(r["payload"])
        if cache.get(aid) is None:
            text = (r["payload"].get("headline", "") + ". " + r["payload"].get("summary", "")).strip()
            pending.append((aid, text))
    for i in range(0, len(pending), batch):
        chunk = pending[i : i + batch]
        scores = scorer.score([t for _, t in chunk])
        for (aid, _), s in zip(chunk, scores):
            cache.put(aid, s)
        n_scored += len(chunk)
        print(f"scored {n_scored}/{len(pending)}")
    return n_scored


if __name__ == "__main__":
    score_all_pending()
