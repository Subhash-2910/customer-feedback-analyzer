"""Helpers for the Streamlit app: read comments from text or uploads, save results, count trends."""
from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import datetime, timezone

from app.dataset import LABELS
from app.storage import Store

KEY = "feedback"
REVIEW_BELOW = 0.6          # predictions less confident than this are flagged for a human to check
TEXT_COLUMNS = ("feedback", "comment", "comments", "text", "review", "message")


def split_comments(data: str | bytes, filename: str = "") -> list[str]:
    """One comment per non-empty line; for a .csv, the first column named like feedback/comment/text."""
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    if filename.lower().endswith(".csv"):
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            return []
        header = [h.strip().lower() for h in rows[0]]
        col = next((header.index(c) for c in TEXT_COLUMNS if c in header), 0)
        return [r[col].strip() for r in rows[1:] if len(r) > col and r[col].strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def needs_review(result: dict, threshold: float = REVIEW_BELOW) -> bool:
    return result["confidence"] < threshold


def save(store: Store, results: list[dict], now: datetime | None = None) -> int:
    """Append classified comments to the store. Returns how many are saved in total."""
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    history = store.get(KEY, [])
    history += [{"time": stamp, "text": r["text"], "label": r["label"], "confidence": r["confidence"],
                 "needs_review": needs_review(r)} for r in results]
    store.put(KEY, history)
    return len(history)


def history(store: Store) -> list[dict]:
    return store.get(KEY, [])


def clear(store: Store) -> None:
    store.put(KEY, [])


def counts(rows: list[dict]) -> dict[str, int]:
    """Comments per category, every category present (0 if none), most frequent first."""
    c = Counter(r["label"] for r in rows)
    return dict(sorted(((l, c[l]) for l in LABELS), key=lambda kv: (-kv[1], LABELS.index(kv[0]))))


def to_csv(rows: list[dict]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=["time", "text", "label", "confidence", "needs_review"])
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()
