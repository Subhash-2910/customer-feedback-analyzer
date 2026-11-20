"""Simple systems the fine-tuned model must beat (used by app/evaluate.py).

- majority:  always predicts the most common training category (with a balanced dataset: chance, 20%)
- keyword:   counts keywords per category: words from the training phrasings plus a few generic ones.
             Words that only occur in the test sets were deliberately left out, so the baseline is not tuned
             on the test. Ties and no-hits fall back to the majority label.
- zero-shot GPT-2 (in evaluate.py): the *untrained* base model picks the category whose name it finds most
  likely after the same prompt - shows what fine-tuning adds over the pretrained model.
"""
from __future__ import annotations

import re
from collections import Counter

from app.dataset import LABELS

KEYWORDS: dict[str, list[str]] = {
    "product_issue": ["broke", "broken", "faulty", "defective", "damaged", "stopped working", "missing",
                      "quality", "fell apart", "crushed", "dented", "expired", "leak", "doesn't work", "does not work"],
    "staff_praise": ["thank", "thanks", "helpful", "friendly", "kind", "lovely", "polite", "great service",
                     "patient", "welcoming", "cheerful", "credit to"],
    "staff_complaint": ["rude", "ignored", "dismissive", "sarcastic", "attitude", "unfriendly", "condescending",
                        "shouted", "impatient", "annoyed", "refused", "unhelpful"],
    "pricing": ["price", "prices", "charged", "overcharged", "expensive", "cost", "costs", "discount", "offer",
                "refund", "sale", "points", "value"],
    "store_experience": ["dirty", "queue", "shelves", "toilets", "layout", "music", "car park", "parking",
                         "till", "tills", "checkout", "aisle", "aisles", "floor", "cramped", "filthy",
                         "fitting room", "trolley"],
}


def majority_label(train_labels: list[str]) -> str:
    counts = Counter(train_labels)
    return max(LABELS, key=lambda l: (counts[l], -LABELS.index(l)))


def keyword_scores(text: str) -> dict[str, int]:
    t = text.lower()
    return {label: sum(len(re.findall(r"(?<!\w)" + re.escape(k) + r"(?!\w)" if k[0].isalnum() else re.escape(k), t))
                       for k in words) for label, words in KEYWORDS.items()}


def keyword_predict(text: str, fallback: str = LABELS[0]) -> str:
    scores = keyword_scores(text)
    best = max(scores.values())
    winners = [l for l, s in scores.items() if s == best]
    return winners[0] if best > 0 and len(winners) == 1 else fallback
