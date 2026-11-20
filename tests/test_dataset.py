"""Milestone 1: categories and the labelled dataset."""
import re
from collections import Counter

from app.dataset import CATEGORIES, HELD_OUT, LABELS, PHRASES, generate, load_jsonl, save_jsonl


def test_three_to_five_categories_each_with_a_description_and_phrasings():
    assert 3 <= len(CATEGORIES) <= 5
    assert set(PHRASES) == set(LABELS)
    assert all(CATEGORIES[c] and len(PHRASES[c]) > HELD_OUT for c in LABELS)


def test_generation_is_reproducible_and_balanced():
    train, test = generate(seed=1)
    assert (train, test) == generate(seed=1)
    assert generate(seed=2)[0] != train
    assert set(Counter(r["label"] for r in train).values()) == {60}
    assert set(Counter(r["label"] for r in test).values()) == {20}
    assert len({r["text"] for r in train}) > 0.8 * len(train)       # mostly distinct comments


def _pattern(phrase):
    parts = re.split(r"\{\w+\}", phrase)
    return re.compile(".+".join(re.escape(p) for p in parts), re.IGNORECASE)


def test_test_split_only_uses_held_out_phrasings():
    train, test = generate()
    for label in LABELS:
        held = [_pattern(p) for p in PHRASES[label][-HELD_OUT:]]
        assert not any(p.search(r["text"]) for p in held for r in train)
        assert all(any(p.search(r["text"]) for p in held) for r in test if r["label"] == label)


def test_jsonl_round_trip(tmp_path):
    rows = [{"text": "Rude cashier", "label": "staff_complaint"}]
    save_jsonl(rows, tmp_path / "x" / "d.jsonl")
    assert load_jsonl(tmp_path / "x" / "d.jsonl") == rows
