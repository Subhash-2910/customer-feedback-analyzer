"""Milestone 5: the Streamlit app and its helpers."""
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import feedback
from app.dataset import LABELS
from app.storage import Store

PAGE = str(Path(__file__).parent.parent / "app" / "main.py")


def test_split_comments_from_text_and_csv():
    assert feedback.split_comments("one\n\n  two  \n") == ["one", "two"]
    csv = b"id,Comment\n1,Rude cashier\n2,\n3,\"Broken kettle, again\"\n"
    assert feedback.split_comments(csv, "survey.CSV") == ["Rude cashier", "Broken kettle, again"]
    assert feedback.split_comments("notes\nGood staff\n", "x.csv") == ["Good staff"]   # falls back to column 1
    assert feedback.split_comments(b"", "x.csv") == []


def test_save_history_counts_and_csv(tmp_path):
    store = Store(str(tmp_path / "s.json"))
    t = datetime(2026, 11, 20, 9, 30, tzinfo=timezone.utc)
    rows = [{"text": "a", "label": "pricing", "confidence": 0.9},
            {"text": "b", "label": "pricing", "confidence": 0.5},
            {"text": "c", "label": "staff_praise", "confidence": 0.99}]
    assert feedback.save(store, rows, now=t) == 3
    assert feedback.save(store, rows[:1], now=t) == 4
    hist = feedback.history(store)
    assert hist[1] == {"time": "2026-11-20T09:30:00+00:00", "text": "b", "label": "pricing",
                       "confidence": 0.5, "needs_review": True}
    c = feedback.counts(hist)
    assert list(c.items())[:2] == [("pricing", 3), ("staff_praise", 1)] and set(c) == set(LABELS)
    assert feedback.to_csv(hist).splitlines()[0] == "time,text,label,confidence,needs_review"
    feedback.clear(store)
    assert feedback.history(store) == []


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))
    return monkeypatch


def test_page_without_a_model_explains_how_to_train(env, tmp_path):
    env.setenv("MODEL_DIR", str(tmp_path / "missing"))
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Customer Feedback Analyzer"
    assert any(m.value == "**5** comment(s) found." for m in at.markdown)        # the example text
    at.button[0].click().run()
    assert "make train" in at.error[0].value


def test_page_classifies_saves_and_shows_trends(env, tmp_path):
    # Reuse the tiny trained model from the milestone 4 tests' recipe.
    from app.model import extend
    from app.train import fine_tune
    from app.dataset import generate
    from tests.conftest import make_tiny_model, make_tiny_tokenizer
    tok = make_tiny_tokenizer()
    model = make_tiny_model(len(tok))
    extend(tok, model)
    rows = generate(n_train_per_label=2, n_test_per_label=1)[0]
    fine_tune(tok, model, rows, rows[:2], tmp_path / "model", epochs=5, lr=3e-3, batch_size=5, use_cpu=True)
    env.setenv("MODEL_DIR", str(tmp_path / "model"))

    at = AppTest.from_file(PAGE, default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert "5 comment(s) in your history" in at.success[0].value
    table = at.dataframe[0].value
    assert list(table.columns) == ["comment", "category", "confidence", "check"] and len(table) == 5
    assert set(table["category"]) <= {l.replace("_", " ") for l in LABELS}
    assert [m.label for m in at.metric] == ["Saved comments", "Most common category", "Flagged for review"]
    assert at.metric[0].value == "5"
    assert len(feedback.history(Store())) == 5
