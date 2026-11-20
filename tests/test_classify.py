"""Milestone 4: the inference function, the baselines and the evaluation metrics."""
import pytest

from app.baselines import keyword_predict, keyword_scores, majority_label
from app.classify import FeedbackClassifier
from app.dataset import LABELS, generate
from app.evaluate import scores
from app.model import extend
from app.train import fine_tune


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A tiny random GPT-2 fine-tuned for a second on 10 comments: enough to memorise them, which checks the
    whole train -> save -> load -> predict path (the real accuracy numbers come from app/evaluate.py)."""
    from tests.conftest import make_tiny_model, make_tiny_tokenizer
    tokenizer = make_tiny_tokenizer()
    model = make_tiny_model(len(tokenizer))
    extend(tokenizer, model)
    train, _ = generate(n_train_per_label=2, n_test_per_label=1)
    out = tmp_path_factory.mktemp("m") / "model"
    fine_tune(tokenizer, model, train, train[:2], out, epochs=60, lr=3e-3, batch_size=5, use_cpu=True)
    return out, train


def test_classify_returns_a_label_with_probabilities(trained):
    clf = FeedbackClassifier.load(str(trained[0]))
    r = clf.classify("The cashier was rude and rolled their eyes at me")
    assert r["label"] in LABELS
    assert set(r["scores"]) == set(LABELS)
    assert abs(sum(r["scores"].values()) - 1) < 1e-3
    assert r["confidence"] == max(r["scores"].values())
    assert 0 <= r["label_mass"] <= 1 and isinstance(r["generated"], str)


def test_fine_tuned_model_learns_the_training_comments(trained):
    out, train = trained
    clf = FeedbackClassifier.load(str(out))
    preds = clf.predict([r["text"] for r in train], batch_size=3)          # uneven batches: padding paths
    acc = sum(p["label"] == r["label"] for p, r in zip(preds, train)) / len(train)
    assert acc >= 0.8
    assert sum(p["generated"].startswith("<|") for p in preds) / len(preds) >= 0.8  # it *generates* label tokens


def test_batched_and_single_predictions_agree(trained):
    clf = FeedbackClassifier.load(str(trained[0]))
    texts = ["Rude", "The toaster broke after one day and the shop would not refund me at all"]
    batch = clf.predict(texts)
    for t, b in zip(texts, batch):
        assert clf.classify(t)["scores"] == pytest.approx(b["scores"], abs=1e-3)


def test_empty_feedback_and_missing_model_are_rejected(trained, tmp_path):
    with pytest.raises(ValueError):
        FeedbackClassifier.load(str(trained[0])).classify("   ")
    with pytest.raises(FileNotFoundError):
        FeedbackClassifier.load(str(tmp_path / "nope"))


def test_keyword_baseline():
    assert keyword_predict("The cashier was RUDE") == "staff_complaint"
    assert keyword_predict("Overcharged again, prices are silly") == "pricing"
    assert keyword_predict("nothing to see here", fallback="pricing") == "pricing"
    assert keyword_scores("thanks, thanks!")["staff_praise"] == 2
    assert keyword_scores("pricey")["pricing"] == 0                       # whole words only
    assert majority_label(["pricing", "staff_praise", "pricing"]) == "pricing"


def test_scores_known_values():
    gold = ["pricing", "pricing", "staff_praise", "product_issue"]
    pred = ["pricing", "staff_praise", "staff_praise", "product_issue"]
    s = scores(gold, pred)
    assert s["accuracy"] == 0.75
    assert s["per_class_f1"]["pricing"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["per_class_f1"]["store_experience"] == 0.0
    assert s["confusion"]["pricing"]["staff_praise"] == 1
    assert scores(gold, gold)["accuracy"] == 1.0
