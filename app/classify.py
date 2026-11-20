"""Milestone 4: classify new feedback with the fine-tuned model.

    python -m app.classify "The cashier was rude to me"

How a prediction is made: the comment is wrapped in the training prompt ("Customer feedback: ...\\nCategory:")
and the model computes the next-token distribution - i.e. the first token it would *generate*. We read the
probability of each category token from that distribution:

- `label`       the category token with the highest probability
- `confidence`  that token's probability, renormalised over the category tokens only (sums to 1 over categories)
- `generated`   the token the model would generate with no restriction at all (greedy decoding). After
                fine-tuning this is almost always the same category token; if it is not, the model is unsure
                and `label_mass` (share of the full distribution on category tokens) is low.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import torch

from app.dataset import LABELS
from app.model import label_token_ids, prompt

MODEL_DIR = os.environ.get("MODEL_DIR", "model/feedback-gpt2")


class FeedbackClassifier:
    def __init__(self, tokenizer, model, device: str = "cpu"):
        self.tokenizer, self.model, self.device = tokenizer, model.to(device).eval(), device
        self.label_ids = label_token_ids(tokenizer)

    @classmethod
    def load(cls, path: str = MODEL_DIR, device: str = "cpu") -> "FeedbackClassifier":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not os.path.isdir(path):
            raise FileNotFoundError(f"No fine-tuned model at {path!r}. Train one first: python -m app.train")
        return cls(AutoTokenizer.from_pretrained(path), AutoModelForCausalLM.from_pretrained(path), device)

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        out = []
        for i in range(0, len(texts), batch_size):
            out += self._batch(texts[i:i + batch_size])
        return out

    def _batch(self, texts: list[str]) -> list[dict]:
        enc = [self.tokenizer(prompt(t), add_special_tokens=False)["input_ids"][-1000:] for t in texts]
        n = max(len(e) for e in enc)
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([e + [pad] * (n - len(e)) for e in enc], device=self.device)
        mask = torch.tensor([[1] * len(e) + [0] * (n - len(e)) for e in enc], device=self.device)
        logits = self.model(input_ids=ids, attention_mask=mask).logits
        last = logits[torch.arange(len(enc)), torch.tensor([len(e) - 1 for e in enc])].float()  # right padding
        full = last.softmax(-1)
        cat = last[:, self.label_ids].softmax(-1)
        results = []
        for t, p_full, p_cat in zip(texts, full, cat):
            k = int(p_cat.argmax())
            results.append({
                "text": t,
                "label": LABELS[k],
                "confidence": round(float(p_cat[k]), 4),
                "scores": {l: round(float(p), 4) for l, p in zip(LABELS, p_cat)},
                "generated": self.tokenizer.decode([int(p_full.argmax())]),
                "label_mass": round(float(p_full[self.label_ids].sum()), 4),
            })
        return results

    def classify(self, text: str) -> dict:
        """Classify one comment. See the module docstring for the fields."""
        if not text or not text.strip():
            raise ValueError("empty feedback")
        return self.predict([text])[0]


@lru_cache(maxsize=2)
def get_classifier(path: str = MODEL_DIR) -> FeedbackClassifier:
    return FeedbackClassifier.load(path)


def classify(text: str, path: str = MODEL_DIR) -> dict:
    """Convenience wrapper: load the saved model once, then classify `text`."""
    return get_classifier(path).classify(text)


if __name__ == "__main__":
    for comment in sys.argv[1:] or ["The kettle stopped working after a week."]:
        r = classify(comment)
        print(f"{r['label']:<17} {r['confidence']:.0%}  {comment}")
