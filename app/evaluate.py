"""Measure the fine-tuned classifier against baselines on comments it never saw.

    python -m app.evaluate       # -> results/metrics.json, results/metrics.md, results/samples.md

Two test sets:
- held-out: 100 generated comments (20 per category) built only from phrasings that never appear in training
- hand-written: 20 realistic comments written separately below (different style, slang, prices, names)

Metrics: accuracy, macro-F1 (mean of per-category F1), per-category F1 and a confusion matrix.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from app.baselines import keyword_predict, majority_label
from app.classify import MODEL_DIR, FeedbackClassifier
from app.dataset import LABELS, load_jsonl
from app.model import BASE_MODEL, prompt

# Written by hand (not by the generator) and labelled before running any model on them.
HANDWRITTEN = [
    ("Bought a toaster last week and it already burns everything on the lowest setting.", "product_issue"),
    ("Shoutout to Jess in the shoe dept, she found the last pair in my size and was super sweet about it",
     "staff_praise"),
    ("Guy at the returns desk basically called me a liar. Unacceptable.", "staff_complaint"),
    ("Why is milk £1.80 now?? It was £1.20 a month ago.", "pricing"),
    ("Couldn't find a single trolley and the floor near the fruit was sticky.", "store_experience"),
    ("The headphones crackle in the left ear straight out of the box.", "product_issue"),
    ("The self-service tills keep saying unexpected item and there is never anyone around to fix it",
     "store_experience"),
    ("Fantastic service from the pharmacist today, really took time to explain my prescription.", "staff_praise"),
    ("Got charged for two bags of apples when I only bought one.", "pricing"),
    ("Lights flickering in aisle 4 gave me a headache, and it smelled like bleach everywhere.", "store_experience"),
    ("The sole on my trainers peeled off after a month of normal walking.", "product_issue"),
    ("Staff member at the deli was chewing gum and didn't even look up when I asked for help.", "staff_complaint"),
    ("Your price match promise is a joke, you would not match a cheaper local shop.", "pricing"),
    ("Big thanks to the two lads who helped me load the wardrobe into my van!", "staff_praise"),
    ("Store was spotless and well organised today, nice change.", "store_experience"),
    ("The milk I bought today was sour even though the date says next week.", "product_issue"),
    ("Checkout girl was really rude when my card got declined.", "staff_complaint"),
    ("Members discount didn't come off my shopping again.", "pricing"),
    ("Manager Kevin sorted out my complaint in five minutes, brilliant.", "staff_praise"),
    ("Waited 25 min to pay because only 1 checkout open", "store_experience"),
]


def scores(gold: list[str], pred: list[str]) -> dict:
    per = {}
    for l in LABELS:
        tp = sum(g == p == l for g, p in zip(gold, pred))
        fp = sum(p == l != g for g, p in zip(gold, pred))
        fn = sum(g == l != p for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[l] = round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0
    return {"accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
            "macro_f1": round(sum(per.values()) / len(per), 4), "per_class_f1": per,
            "confusion": {g: {p: sum(a == g and b == p for a, b in zip(gold, pred)) for p in LABELS} for g in LABELS}}


@torch.no_grad()
def zero_shot_gpt2(texts: list[str], name: str = BASE_MODEL) -> list[str]:
    """Untrained GPT-2: pick the category whose name (e.g. ' staff praise') is most likely after the prompt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name).eval()
    names = [tok(" " + l.replace("_", " "), add_special_tokens=False)["input_ids"] for l in LABELS]
    preds = []
    for t in texts:
        p = tok(prompt(t), add_special_tokens=False)["input_ids"]
        logp = []
        for n in names:
            out = model(torch.tensor([p + n])).logits[0, len(p) - 1:-1].log_softmax(-1)
            logp.append(float(out[torch.arange(len(n)), torch.tensor(n)].mean()))
        preds.append(LABELS[max(range(len(LABELS)), key=logp.__getitem__)])
    return preds


def confusion_md(conf: dict) -> str:
    short = {l: l.replace("_", " ") for l in LABELS}
    lines = ["| true \\ predicted | " + " | ".join(short.values()) + " |", "|---" * (len(LABELS) + 1) + "|"]
    lines += [f"| {short[g]} | " + " | ".join(str(conf[g][p]) for p in LABELS) + " |" for g in LABELS]
    return "\n".join(lines)


def main() -> None:
    train = load_jsonl("dataset/train.jsonl")
    sets = {"held-out": [(r["text"], r["label"]) for r in load_jsonl("dataset/test.jsonl")],
            "hand-written": HANDWRITTEN}
    clf = FeedbackClassifier.load(MODEL_DIR)
    majority = majority_label([r["label"] for r in train])
    report, samples = {}, {}
    for name, rows in sets.items():
        texts, gold = [t for t, _ in rows], [g for _, g in rows]
        start = time.time()
        results = clf.predict(texts)
        ms = 1000 * (time.time() - start) / len(texts)
        preds = [r["label"] for r in results]
        right = [r["confidence"] for r, g in zip(results, gold) if r["label"] == g]
        wrong = [r["confidence"] for r, g in zip(results, gold) if r["label"] != g]
        report[name] = {
            "n": len(rows),
            "majority": scores(gold, [majority] * len(gold)),
            "keyword": scores(gold, [keyword_predict(t, majority) for t in texts]),
            "zero_shot_gpt2": scores(gold, zero_shot_gpt2(texts)),
            "fine_tuned_gpt2": scores(gold, preds),
            "fine_tuned_details": {
                "ms_per_comment_cpu": round(ms, 1),
                "generates_a_category_token": round(sum(r["generated"] in [f"<|{l}|>" for l in LABELS]
                                                        for r in results) / len(results), 4),
                "mean_confidence_when_right": round(sum(right) / len(right), 4) if right else None,
                "mean_confidence_when_wrong": round(sum(wrong) / len(wrong), 4) if wrong else None,
            },
        }
        samples[name] = [(r, g) for r, g in zip(results, gold)]

    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    md = ["# Metrics", "", "Produced by `python -m app.evaluate`. Accuracy / macro-F1 on comments never seen in training.", ""]
    md += ["| system | held-out acc. | held-out macro-F1 | hand-written acc. | hand-written macro-F1 |", "|---|---|---|---|---|"]
    names = {"majority": "majority class (always one label)", "keyword": "keyword rules",
             "zero_shot_gpt2": "zero-shot GPT-2 (no fine-tuning)", "fine_tuned_gpt2": "**fine-tuned GPT-2 (ours)**"}
    for key, label in names.items():
        h, w = report["held-out"][key], report["hand-written"][key]
        md.append(f"| {label} | {h['accuracy']:.0%} | {h['macro_f1']:.2f} | {w['accuracy']:.0%} | {w['macro_f1']:.2f} |")
    for name in sets:
        d = report[name]["fine_tuned_details"]
        md += ["", f"## Fine-tuned model on the {name} set ({report[name]['n']} comments)", "",
               f"- generates a category token without any restriction: {d['generates_a_category_token']:.0%}",
               f"- mean confidence when right: {d['mean_confidence_when_right']}, when wrong: {d['mean_confidence_when_wrong']}",
               f"- speed: {d['ms_per_comment_cpu']} ms per comment on CPU (batched)", "",
               "Per-category F1: " + ", ".join(f"{l} {f:.2f}" for l, f in report[name]["fine_tuned_gpt2"]["per_class_f1"].items()),
               "", confusion_md(report[name]["fine_tuned_gpt2"]["confusion"])]
    Path("results/metrics.md").write_text("\n".join(md) + "\n")

    s = ["# Sample outputs", "", "Every hand-written comment, then the held-out comments the model got wrong.", "",
         "| comment | true | predicted | confidence | |", "|---|---|---|---|---|"]
    rows = samples["hand-written"] + [x for x in samples["held-out"] if x[0]["label"] != x[1]]
    for r, g in rows:
        s.append(f"| {r['text'].replace('|', '/')} | {g} | {r['label']} | {r['confidence']:.0%} | {'✅' if r['label'] == g else '❌'} |")
    Path("results/samples.md").write_text("\n".join(s) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
