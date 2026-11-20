"""Milestone 3: fine-tune GPT-2 with the Hugging Face Trainer to generate the right category token.

    python -m app.train                     # -> model/feedback-gpt2 and results/training.json
    python -m app.train --epochs 3 --base distilgpt2

10% of the training comments are kept aside as a validation set (the test split is never touched here).
On an Apple M3 this takes about a minute; the GPU (MPS) is used automatically when available.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from app.dataset import load_jsonl
from app.model import BASE_MODEL, encode, load_base

MODEL_DIR = os.environ.get("MODEL_DIR", "model/feedback-gpt2")


def to_dataset(records: list[dict], tokenizer) -> Dataset:
    return Dataset.from_list([encode(r["text"], r["label"], tokenizer) for r in records])


class PadCollator:
    """Right-pad a batch: pad id for inputs, 0 for attention, -100 (ignored by the loss) for labels."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda key, v: [b[key] + [v] * (n - len(b[key])) for b in batch]
        return {"input_ids": torch.tensor(pad("input_ids", self.pad_id)),
                "attention_mask": torch.tensor(pad("attention_mask", 0)),
                "labels": torch.tensor(pad("labels", -100))}


def split(records: list[dict], val_frac: float = 0.1, seed: int = 0) -> tuple[list, list]:
    rows = records[:]
    random.Random(seed).shuffle(rows)
    k = max(1, int(len(rows) * val_frac))
    return rows[k:], rows[:k]


def fine_tune(tokenizer, model, train_rows, val_rows, out_dir: str | Path, *, epochs: float = 4,
              lr: float = 5e-5, batch_size: int = 16, seed: int = 0, use_cpu: bool = False) -> dict:
    """Train with the HF Trainer, save model + tokenizer to `out_dir`, return a summary of the run."""
    args = TrainingArguments(
        output_dir=str(Path(out_dir).parent / "checkpoints"), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, warmup_steps=10,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="no", logging_strategy="epoch",
        report_to=[], seed=seed, use_cpu=use_cpu, disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=to_dataset(train_rows, tokenizer),
                      eval_dataset=to_dataset(val_rows, tokenizer), data_collator=PadCollator(tokenizer.pad_token_id))
    start = time.time()
    trainer.train()
    seconds = time.time() - start
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    history = [{k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()
                if k in ("epoch", "loss", "eval_loss")} for h in trainer.state.log_history]
    return {"epochs": epochs, "learning_rate": lr, "batch_size": batch_size, "train_size": len(train_rows),
            "val_size": len(val_rows), "seconds": round(seconds, 1), "device": str(trainer.args.device),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "history": [h for h in history if len(h) > 1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default=MODEL_DIR)
    a = ap.parse_args()
    tokenizer, model = load_base(a.base)
    train_rows, val_rows = split(load_jsonl("dataset/train.jsonl"))
    summary = {"base_model": a.base, **fine_tune(tokenizer, model, train_rows, val_rows, a.out,
                                                   epochs=a.epochs, lr=a.lr)}
    Path("results").mkdir(exist_ok=True)
    Path("results/training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
