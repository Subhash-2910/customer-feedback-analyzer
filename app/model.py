"""Milestone 2: extend the GPT-2 tokenizer and model with one new token per feedback category.

The model learns to read a comment and then *generate* one of these tokens:

    Customer feedback: The cashier was rude.
    Category: <|staff_complaint|><|endoftext|>

Each new token gets its own row in the embedding matrix (GPT-2 ties input and output embeddings, so the same
row is used to predict it). Instead of a random start, the row is initialised to the mean embedding of the
category's words ("staff complaint"), which gives fine-tuning a head start.
"""
from __future__ import annotations

import torch

from app.dataset import LABELS

BASE_MODEL = "gpt2"
PROMPT = "Customer feedback: {text}\nCategory:"
MAX_LEN = 96


def label_token(label: str) -> str:
    return f"<|{label}|>"


LABEL_TOKENS = [label_token(l) for l in LABELS]


def prompt(text: str) -> str:
    return PROMPT.format(text=" ".join(text.split()))


def extend(tokenizer, model) -> list[int]:
    """Add the category tokens to `tokenizer` and `model` (in place). Returns their ids in LABELS order."""
    added = tokenizer.add_tokens(LABEL_TOKENS, special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = label_token_ids(tokenizer)
    if added:
        old = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            for label, tid in zip(LABELS, ids):
                if tid >= old:  # only initialise rows we just created
                    words = tokenizer(" " + label.replace("_", " "), add_special_tokens=False)["input_ids"]
                    emb[tid] = emb[[w for w in words if w < old]].mean(0)
    model.config.pad_token_id = tokenizer.pad_token_id
    return ids


def label_token_ids(tokenizer) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(LABEL_TOKENS)
    if any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise ValueError("tokenizer has not been extended with the category tokens; call extend() first")
    return ids


def encode(text: str, label: str, tokenizer, max_len: int = MAX_LEN) -> dict:
    """Token ids for prompt + label token + EOS. Loss is only computed on the label token and EOS."""
    p = tokenizer(prompt(text), add_special_tokens=False)["input_ids"][: max_len - 2]
    target = [tokenizer.convert_tokens_to_ids(label_token(label)), tokenizer.eos_token_id]
    return {"input_ids": p + target, "attention_mask": [1] * (len(p) + 2), "labels": [-100] * len(p) + target}


def load_base(name: str = BASE_MODEL):
    """Pretrained GPT-2 with the category tokens added (not fine-tuned yet)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    extend(tokenizer, model)
    return tokenizer, model
