"""Milestone 3: fine-tuning with the Hugging Face Trainer (on a tiny model, CPU, a few seconds)."""
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.dataset import generate
from app.model import LABEL_TOKENS, encode, extend
from app.train import PadCollator, fine_tune, split


def test_collator_pads_inputs_attention_and_labels(tiny):
    tokenizer, model = tiny
    extend(tokenizer, model)
    a, b = encode("Rude", "staff_complaint", tokenizer), encode("The kettle broke after a day", "product_issue", tokenizer)
    batch = PadCollator(tokenizer.pad_token_id)([a, b])
    n = len(b["input_ids"])
    assert batch["input_ids"].shape == batch["labels"].shape == (2, n)
    pad = n - len(a["input_ids"])
    assert batch["attention_mask"][0].tolist() == [1] * len(a["input_ids"]) + [0] * pad
    assert batch["labels"][0].tolist()[-pad:] == [-100] * pad


def test_split_is_disjoint_and_reproducible():
    rows = [{"text": str(i), "label": "pricing"} for i in range(50)]
    tr, va = split(rows)
    assert len(va) == 5 and len(tr) == 45 and not {r["text"] for r in tr} & {r["text"] for r in va}
    assert split(rows) == (tr, va)


def test_trainer_lowers_loss_and_saves_a_reloadable_model(tiny, tmp_path):
    tokenizer, model = tiny
    extend(tokenizer, model)
    train, _ = generate(n_train_per_label=8, n_test_per_label=1)
    tr, va = split(train, val_frac=0.2)
    out = tmp_path / "model"
    summary = fine_tune(tokenizer, model, tr, va, out, epochs=6, lr=3e-3, batch_size=8, use_cpu=True)
    losses = [h["loss"] for h in summary["history"] if "loss" in h]
    assert len(losses) == 6 and losses[-1] < losses[0]
    assert summary["train_size"] == 32 and summary["val_size"] == 8
    reloaded_tok = AutoTokenizer.from_pretrained(out)
    reloaded = AutoModelForCausalLM.from_pretrained(out)
    assert reloaded_tok.convert_tokens_to_ids(LABEL_TOKENS) == tokenizer.convert_tokens_to_ids(LABEL_TOKENS)
    assert reloaded.get_input_embeddings().weight.shape[0] == len(tokenizer)
