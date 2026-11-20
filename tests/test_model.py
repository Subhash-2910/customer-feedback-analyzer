"""Milestone 2: extending the tokenizer and model with category tokens."""
import torch

from app.dataset import LABELS
from app.model import LABEL_TOKENS, encode, extend, label_token_ids, prompt


def test_each_category_becomes_one_new_token(tiny):
    tokenizer, model = tiny
    before = len(tokenizer)
    ids = extend(tokenizer, model)
    assert len(tokenizer) == before + len(LABELS)
    assert ids == list(range(before, before + len(LABELS)))
    for tok, tid in zip(LABEL_TOKENS, ids):
        assert tokenizer(tok, add_special_tokens=False)["input_ids"] == [tid]   # never split into pieces
        assert tokenizer.decode([tid]) == tok


def test_model_embeddings_grow_and_new_rows_start_from_the_label_words(tiny):
    tokenizer, model = tiny
    old = model.get_input_embeddings().weight.detach().clone()
    ids = extend(tokenizer, model)
    emb = model.get_input_embeddings().weight
    assert emb.shape[0] == len(tokenizer)
    assert torch.equal(emb[: old.shape[0]], old)                       # existing rows untouched
    words = tokenizer(" staff praise", add_special_tokens=False)["input_ids"]
    assert torch.allclose(emb[ids[LABELS.index("staff_praise")]], old[words].mean(0))
    assert model.get_output_embeddings().weight.shape[0] == len(tokenizer)
    out = model(torch.tensor([[ids[0], ids[1]]]))
    assert out.logits.shape[-1] == len(tokenizer)


def test_extend_twice_is_harmless(tiny):
    tokenizer, model = tiny
    first = extend(tokenizer, model)
    assert extend(tokenizer, model) == first
    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)


def test_label_ids_require_an_extended_tokenizer(tiny):
    tokenizer, _ = tiny
    try:
        label_token_ids(tokenizer)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_encode_masks_the_prompt_and_targets_label_then_eos(tiny):
    tokenizer, model = tiny
    ids = extend(tokenizer, model)
    ex = encode("  Rude   cashier ", "staff_complaint", tokenizer)
    n_prompt = len(tokenizer(prompt("Rude cashier"))["input_ids"])
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"]) == n_prompt + 2
    assert ex["labels"][:n_prompt] == [-100] * n_prompt
    assert ex["labels"][-2:] == [ids[LABELS.index("staff_complaint")], tokenizer.eos_token_id]
    assert len(encode("word " * 500, "pricing", tokenizer, max_len=40)["input_ids"]) == 40
