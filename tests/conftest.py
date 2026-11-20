"""A tiny GPT-2 and tokenizer built offline, so tests run in seconds on CI without downloading anything."""
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from app.dataset import generate


def make_tiny_tokenizer():
    train, test = generate()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    texts = [r["text"] for r in train + test] + ["Customer feedback:\nCategory: product issue staff praise "
                                                  "complaint pricing store experience"]
    tok.train_from_iterator(texts, trainer)
    return GPT2TokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", bos_token="<|endoftext|>",
                             unk_token="<|endoftext|>")


def make_tiny_model(vocab_size, seed=0):
    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=128, vocab_size=vocab_size)
    return GPT2LMHeadModel(cfg)


@pytest.fixture
def tiny():
    """(tokenizer, model) before the category tokens are added."""
    tokenizer = make_tiny_tokenizer()
    return tokenizer, make_tiny_model(len(tokenizer))
