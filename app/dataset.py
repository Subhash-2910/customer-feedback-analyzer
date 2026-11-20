"""Milestone 1: five feedback categories and a labelled dataset of retail customer comments.

    python -m app.dataset        # writes dataset/train.jsonl and dataset/test.jsonl

Each record is {"text": <customer comment>, "label": <category>}.

Comments are generated from hand-written phrasings with slots ({product}, {staff}, {dept}, ...), wrapped in
optional openers/closers and light noise (lowercase, missing final punctuation). Every category has 12
phrasings; the last 3 are *held out* and only appear in the test split, so the test checks whether the model
learned the category rather than memorised sentences.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

CATEGORIES: dict[str, str] = {
    "product_issue": "Faulty, damaged, expired, wrong or poor-quality products",
    "staff_praise": "Compliments about an employee or the service they gave",
    "staff_complaint": "Rude, unhelpful or absent staff",
    "pricing": "Prices, promotions, discounts, refunds and value for money",
    "store_experience": "Cleanliness, layout, queues, stock on shelves, parking and atmosphere",
}
LABELS = list(CATEGORIES)
HELD_OUT = 3  # last N phrasings per category are test-only

PHRASES: dict[str, list[str]] = {
    "product_issue": [
        "the {product} I bought stopped working after two days",
        "my {product} arrived crushed and dented",
        "the {product} was used and dirty when I opened it",
        "the {product} came apart after one use",
        "I got home and the {product} was missing parts",
        "the {product} was faulty and I had to take it back",
        "the {product} is much smaller than the picture on the box",
        "the {product} was a different model from the one on the label",
        "the {product} broke the first time I used it",
        "the {product} was damaged inside the sealed packaging",
        "the quality of the {product} is really poor, it fell apart within a week",
        "the {product} is defective and does not do what it says",
    ],
    "staff_praise": [
        "{staff} at the {dept} counter was incredibly helpful",
        "{staff} went out of their way to find my size",
        "a big thank you to {staff} who was so friendly and patient",
        "the assistant in {dept} gave me great advice",
        "{staff} remembered me from last week and was lovely",
        "the cashier was cheerful and made my day",
        "{staff} carried my bags to the car without being asked",
        "the team in {dept} were polite and knew their stuff",
        "{staff} explained every option clearly and never rushed me",
        "{staff} stayed after closing to help me finish my order",
        "the employee at the {dept} desk was warm and welcoming",
        "{staff} is a credit to your store, so kind to my kids",
    ],
    "staff_complaint": [
        "the cashier was rude and rolled their eyes at me",
        "nobody in {dept} would help, they were all on their phones",
        "{staff} ignored me and kept chatting to a colleague",
        "the staff at the {dept} counter were dismissive and impatient",
        "I asked {staff} a simple question and got a sarcastic answer",
        "the security guard followed me around like a criminal",
        "{staff} shouted at an elderly customer in front of everyone",
        "the assistant in {dept} refused to check the stockroom for me",
        "the employees seemed annoyed that I was even there",
        "{staff} walked off halfway through helping me",
        "the manager was condescending when I raised a problem",
        "the person on the {dept} till was unfriendly and gave me attitude",
    ],
    "pricing": [
        "the {product} costs twice as much here as online",
        "the shelf price for the {product} did not match what I was charged",
        "your prices have gone up again and it is not worth it any more",
        "the buy one get one free offer was not applied at the till",
        "I was overcharged for the {product}",
        "the discount advertised in the flyer was not honoured",
        "far too expensive for what you get",
        "the loyalty points never showed up on my receipt",
        "the sale sign was misleading, the {product} was full price",
        "still waiting for my refund on the {product} after three weeks",
        "great quality but the price tag on the {product} is outrageous",
        "you charged me for a bag even though I brought my own",
    ],
    "store_experience": [
        "the {dept} aisle was dirty and there was rubbish on the floor",
        "the queue at the checkout took over twenty minutes",
        "the shelves in {dept} were empty again",
        "the toilets were filthy and out of soap",
        "the store layout is confusing, I could not find anything",
        "the music is so loud you cannot hear yourself think",
        "the car park is always full and badly lit",
        "only one till was open on a Saturday afternoon",
        "the store was freezing and the floors were wet",
        "the fitting rooms were closed and piled with clothes",
        "the aisles are so cramped I could not get a trolley through",
        "half the self-checkout machines were out of order",
    ],
}

SLOTS = {
    "product": ["kettle", "jacket", "phone charger", "yoghurt", "toaster", "pair of jeans", "headphones",
                "bread", "blender", "backpack", "lamp", "shampoo", "running shoes", "coffee machine"],
    "staff": ["Priya", "Tom", "the lady", "Ahmed", "Maria", "the young man", "Sam", "Chen", "Grace", "Luis"],
    "dept": ["electronics", "bakery", "menswear", "customer service", "garden", "pharmacy", "homeware", "deli"],
}
OPENERS = ["", "", "", "Visited on Saturday. ", "Hi, ", "Just a quick note: ", "Honestly, ",
           "I shop here every week and ", "Not happy. ", "Wanted to say that ", "FYI ", "Store #12: "]
CLOSERS = ["", "", "", ".", "!", ". Please sort this out.", ". Thanks.", ". Will be back.",
           ". Never again.", " :)", ". Just letting you know.", "!!"]


def _fill(phrase: str, rng: random.Random) -> str:
    return phrase.format(**{k: rng.choice(v) for k, v in SLOTS.items()})


def make_comment(label: str, phrase: str, rng: random.Random) -> str:
    opener, closer = rng.choice(OPENERS), rng.choice(CLOSERS)
    body = _fill(phrase, rng)
    text = opener + (body[0].upper() + body[1:] if not opener or opener.endswith(". ") else body) + closer
    if rng.random() < 0.15:
        text = text.lower()
    return text.strip()


def generate(n_train_per_label: int = 60, n_test_per_label: int = 20, seed: int = 0) -> tuple[list, list]:
    """Return (train, test) lists of {"text", "label"}. Test phrasings never appear in train."""
    rng = random.Random(seed)
    train, test = [], []
    for label in LABELS:
        seen, unseen = PHRASES[label][:-HELD_OUT], PHRASES[label][-HELD_OUT:]
        train += [{"text": make_comment(label, rng.choice(seen), rng), "label": label}
                  for _ in range(n_train_per_label)]
        test += [{"text": make_comment(label, rng.choice(unseen), rng), "label": label}
                 for _ in range(n_test_per_label)]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    train, test = generate()
    save_jsonl(train, "dataset/train.jsonl")
    save_jsonl(test, "dataset/test.jsonl")
    print(f"wrote {len(train)} train and {len(test)} test comments over {len(LABELS)} categories")
    for r in train[:5]:
        print(f"  [{r['label']}] {r['text']}")


if __name__ == "__main__":
    main()
