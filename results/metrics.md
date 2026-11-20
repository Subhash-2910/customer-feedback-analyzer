# Metrics

Produced by `python -m app.evaluate`. Accuracy / macro-F1 on comments never seen in training.

| system | held-out acc. | held-out macro-F1 | hand-written acc. | hand-written macro-F1 |
|---|---|---|---|---|
| majority class (always one label) | 20% | 0.07 | 20% | 0.07 |
| keyword rules | 69% | 0.72 | 60% | 0.53 |
| zero-shot GPT-2 (no fine-tuning) | 19% | 0.09 | 20% | 0.08 |
| **fine-tuned GPT-2 (ours)** | 92% | 0.92 | 85% | 0.85 |

## Fine-tuned model on the held-out set (100 comments)

- generates a category token without any restriction: 100%
- mean confidence when right: 0.8945, when wrong: 0.7482
- speed: 22.4 ms per comment on CPU (batched)

Per-category F1: product_issue 0.91, staff_praise 0.91, staff_complaint 0.89, pricing 1.00, store_experience 0.89

| true \ predicted | product issue | staff praise | staff complaint | pricing | store experience |
|---|---|---|---|---|---|
| product issue | 20 | 0 | 0 | 0 | 0 |
| staff praise | 0 | 20 | 0 | 0 | 0 |
| staff complaint | 0 | 4 | 16 | 0 | 0 |
| pricing | 0 | 0 | 0 | 20 | 0 |
| store experience | 4 | 0 | 0 | 0 | 16 |

## Fine-tuned model on the hand-written set (20 comments)

- generates a category token without any restriction: 100%
- mean confidence when right: 0.8957, when wrong: 0.686
- speed: 10.9 ms per comment on CPU (batched)

Per-category F1: product_issue 0.89, staff_praise 0.86, staff_complaint 0.86, pricing 0.89, store_experience 0.75

| true \ predicted | product issue | staff praise | staff complaint | pricing | store experience |
|---|---|---|---|---|---|
| product issue | 4 | 0 | 0 | 0 | 0 |
| staff praise | 0 | 3 | 1 | 0 | 0 |
| staff complaint | 0 | 0 | 3 | 0 | 0 |
| pricing | 0 | 0 | 0 | 4 | 0 |
| store experience | 1 | 0 | 0 | 1 | 3 |
