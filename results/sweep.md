# Training settings we tried

GPT-2 (124M), batch 16, seed 0, Apple M3 GPU (MPS). Accuracy on the two test sets (see [metrics.md](metrics.md)).
Run each with `python -m app.train --epochs E --lr LR --out <dir>` then `MODEL_DIR=<dir> python -m app.evaluate`.

| epochs | learning rate | final val. loss | train time | held-out acc. (n=100) | held-out macro-F1 | hand-written acc. (n=20) |
|---|---|---|---|---|---|---|
| 2 | 5e-5 | 0.453 | 13 s | 79% | 0.78 | 70% |
| **4 (default)** | 5e-5 | 0.049 | 24 s | **92%** | **0.92** | 85% |
| 8 | 5e-5 | 0.001 | 62 s | 89% | 0.89 | 95% |
| 4 | 1e-4 | 0.002 | 27 s | 90% | 0.90 | 85% |
| keyword baseline | – | – | – | 69% | 0.72 | 60% |

Two epochs is clearly too few. From 4 epochs on, the differences are 1–3 held-out comments and 1–2 hand-written
ones, which is within the noise of test sets this small. The default (4 epochs) was fixed before these runs and
was not changed after seeing test results.
