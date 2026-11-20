# One command to rebuild everything from scratch: make all   (then: make app)
PY ?= .venv/bin/python

.PHONY: all setup data train evaluate app test
all: data train evaluate

setup:     ## create .venv and install requirements
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
data:      ## Milestone 1: dataset/train.jsonl and dataset/test.jsonl
	$(PY) -m app.dataset
train:     ## Milestones 2+3: add category tokens to GPT-2, fine-tune -> model/feedback-gpt2
	$(PY) -m app.train
evaluate:  ## Milestone 4: compare against baselines -> results/metrics.md, results/samples.md
	$(PY) -m app.evaluate
app:       ## Milestone 5: the Streamlit app on http://localhost:8501
	$(PY) -m streamlit run app/main.py
test:
	$(PY) -m pytest -q
