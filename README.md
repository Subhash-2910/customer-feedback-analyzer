# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/smdebx95x5/badge.svg)](https://ask.thili.ai/b/smdebx95x5)

This project creates a Customer Feedback Analyzer for retail managers. It uses a fine-tuned GPT-2 model to automatically categorize customer feedback comments, helping managers quickly identify trends and prioritize improvements. The tool streamlines the process of understanding customer sentiment.

## What you are building

**Goal.** You will build a tool that automatically categorizes customer feedback comments into predefined categories using a fine-tuned language model.

**Done when.** You will have a fine-tuned GPT-2 model that classifies new customer comments by generating special label tokens, and an inference pipeline to make predictions, accessible via a simple Streamlit application.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Dataset Creation: Define categories and create labeled examples for fine-tuning.
- [x] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Tokenizer/Model Extension: Extend the GPT-2 tokenizer and model with new category tokens.
- [x] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Fine-tuning: Use the Hugging Face Trainer to fine-tune the GPT-2 model.
- [ ] **4. Implement an inference function to classify new feedback.**  
      Hint: Inference Pipeline: Implement the classification logic for new feedback comments.
- [ ] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Application Integration: Build a simple Streamlit app to input comments and display classifications.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset with more feedback comments and categories to improve model robustness.
- Add functionality to display the model's confidence score for each classification.
- Implement a feature to save the classified feedback comments for later review.
