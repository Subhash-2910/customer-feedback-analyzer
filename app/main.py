"""Milestone 5: Streamlit app. Run with `streamlit run app/main.py` (train the model first: `make train`)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `app.` imports work under streamlit

import pandas as pd
import streamlit as st

from app import feedback
from app.dataset import CATEGORIES
from app.storage import Store

MODEL_DIR = os.environ.get("MODEL_DIR", "model/feedback-gpt2")
EXAMPLE = """The blender I bought last week stopped working after two uses.
Priya at the customer service desk was so patient and kind.
Waited 20 minutes at the checkout, only one till open.
Why is everything 30% more expensive than last month?
The guy in electronics just ignored me and kept scrolling his phone."""

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="🛒", layout="wide")
st.title("Customer Feedback Analyzer")
st.write("Paste customer comments (one per line) or upload a file. A fine-tuned GPT-2 sorts each one into a "
         "category so you can see what to fix first.")


@st.cache_resource(show_spinner="Loading the model…")
def load_classifier(path: str):
    from app.classify import FeedbackClassifier
    return FeedbackClassifier.load(path)


with st.sidebar:
    st.header("Categories")
    for name, desc in CATEGORIES.items():
        st.markdown(f"**{name.replace('_', ' ')}**: {desc}")
    st.caption(f"Comments below {feedback.REVIEW_BELOW:.0%} confidence are flagged for review.")

store = Store()
classify_tab, trends_tab = st.tabs(["Classify", "Trends"])

with classify_tab:
    text = st.text_area("Customer comments, one per line", EXAMPLE, height=160)
    upload = st.file_uploader("…or upload a .txt or .csv (column named feedback, comment or text)", type=["txt", "csv"])
    comments = feedback.split_comments(upload.getvalue(), upload.name) if upload else feedback.split_comments(text)
    st.markdown(f"**{len(comments)}** comment(s) found.")
    keep = st.checkbox("Save results for the Trends tab", value=True)

    if st.button("Classify", type="primary", disabled=not comments):
        try:
            clf = load_classifier(MODEL_DIR)
        except FileNotFoundError:
            st.error(f"No trained model at `{MODEL_DIR}`. Train it first with `make train` "
                     "(about a minute on a laptop), then reload this page.")
            st.stop()
        with st.spinner("Classifying…"):
            results = clf.predict(comments)
        if keep:
            total = feedback.save(store, results)
            st.success(f"Saved. {total} comment(s) in your history.")
        table = pd.DataFrame([{"comment": r["text"], "category": r["label"].replace("_", " "),
                               "confidence": r["confidence"],
                               "check": "⚠️ review" if feedback.needs_review(r) else ""} for r in results])
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"comment": st.column_config.TextColumn(width="large"),
                                    "check": st.column_config.TextColumn(width="small"),
                                    "confidence": st.column_config.ProgressColumn(
                                        "confidence", min_value=0.0, max_value=1.0, format="percent")})
        if len(results) == 1:
            st.bar_chart(pd.Series(results[0]["scores"], name="probability"))

with trends_tab:
    rows = feedback.history(store)
    if not rows:
        st.info("No saved comments yet. Classify some on the first tab.")
    else:
        counts = feedback.counts(rows)
        top = next(iter(counts))
        flagged = sum(r["needs_review"] for r in rows)
        a, b, c = st.columns(3)
        a.metric("Saved comments", len(rows))
        b.metric("Most common category", top.replace("_", " "))
        c.metric("Flagged for review", flagged)
        chart = pd.DataFrame({"category": [k.replace("_", " ") for k in counts], "comments": list(counts.values())})
        st.vega_lite_chart(chart, {
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {"y": {"field": "category", "type": "nominal", "sort": "-x", "title": None},
                         "x": {"field": "comments", "type": "quantitative", "axis": {"tickMinStep": 1}}},
        }, use_container_width=True)
        table = pd.DataFrame(rows[::-1]).assign(label=lambda d: d["label"].str.replace("_", " "))
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", feedback.to_csv(rows), "classified_feedback.csv", "text/csv")
        if st.button("Clear history"):
            feedback.clear(store)
            st.rerun()
