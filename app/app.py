import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.preprocess import ensure_nltk_resources
from src.inference import load_artifacts, predict_text

st.set_page_config(page_title="SMS Spam Detection", layout="centered")
st.title("SMS Spam Detection (TF-IDF + MultinomialNB)")

ensure_nltk_resources()

@st.cache_resource
def _load():
    return load_artifacts()

vectorizer, model = _load()

text = st.text_area("Enter an SMS message")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        pred = predict_text(text, vectorizer, model)
        if pred == 1:
            st.error("Spam")
        else:
            st.success("Not Spam (Ham)")