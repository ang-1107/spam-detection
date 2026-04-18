from __future__ import annotations

import pickle
from typing import Any, Tuple

from .preprocess import transform_text


def load_artifacts(
    vectorizer_path: str = "vectorizer.pkl",
    model_path: str = "model.pkl",
) -> Tuple[Any, Any]:
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def predict_text(text: str, vectorizer, model) -> int:
    """Return 0 = ham, 1 = spam (matches LabelEncoder ordering for this dataset)."""
    transformed = transform_text(text)
    X = vectorizer.transform([transformed])
    pred = model.predict(X)[0]
    return int(pred)
