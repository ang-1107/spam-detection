from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Tuple, Union

from .paths import DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH
from .preprocess import transform_text

PathLike = Union[str, Path]


def load_artifacts(
    vectorizer_path: PathLike = DEFAULT_VECTORIZER_PATH,
    model_path: PathLike = DEFAULT_MODEL_PATH,
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
