from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import transform_text


def add_transformed_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transformed_text"] = df["text"].apply(transform_text)
    return df


def fit_tfidf(df: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    """
    Equivalent to:
      tfidf = TfidfVectorizer()
      X = tfidf.fit_transform(df['transformed_text']).toarray()
    We keep X as sparse for efficiency; model results are equivalent.
    """
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["transformed_text"])
    return tfidf, X
