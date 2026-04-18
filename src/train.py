from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from .features import add_transformed_text, fit_tfidf

@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 2
    vectorizer_path: str = "vectorizer.pkl"
    model_path: str = "model.pkl"

def train_mnb_tfidf(df: pd.DataFrame, cfg: TrainConfig) -> Dict[str, Any]:
    """
      - TF-IDF
      - MultinomialNB
      - save vectorizer.pkl + model.pkl
    """
    df2 = add_transformed_text(df)

    tfidf, X = fit_tfidf(df2)
    y = df2["target"].values

    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    n_test = int(round(len(y_arr) * cfg.test_size))
    min_class = min(int(np.sum(y_arr == c)) for c in classes) if len(classes) else 0
    use_stratify = (
        len(classes) > 1
        and min_class >= 2
        and n_test >= len(classes)
        and (len(y_arr) - n_test) >= len(classes)
    )
    stratify = y if use_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Save artifacts using pickle (exactly as in notebook)
    with open(cfg.vectorizer_path, "wb") as f:
        pickle.dump(tfidf, f)
    with open(cfg.model_path, "wb") as f:
        pickle.dump(mnb, f)

    return {
        "metrics": metrics,
        "vectorizer_path": cfg.vectorizer_path,
        "model_path": cfg.model_path,
    }