from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier

from src.data import convert_windows1252_to_utf8, load_and_clean_dataframe
from src.preprocess import ensure_nltk_resources
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

import subprocess

DEFAULT_RAW_DIR = Path("data") / "raw"
DEFAULT_SPAM_CSV = DEFAULT_RAW_DIR / "spam.csv"
DEFAULT_SPAM_UTF8 = DEFAULT_RAW_DIR / "spam_utf8.csv"

def ensure_dataset(spam_csv_path: Path):
    if spam_csv_path.exists():
        return
    print(f"Dataset not found at {spam_csv_path}. Downloading from Kaggle...")
    DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "scripts/download_dataset.py", "--out-dir", str(DEFAULT_RAW_DIR)])


@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    tokenize: bool = True
    keep_alnum_only: bool = True
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    stemming: bool = True


_ps = PorterStemmer()


def transform_text_variant(text: str, cfg: PreprocessConfig) -> str:
    """
    Generalized version of your notebook's transform_text(), with ablation toggles.
    """
    text = "" if text is None else str(text)
    if cfg.lowercase:
        text = text.lower()

    if cfg.tokenize:
        tokens = nltk.word_tokenize(text)
    else:
        tokens = text.split()

    if cfg.keep_alnum_only:
        tokens = [t for t in tokens if t.isalnum()]

    stops = set(stopwords.words("english")) if cfg.remove_stopwords else set()
    punct = set(string.punctuation) if cfg.remove_punctuation else set()

    filtered = []
    for t in tokens:
        if (not cfg.remove_stopwords or t not in stops) and (not cfg.remove_punctuation or t not in punct):
            filtered.append(t)

    if cfg.stemming:
        filtered = [_ps.stem(t) for t in filtered]

    return " ".join(filtered)


def build_vectorizer(vec_name: str, params: Dict[str, Any]):
    if vec_name == "count":
        return CountVectorizer(**params)
    if vec_name == "tfidf":
        return TfidfVectorizer(**params)
    raise ValueError(f"Unknown vectorizer: {vec_name}")


def build_model(model_name: str, params: Dict[str, Any]):
    if model_name == "GaussianNB":
        return GaussianNB(**params)
    if model_name == "MultinomialNB":
        return MultinomialNB(**params)
    if model_name == "BernoulliNB":
        return BernoulliNB(**params)
    if model_name == "LogReg":
        return LogisticRegression(**params)
    if model_name == "LinearSVC":
        return LinearSVC(**params)
    if model_name == "SVC_sigmoid":
        return SVC(**params)
    if model_name == "KNN":
        return KNeighborsClassifier(**params)
    if model_name == "DT":
        return DecisionTreeClassifier(**params)
    if model_name == "RF":
        return RandomForestClassifier(**params)
    if model_name == "AdaBoost":
        return AdaBoostClassifier(**params)
    if model_name == "Bagging":
        return BaggingClassifier(**params)
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(**params)
    if model_name == "GBDT":
        return GradientBoostingClassifier(**params)
    if model_name == "XGB":
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def iter_param_grid(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def evaluate_run(
    X_text: np.ndarray,
    y: np.ndarray,
    preprocess_cfg: PreprocessConfig,
    vec_name: str,
    vec_params: Dict[str, Any],
    model_name: str,
    model_params: Dict[str, Any],
    test_size: float,
    random_state: int,
    positive_label: int = 1,
) -> Dict[str, Any]:
    # preprocess -> transformed_text
    X_transformed = np.array([transform_text_variant(t, preprocess_cfg) for t in X_text], dtype=object)

    # split
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # vectorize
    vectorizer = build_vectorizer(vec_name, vec_params)
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)

    # model
    clf = build_model(model_name, model_params)

    # GaussianNB requires dense
    if model_name == "GaussianNB":
        X_train_fit = X_train.toarray()
        X_test_eval = X_test.toarray()
    else:
        X_train_fit = X_train
        X_test_eval = X_test

    clf.fit(X_train_fit, y_train)
    y_pred = clf.predict(X_test_eval)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark vectorizers + models + ablations for SMS spam detection.")
    p.add_argument(
        "--input",
        default=str(DEFAULT_SPAM_CSV),
        help=f"Input spam.csv (windows-1252), default: {DEFAULT_SPAM_CSV}",
    )
    p.add_argument(
        "--utf8",
        default=str(DEFAULT_SPAM_UTF8),
        help=f"Converted UTF-8 CSV, default: {DEFAULT_SPAM_UTF8}",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=2)
    p.add_argument("--max-runs", type=int, default=0, help="Optional cap on runs (0 = no cap).")
    p.add_argument("--out-csv", default="benchmark_results.csv")
    p.add_argument("--metric", choices=["precision", "f1", "accuracy"], default="precision")
    args = p.parse_args()

    ensure_nltk_resources()

    input_path = Path(args.input)
    utf8_path = Path(args.utf8)
    ensure_dataset(input_path)

    utf8_path.parent.mkdir(parents=True, exist_ok=True)
    # load data (equivalent to notebook)
    convert_windows1252_to_utf8(str(input_path), str(utf8_path))
    df = load_and_clean_dataframe(str(utf8_path))

    X_text = df["text"].astype(str).values
    y = df["target"].values

    # --- Ablations for preprocessing (toggle significant steps) ---
    preprocess_grid = [
        PreprocessConfig(remove_stopwords=True, stemming=True, keep_alnum_only=True),
        PreprocessConfig(remove_stopwords=True, stemming=False, keep_alnum_only=True),
        PreprocessConfig(remove_stopwords=False, stemming=True, keep_alnum_only=True),
        PreprocessConfig(remove_stopwords=False, stemming=False, keep_alnum_only=True),
        # more variants:
        PreprocessConfig(remove_stopwords=True, stemming=True, keep_alnum_only=False),
        PreprocessConfig(remove_stopwords=True, stemming=False, keep_alnum_only=False),
    ]

    # --- Vectorizers + hyperparams ---
    vectorizer_grids: List[Tuple[str, Dict[str, List[Any]]]] = [
        ("count", {
            "ngram_range": [(1, 1), (1, 2)],
            "max_features": [None, 3000, 5000],
            "min_df": [1, 2],
        }),
        ("tfidf", {
            "ngram_range": [(1, 1), (1, 2)],
            "max_features": [None, 3000, 5000],
            "min_df": [1, 2],
            "use_idf": [True],
            "sublinear_tf": [False, True],
        }),
    ]

    # --- Models + hyperparams ---
    model_grids: List[Tuple[str, Dict[str, List[Any]]]] = [
        ("GaussianNB", {
            # no major hyperparams
        }),
        ("MultinomialNB", {
            "alpha": [0.1, 0.5, 1.0],
        }),
        ("BernoulliNB", {
            "alpha": [0.1, 0.5, 1.0],
        }),
        ("LogReg", {
            "solver": ["liblinear"],
            "penalty": ["l1", "l2"],
            "C": [0.5, 1.0, 2.0],
            "max_iter": [1000],
        }),
        ("LinearSVC", {
            "C": [0.5, 1.0, 2.0],
        }),
        ("SVC_sigmoid", {
            "kernel": ["sigmoid"],
            "gamma": [1.0, "scale"],
            "C": [1.0, 2.0],
        }),
        ("KNN", {
            "n_neighbors": [3, 5, 7],
        }),
        ("DT", {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        }),
        ("RF", {
            "n_estimators": [50, 200],
            "random_state": [args.random_state],
            "max_depth": [None, 10],
        }),
        ("ExtraTrees", {
            "n_estimators": [50, 200],
            "random_state": [args.random_state],
            "max_depth": [None, 10],
        }),
        ("AdaBoost", {
            "n_estimators": [50, 200],
            "random_state": [args.random_state],
            "learning_rate": [0.5, 1.0],
        }),
        ("Bagging", {
            "n_estimators": [50, 200],
            "random_state": [args.random_state],
        }),
        ("GBDT", {
            "n_estimators": [50, 200],
            "random_state": [args.random_state],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3],
        }),
    ]

    model_grids.append(
        (
            "XGB",
            {
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "random_state": [args.random_state],
                "eval_metric": ["logloss"],
                "n_jobs": [-1],
            },
        )
    )

    runs: List[Dict[str, Any]] = []
    best = None
    best_score = -1.0

    run_count = 0
    start = time.time()

    for pp_cfg in preprocess_grid:
        for vec_name, vec_grid in vectorizer_grids:
            for vec_params in iter_param_grid(vec_grid):
                for model_name, model_grid in model_grids:
                    # Some models have empty grids; iterate once
                    model_param_iter = [dict()] if not model_grid else list(iter_param_grid(model_grid))

                    for model_params in model_param_iter:
                        run_count += 1
                        if args.max_runs and run_count > args.max_runs:
                            break

                        run_id = run_count
                        t0 = time.time()

                        try:
                            metrics = evaluate_run(
                                X_text=X_text,
                                y=y,
                                preprocess_cfg=pp_cfg,
                                vec_name=vec_name,
                                vec_params=vec_params,
                                model_name=model_name,
                                model_params=model_params,
                                test_size=args.test_size,
                                random_state=args.random_state,
                                positive_label=1,
                            )
                            score = metrics[args.metric]
                            status = "ok"
                            error = ""
                        except Exception as e:
                            metrics = {
                                "accuracy": None,
                                "precision": None,
                                "f1": None,
                                "confusion_matrix": None,
                                "n_train": None,
                                "n_test": None,
                            }
                            score = None
                            status = "error"
                            error = repr(e)

                        dt = time.time() - t0

                        row = {
                            "run_id": run_id,
                            "status": status,
                            "error": error,
                            "metric": args.metric,
                            "score": score,
                            "accuracy": metrics["accuracy"],
                            "precision": metrics["precision"],
                            "f1": metrics["f1"],
                            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                            "preprocess": json.dumps(asdict(pp_cfg)),
                            "vectorizer": vec_name,
                            "vectorizer_params": json.dumps(vec_params),
                            "model": model_name,
                            "model_params": json.dumps(model_params),
                            "seconds": dt,
                        }
                        runs.append(row)

                        if status == "ok" and score is not None and score > best_score:
                            best_score = float(score)
                            best = {
                                "preprocess": asdict(pp_cfg),
                                "vectorizer": vec_name,
                                "vectorizer_params": vec_params,
                                "model": model_name,
                                "model_params": model_params,
                                "metrics": metrics,
                                "selected_metric": args.metric,
                                "selected_score": best_score,
                            }

                        # Light progress output
                        if run_id % 25 == 0:
                            elapsed = time.time() - start
                            print(f"[{run_id}] elapsed={elapsed:.1f}s best_{args.metric}={best_score:.4f}")

                    if args.max_runs and run_count >= args.max_runs:
                        break
                if args.max_runs and run_count >= args.max_runs:
                    break
            if args.max_runs and run_count >= args.max_runs:
                break
        if args.max_runs and run_count >= args.max_runs:
            break

    out_df = pd.DataFrame(runs)
    out_df.to_csv(args.out_csv, index=False)

    print("\n=== BEST TUPLE ===")
    if best is None:
        print("No successful runs.")
        return

    # Print as a single best tuple-like object
    best_tuple = (
        ("preprocess", best["preprocess"]),
        ("vectorizer", best["vectorizer"]),
        ("vectorizer_params", best["vectorizer_params"]),
        ("model", best["model"]),
        ("model_params", best["model_params"]),
        ("metrics", best["metrics"]),
        ("selected_metric", best["selected_metric"]),
        ("selected_score", best["selected_score"]),
    )
    print(best_tuple)

    print(f"\nSaved all results to: {args.out_csv}")
    print(f"Total runs attempted: {len(runs)}")


if __name__ == "__main__":
    main()