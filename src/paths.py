"""Repo-root paths for saved models (Streamlit Cloud and local runs)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.pkl"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
