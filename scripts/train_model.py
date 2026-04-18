import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data import convert_windows1252_to_utf8, load_and_clean_dataframe
from src.paths import DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH
from src.preprocess import ensure_nltk_resources
from src.train import TrainConfig, train_mnb_tfidf

DEFAULT_RAW_DIR = Path("data") / "raw"
DEFAULT_SPAM_CSV = DEFAULT_RAW_DIR / "spam.csv"
DEFAULT_SPAM_UTF8 = DEFAULT_RAW_DIR / "spam_utf8.csv"


def ensure_dataset(spam_csv_path: Path):
    if spam_csv_path.exists():
        return
    print(f"Dataset not found at {spam_csv_path}. Downloading from Kaggle...")
    DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "scripts/download_dataset.py",
            "--out-dir",
            str(DEFAULT_RAW_DIR),
        ]
    )


def main():
    p = argparse.ArgumentParser(
        description="Train TF-IDF + MultinomialNB spam classifier (notebook-equivalent)."
    )
    p.add_argument(
        "--input",
        default=str(DEFAULT_SPAM_CSV),
        help=f"Input spam.csv (default: {DEFAULT_SPAM_CSV})",
    )
    p.add_argument(
        "--utf8",
        default=str(DEFAULT_SPAM_UTF8),
        help=f"UTF-8 converted CSV output (default: {DEFAULT_SPAM_UTF8})",
    )
    p.add_argument(
        "--vectorizer",
        default=str(DEFAULT_VECTORIZER_PATH),
        help=f"Where to save TF-IDF vectorizer pickle (default: {DEFAULT_VECTORIZER_PATH})",
    )
    p.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Where to save MultinomialNB model pickle (default: {DEFAULT_MODEL_PATH})",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=2)
    args = p.parse_args()

    ensure_nltk_resources()

    input_path = Path(args.input)
    utf8_path = Path(args.utf8)

    ensure_dataset(input_path)

    # Convert encoding and load
    utf8_path.parent.mkdir(parents=True, exist_ok=True)
    convert_windows1252_to_utf8(str(input_path), str(utf8_path))
    df = load_and_clean_dataframe(str(utf8_path))

    cfg = TrainConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        vectorizer_path=args.vectorizer,
        model_path=args.model,
    )

    result = train_mnb_tfidf(df, cfg)

    print("Training complete.")
    print("Metrics:")
    print(result["metrics"])
    print(f"Saved vectorizer to: {result['vectorizer_path']}")
    print(f"Saved model to: {result['model_path']}")


if __name__ == "__main__":
    main()
