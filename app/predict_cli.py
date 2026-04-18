import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.preprocess import ensure_nltk_resources
from src.inference import load_artifacts, predict_text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--vectorizer", default="vectorizer.pkl")
    p.add_argument("--model", default="model.pkl")
    args = p.parse_args()

    ensure_nltk_resources()
    vectorizer, model = load_artifacts(args.vectorizer, args.model)
    pred = predict_text(args.text, vectorizer, model)
    print(pred)

if __name__ == "__main__":
    main()