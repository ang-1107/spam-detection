import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

_ps = PorterStemmer()

def ensure_nltk_resources() -> None:
    """Download NLTK data needed for tokenization and stopwords (handles NLTK 3.9+ punkt_tab)."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

def transform_text(text: str) -> str:
    """
    Steps:
      - lowercase
      - tokenize
      - keep alphanumeric
      - remove stopwords + punctuation
      - Porter stemming
    """
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)

    y = []
    for tok in tokens:
        if tok.isalnum():
            y.append(tok)

    tokens = y[:]
    y.clear()

    stops = set(stopwords.words("english"))
    punct = set(string.punctuation)

    for tok in tokens:
        if tok not in stops and tok not in punct:
            y.append(tok)

    tokens = y[:]
    y.clear()

    for tok in tokens:
        y.append(_ps.stem(tok))

    return " ".join(y)