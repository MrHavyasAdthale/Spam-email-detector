"""Prediction module for the Email Spam Detector.

Loads the trained Multinomial Naïve Bayes model and TF-IDF vectoriser,
applies the same preprocessing used during training, and returns a
prediction with a confidence score.

Usage:
    from predict import predict
    label, confidence = predict("Congratulations! You've won a free iPhone!")
    # label = "spam", confidence = 97.3
"""

import os
import pickle
import re
import string

import nltk
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_model = None
_vectorizer = None
_model_loaded = False


def _ensure_nltk_data():
    """Download NLTK stopwords if not already present."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def preprocess_text(text: str) -> str:
    """Clean and normalise a single text string.

    Must mirror the preprocessing in train_model.py exactly.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    _ensure_nltk_data()
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


def _load_model():
    """Load the model and vectoriser from disk (once)."""
    global _model, _vectorizer, _model_loaded

    if _model_loaded:
        return _model is not None

    _model_loaded = True

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print(
            "⚠️  Model files not found. Please run `python train_model.py` first."
        )
        return False

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        _vectorizer = pickle.load(f)
    return True


def predict(text: str) -> tuple[str, float]:
    """Classify *text* as spam or ham.

    Returns:
        (label, confidence)  where label is ``"spam"`` or ``"ham"`` and
        confidence is a percentage (0-100).
    """
    if not _load_model():
        # Fallback: simple keyword heuristic when model is unavailable
        return _keyword_fallback(text)

    clean = preprocess_text(text)
    X = _vectorizer.transform([clean])
    label = _model.predict(X)[0]
    proba = _model.predict_proba(X)[0]
    confidence = round(max(proba) * 100, 1)
    return label, confidence


def predict_batch(texts: list[str]) -> list[tuple[str, float]]:
    """Classify a batch of texts at once (more efficient)."""
    if not _load_model():
        return [_keyword_fallback(t) for t in texts]

    cleaned = [preprocess_text(t) for t in texts]
    X = _vectorizer.transform(cleaned)
    labels = _model.predict(X)
    probas = _model.predict_proba(X)
    return [
        (label, round(max(proba) * 100, 1))
        for label, proba in zip(labels, probas)
    ]


# ---------------------------------------------------------------------------
# Keyword fallback (used only when trained model is unavailable)
# ---------------------------------------------------------------------------
_SPAM_KEYWORDS = [
    "free", "win", "winner", "prize", "limited", "offer",
    "click", "buy now", "subscribe", "credit", "loan",
    "congratulations", "urgent", "act now", "cash",
]


def _keyword_fallback(text: str) -> tuple[str, float]:
    """Heuristic-only prediction when no model is available."""
    if not text:
        return "ham", 50.0
    t = text.lower()
    hits = sum(1 for kw in _SPAM_KEYWORDS if kw in t)
    if hits >= 2:
        return "spam", min(60.0 + hits * 8, 95.0)
    elif hits == 1:
        return "spam", 55.0
    return "ham", 70.0
