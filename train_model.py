"""Training script for the Email Spam Detector.

Loads a labelled dataset (CSV), preprocesses text, trains a TF-IDF +
Multinomial Naïve Bayes pipeline, evaluates on a held-out test split,
and persists the trained model and vectoriser to disk.

Usage:
    python train_model.py                           # uses default dataset path
    python train_model.py --data_path dataset/spam.csv
"""

import argparse
import os
import pickle
import re
import string
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join("dataset", "spam.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def _ensure_nltk_data():
    """Download NLTK stopwords if not already present."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def preprocess_text(text: str) -> str:
    """Clean and normalise a single text string.

    Steps:
        1. Lower-case
        2. Remove URLs
        3. Remove punctuation
        4. Remove digits
        5. Remove stop-words
        6. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)          # URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)                        # digits
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def load_dataset(data_path: str) -> pd.DataFrame:
    """Load a spam dataset CSV and return a DataFrame with 'label' and 'text'."""
    # Try tab-separated first (SMS Spam Collection format), fall back to comma
    for sep in ["\t", ","]:
        try:
            df = pd.read_csv(data_path, sep=sep, encoding="latin-1")
            if df.shape[1] >= 2:
                break
        except Exception:
            continue
    else:
        sys.exit(f"ERROR: Could not parse {data_path}. Expected CSV with >= 2 columns.")

    # Normalise column names — handle various header styles
    df = df.iloc[:, :2]  # keep only first two columns
    df.columns = ["label", "text"]
    df["label"] = df["label"].str.strip().str.lower()

    # Map common label variants
    label_map = {"ham": "ham", "spam": "spam", "0": "ham", "1": "spam"}
    df["label"] = df["label"].map(label_map)
    df.dropna(subset=["label", "text"], inplace=True)

    print(f"[OK] Loaded {len(df)} samples from {data_path}")
    print(f"     Ham : {(df['label'] == 'ham').sum()}")
    print(f"     Spam: {(df['label'] == 'spam').sum()}")
    return df


def train(data_path: str = DEFAULT_DATA_PATH):
    """End-to-end training pipeline."""
    _ensure_nltk_data()

    # 1. Load
    df = load_dataset(data_path)

    # 2. Preprocess
    print("\n[*] Preprocessing text ...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    # 3. Vectorise
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # 5. Train
    print("\n[*] Training Multinomial Naive Bayes ...")
    # Use lower alpha and balanced class priors to handle imbalanced data
    spam_ratio = (y_train == 'spam').sum() / len(y_train)
    ham_ratio = 1.0 - spam_ratio
    # Give spam class more prior weight to compensate for imbalance
    model = MultinomialNB(alpha=0.1, class_prior=[ham_ratio * 0.85, spam_ratio * 2.0])
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Results ---")
    print(f"    Accuracy : {accuracy:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['ham', 'spam'], zero_division=0)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Save artefacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\n[SAVED] Model      -> {MODEL_PATH}")
    print(f"[SAVED] Vectorizer -> {VECTORIZER_PATH}")

    return accuracy


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the spam detection model."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the labelled CSV dataset (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()
    train(args.data_path)
