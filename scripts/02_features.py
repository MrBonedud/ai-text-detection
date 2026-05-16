"""
02_features.py
--------------
Build TF-IDF features from the cleaned dataset, then create an 80/20
stratified train/test split and persist the resulting artifacts.

Output:
    data/processed/tfidf_matrix.pkl
"""

from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DEFAULT_INPUT = os.path.join(PROJECT_DIR, "data", "processed", "clean_data.csv")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "data", "processed", "tfidf_matrix.pkl")


def load_dataset(input_path: str) -> pd.DataFrame:
    """Load the cleaned dataset and normalize the expected columns."""
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        for candidate in ("content", "body"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "text"})
                break

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the cleaned dataset")
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the cleaned dataset")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TF-IDF feature matrices")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input cleaned CSV path")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output pickle path")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    print("── Loading cleaned dataset ──")
    df = load_dataset(input_path)
    print(f"   Loaded {len(df)} rows")

    X = df["text"]
    y = df["label"]

    print("\n── Splitting dataset ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(f"   Train rows: {len(X_train)}")
    print(f"   Test rows : {len(X_test)}")

    print("\n── Building TF-IDF features ──")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"   Train matrix shape: {X_train_tfidf.shape}")
    print(f"   Test matrix shape : {X_test_tfidf.shape}")
    print(f"   Vocabulary size   : {len(vectorizer.get_feature_names_out())}")

    payload = {
        "X_train_tfidf": X_train_tfidf,
        "X_test_tfidf": X_test_tfidf,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "vectorizer": vectorizer,
        "feature_names": vectorizer.get_feature_names_out(),
    }

    print("\n── Saving TF-IDF artifact ──")
    joblib.dump(payload, output_path)
    print(f"   Saved to {output_path}")


if __name__ == "__main__":
    main()