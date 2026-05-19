"""
04_evaluation.py
----------------
Train Logistic Regression + SVM and compare performance.
"""

from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models_utils import train_logistic_regression, train_svm, train_naive_bayes, load_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DEFAULT_FEATURES = os.path.join(PROJECT_DIR, "data", "processed", "tfidf_matrix.pkl")

DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "results", "metrics", "model_comparison.csv")


def calculate_metrics_both_sets(model, name, X_train, y_train, X_test, y_test):
    if model is None:
        return None

    results = []

    # Train
    train_pred = model.predict(X_train)
    results.append({
        "Model": name,
        "Set": "Train",
        "Accuracy": accuracy_score(y_train, train_pred),
        "Precision": precision_score(y_train, train_pred),
        "Recall": recall_score(y_train, train_pred),
        "F1": f1_score(y_train, train_pred),
    })

    # Test
    test_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Set": "Test",
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred),
        "Recall": recall_score(y_test, test_pred),
        "F1": f1_score(y_test, test_pred),
    })

    return results


def save(metrics, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(metrics)

    df.to_csv(output_path, index=False)

    print("\n── Results ──")
    print(df)
    print(f"\nSaved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features", default=DEFAULT_FEATURES)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print("── Loading features ──")
    data = load_features(args.features)

    X_train = data["X_train_tfidf"]
    X_test = data["X_test_tfidf"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("\n── Training Models ──")

    lr = train_logistic_regression(X_train, y_train)
    svm = train_svm(X_train, y_train)
    nb = train_naive_bayes(X_train, y_train)

    print("\n── Evaluating ──")

    metrics = []

    metrics += calculate_metrics_both_sets(lr, "Logistic Regression", X_train, y_train, X_test, y_test)
    metrics += calculate_metrics_both_sets(svm, "SVM (LinearSVC)", X_train, y_train, X_test, y_test)
    metrics += calculate_metrics_both_sets(nb, "Naive Bayes", X_train, y_train, X_test, y_test)

    save(metrics, args.output)

    print("\n✓ Done")


if __name__ == "__main__":
    main()