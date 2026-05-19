"""
03_models.py
------------
Train Logistic Regression + SVM model on TF-IDF features and generate confusion matrix.
"""

from __future__ import annotations

import argparse
import os

import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DEFAULT_INPUT = os.path.join(PROJECT_DIR, "data", "processed", "tfidf_matrix.pkl")

DEFAULT_OUTPUT = os.path.join(
    PROJECT_DIR, "results", "confusion_matrices", "lr_cm.png"
)


def load_features(input_path: str) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Feature file not found: {input_path}")
    return joblib.load(input_path)


def train_logistic_regression(X_train, y_train):
    print("── Training Logistic Regression ──")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("   Model trained successfully")

    return model


def train_svm(X_train, y_train):
    print("── Training SVM (LinearSVC) ──")

    model = LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=42,
        dual=False
    )

    model.fit(X_train, y_train)
    print("   Model trained successfully")

    return model


def evaluate_model(model, X_test, y_test):
    print("\n── Evaluating Model ──")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f}")

    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred))

    return y_pred


def generate_confusion_matrix_plot(y_test, y_pred, output_path: str):
    print("\n── Generating Confusion Matrix ──")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print("── Loading features ──")
    features = load_features(args.input)

    X_train = features["X_train_tfidf"]
    X_test = features["X_test_tfidf"]
    y_train = features["y_train"]
    y_test = features["y_test"]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    # Logistic Regression
    model = train_logistic_regression(X_train, y_train)

    y_pred = evaluate_model(model, X_test, y_test)

    generate_confusion_matrix_plot(y_test, y_pred, args.output)

    print("\n✓ Done")


if __name__ == "__main__":
    main()