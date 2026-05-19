"""
03_models.py
------------
Train models and generate confusion matrices.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models_utils import train_logistic_regression, train_svm, train_naive_bayes, load_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DEFAULT_INPUT = os.path.join(PROJECT_DIR, "data", "processed", "tfidf_matrix.pkl")

DEFAULT_OUTPUT = os.path.join(
    PROJECT_DIR, "results", "confusion_matrices", "lr_cm.png"
)


def generate_confusion_matrix_plot(model_name, y_test, y_pred, output_path: str):
    print(f"\n── Generating Confusion Matrix ({model_name}) ──")

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
    plt.title(f"Confusion Matrix - {model_name}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT)
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
    print("\n=== Logistic Regression ===")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_output = os.path.join(os.path.dirname(DEFAULT_OUTPUT), "lr_cm.png")
    generate_confusion_matrix_plot("Logistic Regression", y_test, lr_pred, lr_output)

    # SVM
    print("\n=== SVM (LinearSVC) ===")
    svm_model = train_svm(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_output = os.path.join(os.path.dirname(DEFAULT_OUTPUT), "svm_cm.png")
    generate_confusion_matrix_plot("SVM (LinearSVC)", y_test, svm_pred, svm_output)

    # Naive Bayes
    print("\n=== Naive Bayes ===")
    nb_model = train_naive_bayes(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_output = os.path.join(os.path.dirname(DEFAULT_OUTPUT), "nb_cm.png")
    generate_confusion_matrix_plot("Naive Bayes", y_test, nb_pred, nb_output)

    print("\n✓ Done")


if __name__ == "__main__":
    main()