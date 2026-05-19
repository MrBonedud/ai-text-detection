"""
03_models.py
------------
Train Logistic Regression model on TF-IDF features and generate confusion matrix.

This script:
1. Loads the pre-processed TF-IDF features from 02_features.py
2. Trains a Logistic Regression model
3. Generates predictions on the test set
4. Creates and saves a confusion matrix visualization as PNG

Output:
    results/confusion_matrices/lr_cm.png
"""

from __future__ import annotations

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DEFAULT_INPUT = os.path.join(PROJECT_DIR, "data", "processed", "tfidf_matrix.pkl")
DEFAULT_OUTPUT = os.path.join(
    PROJECT_DIR, "results", "confusion_matrices", "lr_cm.png"
)


def load_features(input_path: str) -> dict:
    """Load the TF-IDF features and train/test split from pickle."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Feature file not found: {input_path}\n"
            "Please run 02_features.py first."
        )
    return joblib.load(input_path)


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
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
    """Train SVM (LinearSVC) model.
    
    TODO for partner:
    - Initialize LinearSVC with appropriate hyperparameters (C, max_iter, random_state, dual)
    - Fit the model on X_train and y_train
    - Print "Model trained successfully"
    - Return the fitted model
    """
    print("── Training SVM (LinearSVC) ──")
    # TODO: Implement SVM training here
    model = None
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return predictions and metrics."""
    print("\n── Evaluating Model ──")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f}")
    
    print("\n── Classification Report ──")
    report = classification_report(y_test, y_pred)
    print(report)
    
    return y_pred


def generate_confusion_matrix_plot(y_test, y_pred, output_path: str):
    """Generate and save confusion matrix visualization."""
    print("\n── Generating Confusion Matrix ──")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure and plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   Confusion matrix saved to {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression and generate confusion matrix"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="Input TF-IDF pickle path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="Output confusion matrix PNG path",
    )
    args = parser.parse_args()

    # Load features
    print("── Loading TF-IDF features ──")
    features = load_features(args.input)
    X_train_tfidf = features["X_train_tfidf"]
    X_test_tfidf = features["X_test_tfidf"]
    y_train = features["y_train"]
    y_test = features["y_test"]
    print(f"   Train set: {X_train_tfidf.shape}")
    print(f"   Test set : {X_test_tfidf.shape}")

    # Train model
    model = train_logistic_regression(X_train_tfidf, y_train)

    # Evaluate
    y_pred = evaluate_model(model, X_test_tfidf, y_test)

    # Generate confusion matrix
    generate_confusion_matrix_plot(y_test, y_pred, args.output)

    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
