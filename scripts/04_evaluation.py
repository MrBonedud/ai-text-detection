"""
04_evaluation.py
----------------
Comprehensive model evaluation and comparison.

This script:
1. Loads TF-IDF features
2. Trains Logistic Regression and SVM models
3. Generates metrics (accuracy, precision, recall, F1) for both
4. Saves comparison report to CSV

Output:
    results/metrics/model_comparison.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DEFAULT_FEATURES = os.path.join(PROJECT_DIR, "data", "processed", "tfidf_matrix.pkl")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "results", "metrics", "model_comparison.csv")


def load_features(input_path: str) -> dict:
    """Load the TF-IDF features and train/test split from pickle."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Feature file not found: {input_path}\n"
            "Please run 02_features.py first."
        )
    return joblib.load(input_path)


def train_logistic_regression(X_train, y_train) -> object:
    """Train Logistic Regression model."""
    print("   ├─ Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train) -> object:
    """Train SVM (LinearSVC) model.
    
    TODO for partner:
    - Initialize LinearSVC with appropriate hyperparameters (C, max_iter, random_state, dual)
    - Fit the model on X_train and y_train
    - Return the fitted model
    """
    print("   ├─ Training SVM (LinearSVC)...")
    # TODO: Implement SVM training here
    # Example stub - remove and implement actual SVM
    model = LinearSVC(
        max_iter=2000,
        random_state=42,
        dual=False,
    )
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"   ├─ ⚠ SVM training failed: {e}")
        return None


def calculate_metrics(model: object, model_name: str, X_test, y_test) -> dict:
    """Calculate evaluation metrics for a model."""
    if model is None:
        return None
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }
    
    return metrics


def calculate_metrics_both_sets(
    model: object, model_name: str, X_train, y_train, X_test, y_test
) -> list:
    """Calculate metrics on both train and test sets to detect overfitting."""
    if model is None:
        return None
    
    results = []
    
    # Train metrics
    y_train_pred = model.predict(X_train)
    train_metrics = {
        "Model": model_name,
        "Set": "Train",
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "Precision": precision_score(y_train, y_train_pred, zero_division=0),
        "Recall": recall_score(y_train, y_train_pred, zero_division=0),
        "F1": f1_score(y_train, y_train_pred, zero_division=0),
    }
    results.append(train_metrics)
    
    # Test metrics
    y_test_pred = model.predict(X_test)
    test_metrics = {
        "Model": model_name,
        "Set": "Test",
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Recall": recall_score(y_test, y_test_pred, zero_division=0),
        "F1": f1_score(y_test, y_test_pred, zero_division=0),
    }
    results.append(test_metrics)
    
    return results


def save_comparison(metrics_list: list, output_path: str) -> None:
    """Save model comparison to CSV."""
    print("\n── Saving Results ──")
    
    # Filter out None entries (failed models)
    metrics_list = [m for m in metrics_list if m is not None]
    
    if not metrics_list:
        print("   ⚠ No successful model evaluations to save.")
        return
    
    df = pd.DataFrame(metrics_list)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"   ✓ Comparison saved to {output_path}")
    
    # Print table
    print("\n── Model Comparison ──")
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare multiple models"
    )
    parser.add_argument(
        "--features",
        "-f",
        default=DEFAULT_FEATURES,
        help="Input TF-IDF features pickle path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="Output comparison CSV path",
    )
    args = parser.parse_args()

    # Load features
    print("── Loading TF-IDF Features ──")
    features = load_features(args.features)
    X_train_tfidf = features["X_train_tfidf"]
    X_test_tfidf = features["X_test_tfidf"]
    y_train = features["y_train"]
    y_test = features["y_test"]
    print(f"   ✓ Loaded train: {X_train_tfidf.shape}, test: {X_test_tfidf.shape}")

    # Train models
    print("\n── Training Models ──")
    lr_model = train_logistic_regression(X_train_tfidf, y_train)
    svm_model = train_svm(X_train_tfidf, y_train)

    # Evaluate models on both train and test sets
    print("\n── Evaluating Models ──")
    metrics_list = []
    
    lr_results = calculate_metrics_both_sets(
        lr_model, "Logistic Regression", X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    if lr_results:
        metrics_list.extend(lr_results)
    
    svm_results = calculate_metrics_both_sets(
        svm_model, "SVM (LinearSVC)", X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    if svm_results:
        metrics_list.extend(svm_results)

    # Save comparison
    save_comparison(metrics_list, args.output)

    print("\n✓ Evaluation pipeline complete!")


if __name__ == "__main__":
    main()
