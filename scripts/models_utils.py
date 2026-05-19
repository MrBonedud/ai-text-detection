"""
models_utils.py
---------------
Shared model training and feature loading utilities.
"""

from __future__ import annotations

import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


def load_features(input_path: str) -> dict:
    """Load preprocessed features from pickle file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Feature file not found: {input_path}")
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
    """Train SVM (LinearSVC) model."""
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


def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model."""
    print("── Training Naive Bayes ──")

    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("   Model trained successfully")

    return model
