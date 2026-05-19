"""
03_visualization.py
-------------------
Generate:
1. Word cloud for human text
2. Word cloud for AI text
3. Top TF-IDF terms per class bar chart
"""

from __future__ import annotations

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CLEAN_DATA = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "clean_data.csv"
)

TFIDF_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "tfidf_matrix.pkl"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "charts"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


print("Loading clean dataset...")
df = pd.read_csv(CLEAN_DATA)

human_text = " ".join(df[df["label"] == 0]["text"].astype(str))
ai_text = " ".join(df[df["label"] == 1]["text"].astype(str))


#Word Cloud: Human

print("Generating human word cloud...")

human_wc = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(human_text)

plt.figure(figsize=(12, 6))
plt.imshow(human_wc, interpolation="bilinear")
plt.axis("off")

human_path = os.path.join(
    RESULTS_DIR,
    "wordcloud_human.png"
)

plt.savefig(human_path, bbox_inches="tight")
plt.close()


#Word Cloud: AI 
print("Generating AI word cloud...")

ai_wc = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(ai_text)

plt.figure(figsize=(12, 6))
plt.imshow(ai_wc, interpolation="bilinear")
plt.axis("off")

ai_path = os.path.join(
    RESULTS_DIR,
    "wordcloud_ai.png"
)

plt.savefig(ai_path, bbox_inches="tight")
plt.close()

print("Loading TF-IDF artifact...")

payload = joblib.load(TFIDF_FILE)

X_train = payload["X_train_tfidf"]
y_train = payload["y_train"]
feature_names = payload["feature_names"]


# Separate classes
human_rows = X_train[y_train == 0]
ai_rows = X_train[y_train == 1]

# Mean TF-IDF score per term
human_means = human_rows.mean(axis=0).A1
ai_means = ai_rows.mean(axis=0).A1

# Top terms
top_n = 20

human_top_idx = human_means.argsort()[-top_n:]
ai_top_idx = ai_means.argsort()[-top_n:]

human_terms = [feature_names[i] for i in human_top_idx]
human_scores = human_means[human_top_idx]

ai_terms = [feature_names[i] for i in ai_top_idx]
ai_scores = ai_means[ai_top_idx]


fig, axes = plt.subplots(2, 1, figsize=(14, 12))

axes[0].barh(human_terms, human_scores)
axes[0].set_title("Top 20 TF-IDF Terms — Human Text")

axes[1].barh(ai_terms, ai_scores)
axes[1].set_title("Top 20 TF-IDF Terms — AI Text")

plt.tight_layout()

top_terms_path = os.path.join(
    RESULTS_DIR,
    "top_terms.png"
)

plt.savefig(top_terms_path)
plt.close()

print("\nDone.")
print(f"Saved:")
print(f" - {human_path}")
print(f" - {ai_path}")
print(f" - {top_terms_path}")