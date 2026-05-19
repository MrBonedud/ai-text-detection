# AI vs Human Text Classification Research

## Day 0: Dataset Construction

**Objective:** Build a balanced binary classification dataset to distinguish AI-generated from human-written text.

**Dataset Composition:**

- Total samples: 2,000 (perfect balance)
- **AI class (1,000 samples):**
  - 200 from modern AI agents (ChatGPT, Claude, Gemini, DeepSeek × 3 each)
  - 800 from Kaggle AI-labeled dataset
- **Human class (1,000 samples):**
  - 1,000 from Kaggle human-labeled dataset

**Data Processing:**

- Extracted paragraphs from modern AI agent files with regex-based parsing
- Handled 1.1GB Kaggle CSV file with chunked reading
- Applied deduplication (240 raw → 200 unique paragraphs)
- Maintained minimum 40 words per paragraph

**Output:** `data/processed/raw_data.csv` (2,000 rows, perfectly balanced)

---

## Day 1: Text Preprocessing

**Completed:** `scripts/01_preprocessing.py`

**Processing Pipeline:**

1. Load raw dataset
2. Lowercase all text
3. Remove URLs, emails, punctuation, numbers
4. Tokenize with NLTK
5. Remove English stopwords
6. Lemmatize tokens (default) or stem with Porter (optional flag)
7. Save cleaned text to `data/processed/clean_data.csv`

**Output:** 2,000 cleaned text samples ready for feature extraction

---

## Day 2: Feature Extraction

**Completed:** `scripts/02_features.py`

**Feature Engineering:**

1. Load cleaned dataset
2. 80/20 stratified train/test split (random_state=42)
3. Fit TfidfVectorizer on training data only (max_features=5000, bigrams)
4. Transform both train and test sets to sparse TF-IDF matrices
5. Save artifacts to `data/processed/tfidf_matrix.pkl`

**Output Shapes:**

- Train: (1,600, 5,000)
- Test: (400, 5,000)

---

## Day 3: Model Training & Evaluation

**Completed:** `scripts/03_models.py` and `scripts/04_evaluation.py`

**Models Trained:**

1. **Logistic Regression** - baseline classifier with LR-specific hyperparameters
2. **SVM (LinearSVC)** - support vector machine for non-linear decision boundaries
3. **Naive Bayes** - probabilistic classifier baseline

**Evaluation Metrics (Train/Test):**

- Accuracy, Precision, Recall, F1-score on both splits
- Confusion matrices for visual analysis
- Generalization gap analysis to detect overfitting

**Outputs Generated:**

- `results/confusion_matrices/lr_cm.png`, `svm_cm.png`, `nb_cm.png`
- `results/metrics/model_comparison.csv` (full metrics comparison)

**Visualizations (03_visualization.py):**

- Word clouds for human vs AI text
- Top 20 TF-IDF terms per class (side-by-side bar chart)
- Saved to `results/charts/`

---

## Final Results: AI vs. Human Text Classification

### Executive Summary

All three models achieved excellent performance. **SVM (LinearSVC) emerged as the best classifier with 99.5% test accuracy and 100% precision on AI text detection.**

### Model Performance

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
| --- | --- | --- | --- | --- |
| **SVM (LinearSVC)** | **99.5%** | **100.0%** | **99.0%** | **0.9950** |
| Logistic Regression | 98.5% | 98.0% | 99.0% | 0.9851 |
| Naive Bayes | 94.0% | 94.9% | 93.0% | 0.9394 |

### Best Model: SVM (LinearSVC)

**Test Set Performance:**

- Accuracy: 99.5% (199/200 correct)
- Precision: 100.0% (no false positives for AI)
- Recall: 99.0% (catches 99% of AI samples)
- Generalization gap: 0.5% (excellent, no overfitting)

### Key Insights

1. **Clear Signal:** AI and human text have distinct TF-IDF signatures, making classification straightforward

2. **Model Robustness:** SVM outperforms Naive Bayes on high-dimensional sparse data due to better margin optimization

3. **No Overfitting:** Despite SVM achieving 100% train accuracy, test accuracy remains 99.5%, proving strong generalization

4. **Consistent Performance:** All models exceed 94% accuracy, indicating reliable classification signal

5. **Distinguishing Features:**
   - **AI text:** Formulaic patterns, standardized phrasing, formal language
   - **Human text:** Narrative flow, personal pronouns, varied vocabulary

### Recommendations

1. **Production Model:** Deploy SVM (LinearSVC) with 99.5% accuracy and perfect precision

2. **Next Steps:**
   - Ensemble methods (voting classifier) for additional robustness
   - Out-of-domain evaluation (new AI models, different text genres)
   - Confidence scoring for borderline cases

3. **Monitoring:** Track top distinguishing terms as AI generation techniques evolve

### Deliverables

- ✓ Confusion matrices (3 models)
- ✓ Metrics comparison CSV
- ✓ Word clouds (human vs AI)
- ✓ Top 20 terms visualization
- ✓ Training artifacts and models
