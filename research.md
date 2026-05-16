# AI vs Human Text Classification Research

## Day 0: Dataset Construction

### Objective

Build a balanced binary classification dataset to distinguish between AI-generated and human-written text.

### Dataset Composition

- **Total samples:** 2,000
- **AI class (label=1):** 1,000 samples
  - Modern AI agents: 200 (ChatGPT, Claude, Gemini, DeepSeek × 3 each)
  - Kaggle AI-labeled: 800
- **Human class (label=0):** 1,000 samples
  - Kaggle human-labeled: 1,000

### Technical Implementation

#### Data Sources

1. **Modern AI samples** (`data/raw/samples/ai_generated/`)
   - 12 text files (4 AI agents × 3 samples each)
   - Regex-based paragraph extraction handling multiple output formats
   - Deduplication applied (240 raw → 200 unique paragraphs)
   - Minimum length: 40 words per paragraph

2. **Kaggle AI_Human.csv**
   - Chunked reading (10,000 rows at a time) to handle 1.1GB file size
   - Early-exit optimization to avoid full scan
   - Separated by label: AI (1) and human (0)

#### Processing Pipeline

- Extract paragraphs from all modern agent files
- Calculate Kaggle AI samples needed (1000 total - 200 modern = 800)
- Load required Kaggle samples from CSV
- Merge AI sources (modern + Kaggle)
- Combine with human samples
- Shuffle dataset
- Save to `data/processed/raw_data.csv`

### Output

- **File:** `data/processed/raw_data.csv`
- **Schema:** `text, label` (CSV format)
- **Class distribution:** Perfect balance (1000 each)

### Notes

- Mixed-quality AI samples intentionally kept; quirks and oddities help train robust classifiers
- Dataset composition balances high-quality modern agents with broader Kaggle coverage
- No synthetic upsampling—all real, diverse samples

---

## Day 1: Text Preprocessing (`01_preprocessing.py`)

### Day 1 Tasks

- [ ] Load `raw_data.csv`
- [ ] Lowercase text
- [ ] Remove URLs, punctuation, numbers
Completed: implemented `scripts/01_preprocessing.py` which performs the full cleaning pipeline and writes `data/processed/clean_data.csv`.

- **Steps implemented:**
  - Load `data/processed/raw_data.csv` (auto-detects `text`/`content`/`body` column names)
  - Lowercase
  - Remove URLs and email addresses
  - Remove punctuation and numbers (keeps letters and spaces)
  - Tokenize with NLTK
  - Remove NLTK English stopwords
  - Lemmatize tokens (default) and optionally stem with Porter if `--use-stemmer` is passed
  - Rejoin tokens and save to `data/processed/clean_data.csv`

**Implementation notes:**

- The script auto-downloads required NLTK data (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`) on first run.
- Lemmatization is the default because it preserves real words (better interpretability and model features); stemming is available as an opt-in flag (`--use-stemmer`) for cases where aggressive vocabulary reduction is desired.
- The script provides a small before/after sample preview and prints dataset statistics (counts, average length, tokens/sample).

**Usage examples:**

```python
python scripts/01_preprocessing.py
python scripts/01_preprocessing.py --use-stemmer
python scripts/01_preprocessing.py -i data/processed/raw_data.csv -o data/processed/clean_data.csv
```

- [ ] Remove stopwords (nltk)
- [ ] Stem or lemmatize tokens
- [ ] Save to `data/processed/clean_data.csv`
- [ ] Screenshot before/after sample → hand off to partner

### Next Steps

- [x] Train/validation/test split
- [x] Feature extraction / embeddings
- [ ] Baseline model training
- [ ] Performance evaluation

---

## Day 2: Feature Extraction (`02_features.py`)

### Day 2 Tasks

Completed: implemented `scripts/02_features.py` to build TF-IDF features from `data/processed/clean_data.csv`, using an 80/20 stratified split with `random_state=42`.

- **Steps implemented:**
  - Load `data/processed/clean_data.csv` (auto-detects `text`/`content`/`body` column names)
  - Split the dataset with `train_test_split(test_size=0.2, stratify=y, random_state=42)`
  - Fit a `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` on the training split only
  - Transform both train and test text into sparse TF-IDF matrices
  - Save matrices, labels, vectorizer, and feature names to `data/processed/tfidf_matrix.pkl`

**Output:**

- **File:** `data/processed/tfidf_matrix.pkl`
- **Contents:** `X_train_tfidf`, `X_test_tfidf`, `y_train`, `y_test`, `vectorizer`, `feature_names`
- **Observed shapes:** train `(1600, 5000)`, test `(400, 5000)`

**Implementation notes:**

- The vectorizer is fit only on the training fold to avoid leakage.
- The artifact is saved with `joblib` so it can be reused directly for model training.
