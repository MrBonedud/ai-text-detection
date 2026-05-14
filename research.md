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

### Tasks

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

- [ ] Train/validation/test split
- [ ] Feature extraction / embeddings
- [ ] Baseline model training
- [ ] Performance evaluation
