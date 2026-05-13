"""
build_dataset.py
----------------
Parses all AI-generated .txt files and merges them with the Kaggle human-text
dataset into a single balanced raw_data.csv.

Expected folder layout (relative to the project root):
        data/
            raw/
                samples/
                    ai_generated/   <- all 12 .txt files go here
                    kaggle/         <- kaggle CSV goes here
            processed/          <- output raw_data.csv written here
"""

import os
import re
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR     = os.path.dirname(BASE_DIR)
AI_DIR          = os.path.join(PROJECT_DIR, "data", "raw", "samples", "ai_generated")
KAGGLE_DIR      = os.path.join(PROJECT_DIR, "data", "raw", "samples", "kaggle")
OUTPUT_DIR      = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "raw_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sample_to_target(df: pd.DataFrame, target_count: int, label_name: str) -> pd.DataFrame:
    """Sample up to target_count rows (no upsampling—use what's available)."""
    if df.empty:
        raise ValueError(f"No rows available for {label_name} samples")

    actual_count = min(len(df), target_count)
    sampled = df.sample(n=actual_count, random_state=42).reset_index(drop=True)
    print(f"    Sampled {actual_count} {label_name} rows")
    return sampled


# ── Paragraph parser ───────────────────────────────────────────────────────
def parse_paragraphs(filepath: str) -> list[str]:
    """
    Extracts numbered paragraphs from a txt file.

    Handles all four agent formats:
      • ChatGPT  : "1. Text here..."
      • Gemini   : "1.  \\nText here..."  (number then newline)
      • DeepSeek : "1. **Bold title** text..." + optional "Run N" header
      • Claude   : markdown headers (##), bold numbers (**1.**), titled sections
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # Normalise line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Remove markdown headings and horizontal rules (Claude format)
    text = re.sub(r'^#{1,6}.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-{2,}$',    '', text, flags=re.MULTILINE)

    # Remove "Run N" header lines (DeepSeek format)
    text = re.sub(r'^Run\s+\d+\s*$', '', text, flags=re.MULTILINE)

    # Split on paragraph numbers: matches "1." … "20." with optional bold (**)
    # Works whether the text follows on the same line or the next line
    chunks = re.split(r'\n?\s*\*{0,2}\d{1,2}\.\*{0,2}\s*\n?', text)

    paragraphs = []
    for chunk in chunks:
        # Strip bold markers, inline titles (e.g. "Climate Change — "),
        # and excess whitespace
        chunk = re.sub(r'\*{1,2}', '', chunk)                  # remove ** or *
        chunk = re.sub(r'^[A-Z][^\n—–-]{0,60}[—–-]\s*', '', chunk)  # "Title — "
        chunk = re.sub(r'\s+', ' ', chunk).strip()

        # Keep only chunks long enough to be a real paragraph (> 40 words)
        if len(chunk.split()) > 40:
            paragraphs.append(chunk)

    return paragraphs


# ── Load AI samples ────────────────────────────────────────────────────────
def load_ai_samples() -> pd.DataFrame:
    """Load all AI samples from modern agent .txt files."""
    records = []
    txt_files = [f for f in os.listdir(AI_DIR) if f.endswith(".txt")]

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {AI_DIR}")

    for filename in sorted(txt_files):
        filepath = os.path.join(AI_DIR, filename)
        paragraphs = parse_paragraphs(filepath)
        print(f"    {filename:30s} → {len(paragraphs)} paragraphs")
        for para in paragraphs:
            records.append({"text": para, "label": 1, "source": filename})

    ai_df = pd.DataFrame(records).drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"    Total modern agent AI: {len(ai_df)} samples")
    return ai_df


# ── Load Kaggle AI and human samples ───────────────────────────────────────
def load_kaggle_samples(ai_target: int, human_target: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_files = [f for f in os.listdir(KAGGLE_DIR) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {KAGGLE_DIR}")

    ai_records = []
    human_records = []
    rows_scanned = 0
    
    for f in csv_files:
        filepath = os.path.join(KAGGLE_DIR, f)
        print(f"    Reading {f} in chunks...")
        
        # Read large CSV in chunks to avoid memory issues
        for chunk in pd.read_csv(filepath, chunksize=10000, low_memory=False, on_bad_lines='skip'):
            rows_scanned += len(chunk)
            
            # Find the text and label columns
            text_col = next(
                (c for c in chunk.columns if "text" in c.lower()),
                chunk.columns[0]
            )
            label_col = next(
                (c for c in chunk.columns if "label" in c.lower() or "generated" in c.lower()),
                None
            )
            
            if not label_col:
                print(f"    Warning: no label column found, skipping Kaggle samples")
                break
            
            # Filter AI rows (label == 1)
            chunk_ai = chunk[chunk[label_col] == 1].copy()
            chunk_ai = chunk_ai[[text_col]].rename(columns={text_col: "text"})
            chunk_ai = chunk_ai.dropna(subset=["text"])
            chunk_ai = chunk_ai[chunk_ai["text"].str.split().str.len() > 40]
            ai_records.append(chunk_ai)
            
            # Filter human rows (label == 0)
            chunk_human = chunk[chunk[label_col] == 0].copy()
            chunk_human = chunk_human[[text_col]].rename(columns={text_col: "text"})
            chunk_human = chunk_human.dropna(subset=["text"])
            chunk_human = chunk_human[chunk_human["text"].str.split().str.len() > 40]
            human_records.append(chunk_human)
            
            ai_total = sum(len(r) for r in ai_records)
            human_total = sum(len(r) for r in human_records)
            print(f"      Scanned {rows_scanned} rows → {ai_total} AI, {human_total} human")
            
            # Early exit if we have enough of both
            if ai_total >= ai_target and human_total >= human_target:
                break
        
        if sum(len(r) for r in ai_records) >= ai_target and sum(len(r) for r in human_records) >= human_target:
            break

    ai_df = pd.concat(ai_records, ignore_index=True) if ai_records else pd.DataFrame()
    human_df = pd.concat(human_records, ignore_index=True) if human_records else pd.DataFrame()
    
    ai_df["label"]  = 1
    ai_df["source"] = "kaggle"
    ai_df = ai_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    human_df["label"]  = 0
    human_df["source"] = "kaggle"
    human_df = human_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Sample to targets
    ai_sampled = sample_to_target(ai_df, ai_target, "Kaggle AI")
    human_sampled = sample_to_target(human_df, human_target, "Kaggle human")
    
    return ai_sampled, human_sampled


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"\n── Loading modern agent AI samples ──")
    agent_ai = load_ai_samples()
    n_agent = len(agent_ai)

    # Calculate how many Kaggle AI samples we need to reach 1000 total
    kaggle_ai_needed = max(0, 1000 - n_agent)
    
    print(f"\n── Loading Kaggle AI + human samples ──")
    print(f"    Need {kaggle_ai_needed} Kaggle AI samples to reach 1000 total")
    kaggle_ai, kaggle_human = load_kaggle_samples(ai_target=kaggle_ai_needed, human_target=1000)

    # Combine both AI sources
    ai_combined = pd.concat([agent_ai, kaggle_ai], ignore_index=True)
    ai_combined = ai_combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"\n── Merging dataset ──")
    print(f"    AI sources:")
    print(f"      - Modern agents: {n_agent}")
    print(f"      - Kaggle AI:     {len(kaggle_ai)}")
    print(f"      - Combined AI:   {len(ai_combined)}")
    print(f"    Human (Kaggle):    {len(kaggle_human)}")

    # Merge and shuffle
    combined = pd.concat([ai_combined, kaggle_human], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    combined[["text", "label"]].to_csv(OUTPUT_FILE, index=False)

    print(f"\n── Dataset saved to: {OUTPUT_FILE}")
    print(f"   Total rows : {len(combined)}")
    print(f"   AI  (1)    : {(combined['label'] == 1).sum()}")
    print(f"   Human (0)  : {(combined['label'] == 0).sum()}")
    print(f"\n   Label distribution:\n{combined['label'].value_counts()}\n")


if __name__ == "__main__":
    main()