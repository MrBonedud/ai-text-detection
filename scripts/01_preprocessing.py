import pandas as pd
import re
import nltk
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab/english')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Initialize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def clean_text(text, use_stemmer=False):
    """
    Clean and preprocess text:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove punctuation and numbers
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize
    """
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation and numbers, keep spaces and basic letters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords, lemmatize (and optionally stem)
    if use_stemmer:
        tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens if token not in stop_words and len(token) > 1]
    else:
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]
    
    # Rejoin
    cleaned = ' '.join(tokens)
    
    return cleaned

def main():
    parser = argparse.ArgumentParser(description='Preprocess raw text CSV')
    parser.add_argument('--input', '-i', default=os.path.join('data', 'processed', 'raw_data.csv'), help='Input CSV path')
    parser.add_argument('--output', '-o', default=os.path.join('data', 'processed', 'clean_data.csv'), help='Output CSV path')
    parser.add_argument('--use-stemmer', action='store_true', help='Apply Porter stemming after lemmatization')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    print("── Loading dataset ──")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} samples")
    
    print("\n── Preprocessing text ──")
    
    # Store originals for before/after comparison
    before_samples = df['text'].head(3).copy()
    
    # Ensure column name
    if 'text' not in df.columns:
        # try common alternatives
        for c in ['content', 'body']:
            if c in df.columns:
                df = df.rename(columns={c: 'text'})
                break

    # Apply cleaning (skip missing values)
    df['text'] = df['text'].fillna('').astype(str).apply(lambda t: clean_text(t, use_stemmer=args.use_stemmer))
    
    after_samples = df['text'].head(3).copy()
    
    print(f"   Preprocessed all {len(df)} samples (use_stemmer={args.use_stemmer})")
    
    print("\n── Before/After Samples ──\n")
    for i in range(3):
        print(f"Sample {i+1}:")
        print(f"  BEFORE ({len(before_samples.iloc[i])} chars):")
        print(f"    {before_samples.iloc[i][:150]}...")
        print(f"\n  AFTER ({len(after_samples.iloc[i])} chars):")
        print(f"    {after_samples.iloc[i][:150]}...")
        print()
    
    print("── Saving cleaned dataset ──")
    df.to_csv(output_path, index=False)
    print(f"   Saved to {output_path}")
    
    print("\n── Dataset Statistics ──")
    print(f"   Total rows: {len(df)}")
    print(f"   AI (label=1): {len(df[df['label']==1])}")
    print(f"   Human (label=0): {len(df[df['label']==0])}")
    print(f"   Avg text length: {df['text'].str.len().mean():.0f} chars")
    print(f"   Avg tokens per sample: {df['text'].str.split().str.len().mean():.1f}")

if __name__ == '__main__':
    main()