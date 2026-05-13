import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
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
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]
    
    # Rejoin
    cleaned = ' '.join(tokens)
    
    return cleaned

def main():
    input_path = 'data/processed/raw_data.csv'
    output_path = 'data/processed/clean_data.csv'
    
    print("── Loading dataset ──")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} samples")
    
    print("\n── Preprocessing text ──")
    
    # Store originals for before/after comparison
    before_samples = df['text'].head(3).copy()
    
    # Apply cleaning
    df['text'] = df['text'].apply(clean_text)
    
    after_samples = df['text'].head(3).copy()
    
    print(f"   Preprocessed all {len(df)} samples")
    
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
