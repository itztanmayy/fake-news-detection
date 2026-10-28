"""
Text Cleaning and Preprocessing Pipeline.
Course: CSET485 - AI and Society, Bennett University
Team ID: 39
"""
import os
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from src.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, set_seed

# Initialize NLTK resources
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

try:
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize("testing")
except Exception:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    LEMMATIZER = WordNetLemmatizer()

PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def clean_text(text: str) -> str:
    """
    Standard preprocessing for Classical ML and LSTM:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation
    5. Remove digits and numbers
    6. Tokenize
    7. Remove stopwords
    8. Lemmatize
    9. Rejoin tokens
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    
    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    
    # 4. Remove punctuation
    text = text.translate(PUNCT_TABLE)
    
    # 5. Remove digits
    text = re.sub(r"\d+", " ", text)
    
    # 6. Tokenize
    tokens = word_tokenize(text)
    
    # 7 & 8. Stopwords removal and Lemmatization
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]
    
    # 9. Rejoin
    return " ".join(cleaned_tokens)

def clean_text_bert(text: str) -> str:
    """
    Light preprocessing for BERT:
    BERT requires full sentences with stopwords and punctuation intact
    to capture bidirectional contextual representations.
    Only remove URLs, HTML tags, and redundant whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _clean_chunk(texts):
    return [clean_text(t) for t in texts]

def _clean_bert_chunk(texts):
    return [clean_text_bert(t) for t in texts]

def parallel_clean(texts, clean_fn, n_workers=8, chunk_size=2000):
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        if clean_fn == clean_text:
            futures = [executor.submit(_clean_chunk, c) for c in chunks]
        else:
            futures = [executor.submit(_clean_bert_chunk, c) for c in chunks]
        for f in tqdm(futures, desc=f"Parallel {clean_fn.__name__}"):
            results.extend(f.result())
    return results

def build_isot_dataset(force_recompute: bool = False):
    """
    Load raw ISOT True.csv and Fake.csv, clean text, split into
    train (70%), val (15%), test (15%) with stratified sampling, and save CSVs.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    full_path = os.path.join(PROCESSED_DATA_DIR, "isot_full.csv")
    train_path = os.path.join(PROCESSED_DATA_DIR, "isot_train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "isot_val.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "isot_test.csv")
    
    if not force_recompute and all(os.path.exists(p) for p in [full_path, train_path, val_path, test_path]):
        print(f"[CACHE] Processed ISOT datasets already exist in {PROCESSED_DATA_DIR}.")
        return train_path, val_path, test_path, full_path

    set_seed(42)
    print("Loading raw ISOT datasets...")
    true_file = os.path.join(RAW_DATA_DIR, "isot", "True.csv")
    fake_file = os.path.join(RAW_DATA_DIR, "isot", "Fake.csv")
    
    df_true = pd.read_csv(true_file)
    df_fake = pd.read_csv(fake_file)
    
    df_true["label"] = 1  # Real news
    df_fake["label"] = 0  # Fake news
    
    print(f"Loaded Real articles: {len(df_true):,}, Fake articles: {len(df_fake):,}")
    
    df_combined = pd.concat([df_true, df_fake], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Construct combined text
    df_combined["title"] = df_combined["title"].fillna("")
    df_combined["text"] = df_combined["text"].fillna("")
    df_combined["combined_text"] = df_combined["title"] + " " + df_combined["text"]
    
    # Preprocessing
    print("Applying text preprocessing pipeline (clean_text for Classical/LSTM, clean_text_bert for BERT)...")
    texts = df_combined["combined_text"].tolist()
    
    try:
        # Multiprocessing across CPU cores
        df_combined["clean_text"] = parallel_clean(texts, clean_text, n_workers=8)
        df_combined["clean_text_bert"] = parallel_clean(texts, clean_text_bert, n_workers=8)
    except Exception as e:
        print(f"Multiprocessing encountered {e}; falling back to series progress_apply...")
        tqdm.pandas(desc="Cleaning text")
        df_combined["clean_text"] = df_combined["combined_text"].progress_apply(clean_text)
        df_combined["clean_text_bert"] = df_combined["combined_text"].progress_apply(clean_text_bert)
    
    # Remove empty cleaned texts if any
    df_combined = df_combined[df_combined["clean_text"].str.strip() != ""].reset_index(drop=True)
    
    # Save full processed dataset
    df_combined.to_csv(full_path, index=False)
    print(f"Saved full processed dataset to {full_path} (total rows: {len(df_combined):,})")
    
    # Stratified Split: 70% Train, 15% Val, 15% Test
    print("Splitting dataset: 70% Train, 15% Validation, 15% Test (Stratified)...")
    train_df, temp_df = train_test_split(
        df_combined,
        test_size=0.30,
        random_state=42,
        stratify=df_combined["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train split: {len(train_df):,} rows -> {train_path}")
    print(f"Validation split: {len(val_df):,} rows -> {val_path}")
    print(f"Test split: {len(test_df):,} rows -> {test_path}")
    
    return train_path, val_path, test_path, full_path

def load_liar_test():
    """
    Load LIAR test set for cross-dataset generalization.
    Label mapping:
      pants-fire, false, barely-true -> FAKE (0)
      half-true, mostly-true, true -> REAL (1)
    """
    liar_path = os.path.join(RAW_DATA_DIR, "liar", "test.tsv")
    cols = [
        "id", "label_raw", "statement", "subject", "speaker",
        "job_title", "state_info", "party_affiliation",
        "barely_true_counts", "false_counts", "half_true_counts",
        "mostly_true_counts", "pants_on_fire_counts", "context"
    ]
    df = pd.read_csv(liar_path, sep="\t", header=None, names=cols)
    
    binary_map = {
        "pants-fire": 0,
        "false": 0,
        "barely-true": 0,
        "half-true": 1,
        "mostly-true": 1,
        "true": 1
    }
    
    df["label"] = df["label_raw"].map(binary_map)
    df = df.dropna(subset=["label", "statement"]).copy()
    df["label"] = df["label"].astype(int)
    
    df["clean_text"] = df["statement"].apply(clean_text)
    df["clean_text_bert"] = df["statement"].apply(clean_text_bert)
    
    return df

if __name__ == "__main__":
    build_isot_dataset()
