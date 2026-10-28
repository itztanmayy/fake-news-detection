"""
Feature Extraction and Dataset Utilities.
Includes:
1. TF-IDF vectorizer wrapper (unigrams + bigrams, sublinear_tf=True, max_features=50000)
2. Vocabulary builder & PyTorch Dataset for LSTM
3. HuggingFace BERT Tokenizer Dataset wrapper
Course: CSET485 - AI and Society, Bennett University
Team ID: 39
"""
import os
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

from src.utils import MODELS_DIR, save_pickle, load_pickle

# ==========================================
# 1. TF-IDF Extractor
# ==========================================
class TfidfFeatureExtractor:
    def __init__(self, max_features: int = 50000, ngram_range: Tuple[int, int] = (1, 2), sublinear_tf: bool = True):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf
        )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        print(f"Fitting TF-IDF Vectorizer (max_features={self.vectorizer.max_features}, ngram_range={self.vectorizer.ngram_range})...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]):
        if not self.is_fitted:
            raise ValueError("TF-IDF Vectorizer must be fitted before transforming.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        self.fit(texts)
        return self.transform(texts)

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        save_pickle(self.vectorizer, filepath)
        print(f"Saved TF-IDF Vectorizer to {filepath}")

    @classmethod
    def load(cls, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        extractor = cls()
        extractor.vectorizer = load_pickle(filepath)
        extractor.is_fitted = True
        return extractor

# ==========================================
# 2. LSTM Vocabulary and Dataset
# ==========================================
class LSTMVocab:
    PAD_TOKEN = "<pad>"  # index 0
    UNK_TOKEN = "<unk>"  # index 1

    def __init__(self, max_vocab_size: int = 20000, max_len: int = 300):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build_vocab(self, texts: List[str]):
        print(f"Building LSTM Vocabulary from {len(texts):,} texts (max size: {self.max_vocab_size})...")
        counter = Counter()
        for text in texts:
            if isinstance(text, str):
                tokens = text.split()
                counter.update(tokens)
        
        # Most frequent tokens excluding reserved indices 0, 1
        most_common = counter.most_common(self.max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"Vocabulary built: {len(self.word2idx)} unique tokens.")
        return self

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            tokens = []
        else:
            tokens = text.split()
        
        indices = [self.word2idx.get(w, self.word2idx[self.UNK_TOKEN]) for w in tokens]
        
        # Pad or truncate to max_len
        if len(indices) < self.max_len:
            indices = indices + [self.word2idx[self.PAD_TOKEN]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return indices

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
        save_pickle({"word2idx": self.word2idx, "idx2word": self.idx2word, "max_len": self.max_len}, filepath)
        print(f"Saved LSTM Vocabulary to {filepath}")

    @classmethod
    def load(cls, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
        data = load_pickle(filepath)
        vocab = cls(max_len=data["max_len"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        return vocab

class LSTMDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: LSTMVocab):
        self.labels = np.array(labels, dtype=np.float32)
        print(f"Encoding {len(texts):,} sequences for LSTM Dataset...")
        self.encoded_texts = [vocab.encode(t) for t in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# ==========================================
# 3. BERT Dataset
# ==========================================
class BertNewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }
