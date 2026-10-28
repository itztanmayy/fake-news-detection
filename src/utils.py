"""
Utility Functions for Fake News Detection System.
Course: CSET485 - AI and Society, Bennett University
Team ID: 39
"""
import os
import random
import pickle
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set random seed across all libraries for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Return PyTorch device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_pickle(obj, filepath: str):
    """Save an object via pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str):
    """Load an object via pickle."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Common directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")

