"""
Bidirectional LSTM Model for Fake News Classification.
Architecture:
1. Embedding Layer: 20,000 vocab x 128 dimensions
2. Bidirectional LSTM: 128 units each direction (256 total), Dropout = 0.3
3. Dense Layer: 64 units, ReLU activation
4. Dropout: 0.3
5. Output Layer: 1 unit, Sigmoid activation
Course: CSET485 - AI and Society, Bennett University
Team ID: 39
"""
import os
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import MODELS_DIR, PLOTS_DIR, set_seed, get_device
from src.features import LSTMVocab, LSTMDataset
from src.evaluate import evaluate_predictions
from src.preprocess import build_isot_dataset, load_liar_test

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int = 20000, embed_dim: int = 128, hidden_dim: int = 128, dense_dim: int = 64, dropout_p: float = 0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p if 1 > 1 else 0.0  # single layer
        )
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(hidden_dim * 2, dense_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc_out = nn.Linear(dense_dim, 1)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        lstm_out, (hn, cn) = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        
        # Max-pooling over time (or taking concatenation of final forward/backward hidden states)
        # We concatenate last forward hn[-2] and backward hn[-1]
        last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # [batch_size, hidden_dim * 2]
        
        x = self.dropout1(last_hidden)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc_out(x).squeeze(1)  # [batch_size]
        return logits

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
        
    return total_loss / total, correct / total

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().numpy().tolist())
            
    return total_loss / total, correct / total, all_preds

def run_lstm_pipeline(epochs: int = 4, batch_size: int = 64, lr: float = 0.001, subsample_train: int = 12000):
    set_seed(42)
    device = get_device()
    print(f"Using device for LSTM: {device}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Load preprocessed splits
    train_path, val_path, test_path, _ = build_isot_dataset()
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    if subsample_train is not None and subsample_train < len(train_df):
        print(f"Using representative sample of {subsample_train:,} training records for optimized CPU training...")
        train_df = train_df.sample(n=subsample_train, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(n=min(3000, len(val_df)), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=min(3000, len(test_df)), random_state=42).reset_index(drop=True)
    
    train_texts = train_df["clean_text"].fillna("").tolist()
    train_labels = train_df["label"].tolist()
    
    val_texts = val_df["clean_text"].fillna("").tolist()
    val_labels = val_df["label"].tolist()
    
    test_texts = test_df["clean_text"].fillna("").tolist()
    test_labels = test_df["label"].tolist()
    
    # 2. Build Vocabulary
    vocab_file = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
    vocab = LSTMVocab(max_vocab_size=20000, max_len=300)
    vocab.build_vocab(train_texts)
    vocab.save(vocab_file)
    
    # 3. Create Datasets & DataLoaders
    train_dataset = LSTMDataset(train_texts, train_labels, vocab)
    val_dataset = LSTMDataset(val_texts, val_labels, vocab)
    test_dataset = LSTMDataset(test_texts, test_labels, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Instantiate Model
    model = BiLSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embed_dim=128,
        hidden_dim=128,
        dense_dim=64,
        dropout_p=0.3
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 5. Training with Early Stopping (patience=3)
    patience = 3
    patience_counter = 0
    best_val_loss = float("inf")
    best_model_weights = None
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    print("\n" + "=" * 65)
    print(f"Training Bidirectional LSTM (Epochs={epochs}, Batch Size={batch_size}, LR={lr})")
    print("=" * 65)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = eval_epoch(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, os.path.join(MODELS_DIR, "lstm_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no val loss improvement for {patience} epochs).")
                break
                
    # Load best weights
    model.load_state_dict(best_model_weights)
    print(f"Loaded best LSTM checkpoint (Val Loss: {best_val_loss:.4f})")
    
    # 6. Plot Training Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], marker="o", label="Train Loss", linewidth=2)
    axes[0].plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], marker="s", label="Val Loss", linewidth=2)
    axes[0].set_title("BiLSTM Loss Curves", fontsize=12)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Binary Cross Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()
    
    axes[1].plot(range(1, len(history["train_acc"]) + 1), [a * 100 for a in history["train_acc"]], marker="o", label="Train Accuracy", linewidth=2)
    axes[1].plot(range(1, len(history["val_acc"]) + 1), [a * 100 for a in history["val_acc"]], marker="s", label="Val Accuracy", linewidth=2)
    axes[1].set_title("BiLSTM Accuracy Curves", fontsize=12)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    curve_path = os.path.join(PLOTS_DIR, "lstm_training_curves.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved LSTM training curves to {curve_path}")
    
    # 7. Evaluate on Test Set
    _, test_acc, test_preds = eval_epoch(model, test_loader, criterion, device)
    evaluate_predictions(test_labels, test_preds, model_name="Bidirectional LSTM", dataset_name="ISOT Test")
    
    # 8. Cross-Dataset Generalization on LIAR
    print("\nEvaluating BiLSTM on LIAR Cross-Dataset Generalization Test...")
    liar_df = load_liar_test()
    liar_texts = liar_df["clean_text"].fillna("").tolist()
    liar_labels = liar_df["label"].tolist()
    liar_dataset = LSTMDataset(liar_texts, liar_labels, vocab)
    liar_loader = DataLoader(liar_dataset, batch_size=batch_size, shuffle=False)
    
    _, _, liar_preds = eval_epoch(model, liar_loader, criterion, device)
    evaluate_predictions(liar_labels, liar_preds, model_name="Bidirectional LSTM", dataset_name="LIAR Generalization")
    
    return model, history

def tune_lstm_hyperparameters():
    """
    Demonstrates tuning on learning rate [0.01, 0.001, 0.0001]
    and batch size [32, 64, 128] as specified in Section 8.
    """
    print("\n" + "=" * 65)
    print("HYPERPARAMETER TUNING: BiLSTM Learning Rate and Batch Size")
    print("=" * 65)
    
    set_seed(42)
    device = get_device()
    train_path, val_path, _, _ = build_isot_dataset()
    train_df = pd.read_csv(train_path).sample(n=6000, random_state=42)  # representative subset for tuning speed
    val_df = pd.read_csv(val_path).sample(n=1500, random_state=42)
    
    vocab = LSTMVocab(max_vocab_size=10000, max_len=200)
    vocab.build_vocab(train_df["clean_text"].fillna("").tolist())
    
    train_ds = LSTMDataset(train_df["clean_text"].fillna("").tolist(), train_df["label"].tolist(), vocab)
    val_ds = LSTMDataset(val_df["clean_text"].fillna("").tolist(), val_df["label"].tolist(), vocab)
    criterion = nn.BCEWithLogitsLoss()
    
    # 1. Learning Rate sensitivity (fixed batch_size=64)
    lrs = [0.01, 0.001, 0.0001]
    lr_accs = []
    for lr_val in lrs:
        train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=64, shuffle=False)
        m = BiLSTMClassifier(vocab_size=len(vocab.word2idx), embed_dim=64, hidden_dim=64, dense_dim=32).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=lr_val)
        for _ in range(2):
            train_epoch(m, train_ld, opt, criterion, device)
        _, val_acc, _ = eval_epoch(m, val_ld, criterion, device)
        lr_accs.append(val_acc)
        print(f"LR: {lr_val:<8} -> Validation Accuracy: {val_acc*100:.2f}%")
        
    # 2. Batch Size sensitivity (fixed lr=0.001)
    batch_sizes = [32, 64, 128]
    bs_accs = []
    for bs_val in batch_sizes:
        train_ld = DataLoader(train_ds, batch_size=bs_val, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=bs_val, shuffle=False)
        m = BiLSTMClassifier(vocab_size=len(vocab.word2idx), embed_dim=64, hidden_dim=64, dense_dim=32).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=0.001)
        for _ in range(2):
            train_epoch(m, train_ld, opt, criterion, device)
        _, val_acc, _ = eval_epoch(m, val_ld, criterion, device)
        bs_accs.append(val_acc)
        print(f"Batch Size: {bs_val:<4} -> Validation Accuracy: {val_acc*100:.2f}%")
        
    # Plot tuning results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([str(l) for l in lrs], [a * 100 for a in lr_accs], marker="o", color="navy", linewidth=2)
    axes[0].set_title("BiLSTM Sensitivity: Learning Rate", fontsize=12)
    axes[0].set_xlabel("Learning Rate", fontsize=11)
    axes[0].set_ylabel("Validation Accuracy (%)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1].plot([str(b) for b in batch_sizes], [a * 100 for a in bs_accs], marker="s", color="darkgreen", linewidth=2)
    axes[1].set_title("BiLSTM Sensitivity: Batch Size", fontsize=12)
    axes[1].set_xlabel("Batch Size", fontsize=11)
    axes[1].set_ylabel("Validation Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    tuning_plot = os.path.join(PLOTS_DIR, "lstm_hyperparameter_tuning.png")
    plt.savefig(tuning_plot, dpi=300)
    plt.close()
    print(f"Saved LSTM hyperparameter tuning plot to {tuning_plot}")

if __name__ == "__main__":
    run_lstm_pipeline(epochs=4, batch_size=64, lr=0.001, subsample_train=12000)
    tune_lstm_hyperparameters()
