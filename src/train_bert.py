"""
BERT Fine-Tuning for Fake News Classification.
Uses HuggingFace bert-base-uncased with BertForSequenceClassification.
Features:
- AdamW optimizer with linear warmup schedule
- Transfer learning with top transformer layers fine-tuning
- Early stopping & model checkpointing to models/bert_finetuned/
- Hyperparameter tuning across learning rates [5e-5, 2e-5, 1e-5]
- Evaluation on ISOT Test set & LIAR Cross-Dataset Generalization test
Course: CSET485 - AI and Society, Bennett University
Team ID: 39 (Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka)
"""
import os
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import MODELS_DIR, PLOTS_DIR, set_seed, get_device
from src.features import BertNewsDataset
from src.evaluate import evaluate_predictions
from src.preprocess import build_isot_dataset, load_liar_test

def train_bert_epoch(model, dataloader, optimizer, scheduler, device, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training BERT", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        
    return total_loss / total, correct / total

def eval_bert_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating BERT", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().numpy().tolist())
            
    return total_loss / total, correct / total, all_preds

def run_bert_pipeline(
    model_name: str = "bert-base-uncased",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 64,
    subsample_train: int = 600,
    freeze_base: bool = True
):
    set_seed(42)
    device = get_device()
    print(f"Using device for BERT fine-tuning: {device}")
    
    save_dir = os.path.join(MODELS_DIR, "bert_finetuned")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Load data
    train_path, val_path, test_path, _ = build_isot_dataset()
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    if subsample_train is not None and subsample_train < len(train_df):
        print(f"Subsampling {subsample_train:,} training records for efficient CPU fine-tuning...")
        train_df = train_df.sample(n=subsample_train, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(n=min(250, len(val_df)), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=min(500, len(test_df)), random_state=42).reset_index(drop=True)
        
    print(f"Tokenizer: Loading {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    train_dataset = BertNewsDataset(train_df["clean_text_bert"].fillna(""), train_df["label"], tokenizer, max_length=max_length)
    val_dataset = BertNewsDataset(val_df["clean_text_bert"].fillna(""), val_df["label"], tokenizer, max_length=max_length)
    test_dataset = BertNewsDataset(test_df["clean_text_bert"].fillna(""), test_df["label"], tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Instantiate Model
    print(f"Loading Pretrained {model_name} for Binary Sequence Classification...")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    
    if freeze_base:
        print("Transfer Learning Mode: Freezing BERT base embeddings and lower layers (0-7), fine-tuning upper layers...")
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False
                
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # Optimizer and Linear Warmup Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.10)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    print("\n" + "=" * 65)
    print(f"Fine-Tuning BERT (Epochs={epochs}, Batch Size={batch_size}, LR={lr}, Warmup Steps={warmup_steps})")
    print("=" * 65)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_bert_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, _ = eval_bert_epoch(model, val_loader, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"--> Best model improved! Saving weights to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
    # Load best saved model for evaluation
    model = BertForSequenceClassification.from_pretrained(save_dir).to(device)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], marker="o", label="Train Loss", linewidth=2)
    plt.plot(range(1, epochs + 1), history["val_loss"], marker="s", label="Val Loss", linewidth=2)
    plt.title("BERT Fine-Tuning Loss", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], marker="o", label="Train Acc", linewidth=2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], marker="s", label="Val Acc", linewidth=2)
    plt.title("BERT Fine-Tuning Accuracy", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    bert_curve_path = os.path.join(PLOTS_DIR, "bert_training_curves.png")
    plt.savefig(bert_curve_path, dpi=300)
    plt.close()
    print(f"Saved BERT training curves to {bert_curve_path}")
    
    # Evaluate on Test Set
    _, test_acc, test_preds = eval_bert_epoch(model, test_loader, device)
    evaluate_predictions(test_df["label"].tolist(), test_preds, model_name="BERT Fine-Tuned", dataset_name="ISOT Test")
    
    # Evaluate on LIAR Test Set (Cross-Dataset Generalization)
    print("\nEvaluating BERT on LIAR Cross-Dataset Generalization Test...")
    liar_df = load_liar_test()
    liar_dataset = BertNewsDataset(liar_df["clean_text_bert"].fillna(""), liar_df["label"], tokenizer, max_length=max_length)
    liar_loader = DataLoader(liar_dataset, batch_size=batch_size, shuffle=False)
    
    _, _, liar_preds = eval_bert_epoch(model, liar_loader, device)
    evaluate_predictions(liar_df["label"].tolist(), liar_preds, model_name="BERT Fine-Tuned", dataset_name="LIAR Generalization")
    
    return model, tokenizer

def tune_bert_learning_rates():
    """
    Demonstrates hyperparameter sensitivity across learning rates [5e-5, 2e-5, 1e-5]
    as specified in Section 8.
    """
    print("\n" + "=" * 65)
    print("HYPERPARAMETER TUNING: BERT Learning Rates [5e-5, 2e-5, 1e-5]")
    print("=" * 65)
    
    set_seed(42)
    device = get_device()
    train_path, val_path, _, _ = build_isot_dataset()
    train_df = pd.read_csv(train_path).sample(n=200, random_state=42).reset_index(drop=True)
    val_df = pd.read_csv(val_path).sample(n=100, random_state=42).reset_index(drop=True)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = BertNewsDataset(train_df["clean_text_bert"].fillna(""), train_df["label"], tokenizer, max_length=64)
    val_dataset = BertNewsDataset(val_df["clean_text_bert"].fillna(""), val_df["label"], tokenizer, max_length=64)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    lrs = [5e-5, 2e-5, 1e-5]
    lr_accs = []
    
    for lr_val in lrs:
        m = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
        for param in m.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in m.bert.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False
        opt = AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr_val, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=3, num_training_steps=len(train_loader))
        train_bert_epoch(m, train_loader, opt, sched, device)
        _, val_acc, _ = eval_bert_epoch(m, val_loader, device)
        lr_accs.append(val_acc)
        print(f"BERT LR: {lr_val:<8} -> Validation Accuracy: {val_acc*100:.2f}%")
        
    plt.figure(figsize=(7, 5))
    plt.plot([str(l) for l in lrs], [a * 100 for a in lr_accs], marker="o", color="purple", linewidth=2)
    plt.title("BERT Hyperparameter Sensitivity: Learning Rate", fontsize=12)
    plt.xlabel("Learning Rate", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "bert_hyperparameter_tuning.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved BERT hyperparameter tuning plot to {plot_path}")

if __name__ == "__main__":
    run_bert_pipeline(subsample_train=600, epochs=2, batch_size=32, max_length=64)
    tune_bert_learning_rates()
