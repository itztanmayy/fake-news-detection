"""
Train Classical ML Models:
1. Logistic Regression (Baseline)
2. Support Vector Machine (LinearSVC)
Demonstrates hyperparameter tuning on C: [0.01, 0.1, 1.0, 10.0]
Evaluates on ISOT Test set and LIAR Cross-Dataset generalization test.
Course: CSET485 - AI and Society, Bennett University
Team ID: 39
"""
import os
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

from src.utils import MODELS_DIR, PLOTS_DIR, set_seed, save_pickle
from src.features import TfidfFeatureExtractor
from src.evaluate import evaluate_predictions
from src.preprocess import build_isot_dataset, load_liar_test

def run_classical_pipeline():
    set_seed(42)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load data
    train_path, val_path, test_path, _ = build_isot_dataset()
    print("Loading preprocessed splits for classical models...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Handle any null values
    train_df["clean_text"] = train_df["clean_text"].fillna("")
    val_df["clean_text"] = val_df["clean_text"].fillna("")
    test_df["clean_text"] = test_df["clean_text"].fillna("")

    # 2. TF-IDF Feature Extraction
    tfidf = TfidfFeatureExtractor(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_df["clean_text"])
    y_train = train_df["label"].values

    X_val = tfidf.transform(val_df["clean_text"])
    y_val = val_df["label"].values

    X_test = tfidf.transform(test_df["clean_text"])
    y_test = test_df["label"].values

    # Save TF-IDF Vectorizer
    tfidf.save(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    # ==========================================
    # HYPERPARAMETER TUNING: C in [0.01, 0.1, 1.0, 10.0]
    # ==========================================
    c_values = [0.01, 0.1, 1.0, 10.0]
    lr_val_accs = []
    svm_val_accs = []

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING: Testing C in [0.01, 0.1, 1.0, 10.0] on Validation Set")
    print("=" * 60)

    for c in c_values:
        # Logistic Regression
        lr_tune = LogisticRegression(C=c, max_iter=1000, solver="lbfgs", random_state=42)
        lr_tune.fit(X_train, y_train)
        lr_acc = lr_tune.score(X_val, y_val)
        lr_val_accs.append(lr_acc)

        # Linear SVC
        svm_tune = LinearSVC(C=c, max_iter=2000, random_state=42, dual=False)
        svm_tune.fit(X_train, y_train)
        svm_acc = svm_tune.score(X_val, y_val)
        svm_val_accs.append(svm_acc)

        print(f"C = {c:<6} | LR Val Accuracy: {lr_acc:.4f} | SVM Val Accuracy: {svm_acc:.4f}")

    # Plot Hyperparameter Sensitivity
    plt.figure(figsize=(8, 5))
    plt.plot(c_values, [a * 100 for a in lr_val_accs], marker="o", label="Logistic Regression", linewidth=2)
    plt.plot(c_values, [a * 100 for a in svm_val_accs], marker="s", label="Support Vector Machine (LinearSVC)", linewidth=2)
    plt.xscale("log")
    plt.title("Hyperparameter Sensitivity: Inverse Regularization Strength (C)", fontsize=12, pad=12)
    plt.xlabel("C Value (log scale)", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    tuning_plot_path = os.path.join(PLOTS_DIR, "classical_hyperparameter_tuning.png")
    plt.savefig(tuning_plot_path, dpi=300)
    plt.close()
    print(f"Saved hyperparameter tuning plot to {tuning_plot_path}")

    # ==========================================
    # MODEL 1: Final Logistic Regression (C=1.0)
    # ==========================================
    print("\nTraining final Logistic Regression model (C=1.0, max_iter=1000)...")
    lr_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    lr_model.fit(X_train, y_train)
    save_pickle(lr_model, os.path.join(MODELS_DIR, "lr_model.pkl"))

    lr_test_preds = lr_model.predict(X_test)
    evaluate_predictions(y_test, lr_test_preds, model_name="Logistic Regression", dataset_name="ISOT Test")

    # ==========================================
    # MODEL 2: Final Support Vector Machine (C=1.0)
    # ==========================================
    print("\nTraining final Support Vector Machine model (LinearSVC, C=1.0, max_iter=2000)...")
    svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42, dual=False)
    svm_model.fit(X_train, y_train)
    save_pickle(svm_model, os.path.join(MODELS_DIR, "svm_model.pkl"))

    svm_test_preds = svm_model.predict(X_test)
    evaluate_predictions(y_test, svm_test_preds, model_name="Support Vector Machine", dataset_name="ISOT Test")

    # ==========================================
    # CROSS-DATASET EVALUATION: LIAR Test Set
    # ==========================================
    print("\nEvaluating Classical Models on LIAR Cross-Dataset Generalization Test...")
    liar_df = load_liar_test()
    liar_df["clean_text"] = liar_df["clean_text"].fillna("")
    X_liar = tfidf.transform(liar_df["clean_text"])
    y_liar = liar_df["label"].values

    lr_liar_preds = lr_model.predict(X_liar)
    evaluate_predictions(y_liar, lr_liar_preds, model_name="Logistic Regression", dataset_name="LIAR Generalization")

    svm_liar_preds = svm_model.predict(X_liar)
    evaluate_predictions(y_liar, svm_liar_preds, model_name="Support Vector Machine", dataset_name="LIAR Generalization")

    print("\nClassical models training and evaluation completed successfully!")

if __name__ == "__main__":
    run_classical_pipeline()
