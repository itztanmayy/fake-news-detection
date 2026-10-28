"""
Unified Evaluation and Metric Reporting Module.
Computes:
- Accuracy, Precision (macro), Recall (macro), F1-score (macro)
- Per-class classification report
- Confusion matrix heatmap saved to results/confusion_matrices/
- Appends/updates results in results/metrics_summary.csv
Course: CSET485 - AI and Society, Bennett University
Team ID: 39
"""
import os
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.utils import RESULTS_DIR, CM_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

def evaluate_predictions(y_true, y_pred, model_name: str, dataset_name: str = "ISOT Test", save_plot: bool = True):
    """
    Evaluate binary classification predictions, print report, save confusion matrix,
    and record metrics to results/metrics_summary.csv.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS: {model_name} on {dataset_name}")
    print("=" * 65)
    print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall:    {rec:.4f}")
    print(f"Macro F1-Score:  {f1:.4f}")
    print("-" * 65)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["FAKE (0)", "REAL (1)"], digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 65)
    
    if save_plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fake (0)", "Real (1)"],
            yticklabels=["Fake (0)", "Real (1)"]
        )
        plt.title(f"Confusion Matrix - {model_name} ({dataset_name})", fontsize=12, pad=12)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.tight_layout()
        
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        plot_path = os.path.join(CM_DIR, f"{safe_name}_cm.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix heatmap to {plot_path}")
    
    # Save to metrics_summary.csv
    update_metrics_summary(model_name, dataset_name, acc, prec, rec, f1)
    
    return {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }

def update_metrics_summary(model_name: str, dataset_name: str, acc: float, prec: float, rec: float, f1: float):
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    new_row = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    }
    
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        # Update if model and dataset match, else append
        mask = (df_summary["Model"] == model_name) & (df_summary["Dataset"] == dataset_name)
        if mask.any():
            df_summary.loc[mask, ["Accuracy", "Precision", "Recall", "F1-Score"]] = [
                round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)
            ]
        else:
            df_summary = pd.concat([df_summary, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_summary = pd.DataFrame([new_row])
        
    df_summary.to_csv(summary_path, index=False)
    print(f"Updated metrics summary table at {summary_path}")

def get_metrics_table():
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    if os.path.exists(summary_path):
        return pd.read_csv(summary_path)
    return pd.DataFrame()

if __name__ == "__main__":
    # Self-test dummy
    y_t = [0, 1, 0, 1, 1, 0, 1, 0]
    y_p = [0, 1, 0, 0, 1, 0, 1, 1]
    evaluate_predictions(y_t, y_p, "Self-Test Baseline", "Test Set")
