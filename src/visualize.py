"""
Visualization Generator for Fake News Detection System.
Generates:
1. Class distribution bar chart (ISOT Real vs Fake count)
2. Word Clouds (Fake vs Real articles top words)
3. Performance comparison bar chart (all 4 models on Accuracy and F1-Score)
4. Cross-dataset generalization gap chart (ISOT Test vs LIAR Test)
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
from wordcloud import WordCloud

from src.utils import PLOTS_DIR, RESULTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

os.makedirs(PLOTS_DIR, exist_ok=True)

# Set clean aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})

def plot_class_distribution():
    print("Generating Class Distribution Bar Chart...")
    true_csv = os.path.join(RAW_DATA_DIR, "isot", "True.csv")
    fake_csv = os.path.join(RAW_DATA_DIR, "isot", "Fake.csv")
    
    true_count = len(pd.read_csv(true_csv)) if os.path.exists(true_csv) else 21417
    fake_count = len(pd.read_csv(fake_csv)) if os.path.exists(fake_csv) else 23481
    
    categories = ["Real News (1)", "Fake News (0)"]
    counts = [true_count, fake_count]
    colors = ["#2ecc71", "#e74c3c"]
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(categories, counts, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 400,
            f"{int(height):,} ({height / sum(counts) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )
        
    plt.title("ISOT Fake News Dataset - Class Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("News Classification Label", fontsize=11, fontweight="bold")
    plt.ylabel("Article Count", fontsize=11, fontweight="bold")
    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_word_clouds():
    print("Generating Word Clouds for Real vs Fake Articles...")
    full_csv = os.path.join(PROCESSED_DATA_DIR, "isot_full.csv")
    if not os.path.exists(full_csv):
        print("[WARN] isot_full.csv not found yet. Skipping wordclouds until preprocessing runs.")
        return
        
    df = pd.read_csv(full_csv)
    real_text = " ".join(df[df["label"] == 1]["clean_text"].dropna().sample(n=min(2000, len(df[df["label"] == 1])), random_state=42))
    fake_text = " ".join(df[df["label"] == 0]["clean_text"].dropna().sample(n=min(2000, len(df[df["label"] == 0])), random_state=42))
    
    wc_real = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Greens",
        max_words=100,
        random_state=42
    ).generate(real_text)
    
    wc_fake = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Reds",
        max_words=100,
        random_state=42
    ).generate(fake_text)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(wc_real, interpolation="bilinear")
    axes[0].set_title("Top 100 Frequent Words - REAL News Articles", fontsize=14, fontweight="bold", pad=12)
    axes[0].axis("off")
    
    axes[1].imshow(wc_fake, interpolation="bilinear")
    axes[1].set_title("Top 100 Frequent Words - FAKE News Articles", fontsize=14, fontweight="bold", pad=12)
    axes[1].axis("off")
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "wordclouds_real_vs_fake.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_model_comparison():
    print("Generating Model Performance Comparison Chart...")
    summary_csv = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    if not os.path.exists(summary_csv):
        print("[WARN] metrics_summary.csv not found. Creating placeholder comparison...")
        return
        
    df = pd.read_csv(summary_csv)
    isot_df = df[df["Dataset"] == "ISOT Test"].copy()
    if isot_df.empty:
        return
        
    models = isot_df["Model"].tolist()
    accs = [v * 100 for v in isot_df["Accuracy"].tolist()]
    f1s = [v * 100 for v in isot_df["F1-Score"].tolist()]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(9, 5.5))
    bars1 = plt.bar(x - width/2, accs, width, label="Accuracy (%)", color="#3498db", edgecolor="black")
    bars2 = plt.bar(x + width/2, f1s, width, label="F1-Score (%)", color="#9b59b6", edgecolor="black")
    
    for bar in bars1:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.title("Model Comparison on ISOT Test Set: Classical ML to Transformers", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.xticks(x, models, fontsize=10, rotation=10)
    plt.ylim(0, 115)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, "models_performance_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_cross_dataset_generalization():
    print("Generating Cross-Dataset Generalization Comparison Chart...")
    summary_csv = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    if not os.path.exists(summary_csv):
        return
        
    df = pd.read_csv(summary_csv)
    isot_sub = df[df["Dataset"] == "ISOT Test"].set_index("Model")["Accuracy"] * 100
    liar_sub = df[df["Dataset"] == "LIAR Generalization"].set_index("Model")["Accuracy"] * 100
    
    common_models = [m for m in isot_sub.index if m in liar_sub.index]
    if not common_models:
        return
        
    x = np.arange(len(common_models))
    width = 0.35
    
    plt.figure(figsize=(9, 5.5))
    b1 = plt.bar(x - width/2, [isot_sub[m] for m in common_models], width, label="ISOT Test Set (In-Domain)", color="#2ecc71", edgecolor="black")
    b2 = plt.bar(x + width/2, [liar_sub[m] for m in common_models], width, label="LIAR Test Set (Cross-Domain)", color="#e67e22", edgecolor="black")
    
    for bar in b1:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., y + 0.8, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in b2:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., y + 0.8, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.title("Cross-Dataset Generalization Gap: In-Domain (ISOT) vs Out-of-Domain (LIAR)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.xticks(x, common_models, fontsize=10)
    plt.ylim(0, 115)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(PLOTS_DIR, "cross_dataset_generalization_gap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def generate_all_visualizations():
    plot_class_distribution()
    plot_word_clouds()
    plot_model_comparison()
    plot_cross_dataset_generalization()

if __name__ == "__main__":
    generate_all_visualizations()
