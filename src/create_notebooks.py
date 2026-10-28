"""
Generator for all 5 Jupyter Notebooks.
Ensures reproducible, clearly annotated, modular notebooks:
1. 01_eda.ipynb
2. 02_preprocessing.ipynb
3. 03_classical_models.ipynb
4. 04_lstm_model.ipynb
5. 05_bert_model.ipynb
Course: CSET485 - AI and Society, Bennett University
Team ID: 39
"""
import os
import nbformat as nbf

NOTEBOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 01 - Exploratory Data Analysis (EDA)
### Project: Fake News Detection Using Natural Language Processing
**Course:** CSET485 – AI and Society, Bennett University  
**Team ID:** 39 | **Team Members:** Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka

---

### Objectives
1. Inspect the raw **ISOT Fake News Dataset** (`True.csv` and `Fake.csv`).
2. Analyze class distribution (Real vs Fake).
3. Compute and visualize text length and word count distributions.
4. Analyze category / subject distributions.
5. Generate word clouds for Real and Fake news.
"""),
        nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})

print("Libraries imported successfully.")
"""),
        nbf.v4.new_markdown_cell("""### 1. Load Raw ISOT Data"""),
        nbf.v4.new_code_cell("""DATA_DIR = os.path.join("..", "data", "raw", "isot")
df_true = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
df_fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))

print(f"Real articles shape: {df_true.shape}")
print(f"Fake articles shape: {df_fake.shape}")
display(df_true.head(2))
display(df_fake.head(2))
"""),
        nbf.v4.new_markdown_cell("""### 2. Class Distribution Analysis"""),
        nbf.v4.new_code_cell("""counts = [len(df_true), len(df_fake)]
labels = ["Real News (1)", "Fake News (0)"]
colors = ["#2ecc71", "#e74c3c"]

plt.figure(figsize=(7, 4.5))
bars = plt.bar(labels, counts, color=colors, width=0.5, edgecolor="black")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, y + 400, f"{int(y):,} ({y/sum(counts)*100:.1f}%)", ha="center", fontweight="bold")

plt.title("ISOT Dataset Class Balance", fontsize=12, fontweight="bold")
plt.ylabel("Article Count", fontsize=11)
plt.ylim(0, max(counts) * 1.15)
plt.show()
"""),
        nbf.v4.new_markdown_cell("""### 3. Subject Distribution Analysis"""),
        nbf.v4.new_code_cell("""plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df_true["subject"].value_counts().plot(kind="bar", color="#2ecc71", edgecolor="black")
plt.title("Real News Subjects")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
df_fake["subject"].value_counts().plot(kind="bar", color="#e74c3c", edgecolor="black")
plt.title("Fake News Subjects")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_markdown_cell("""### 4. Text Length and Word Count Analysis"""),
        nbf.v4.new_code_cell("""df_true["word_count"] = df_true["text"].apply(lambda t: len(str(t).split()))
df_fake["word_count"] = df_fake["text"].apply(lambda t: len(str(t).split()))

print("Real News Word Count Stats:")
print(df_true["word_count"].describe())

print("\\nFake News Word Count Stats:")
print(df_fake["word_count"].describe())

plt.figure(figsize=(9, 4.5))
sns.kdeplot(df_true["word_count"], label="Real News", color="#2ecc71", fill=True, clip=(0, 2000))
sns.kdeplot(df_fake["word_count"], label="Fake News", color="#e74c3c", fill=True, clip=(0, 2000))
plt.title("Distribution of Article Word Count (Capped at 2,000 words)", fontsize=12, fontweight="bold")
plt.xlabel("Word Count")
plt.ylabel("Density")
plt.legend()
plt.show()
""")
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "01_eda.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Generated notebooks/01_eda.ipynb")

def create_preprocessing_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 02 - Preprocessing Pipeline
### Project: Fake News Detection Using Natural Language Processing
**Course:** CSET485 – AI and Society, Bennett University  
**Team ID:** 39 | **Team Members:** Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka

---

### Objectives
1. Implement the standardized text cleaning pipeline (lowercasing, URLs, HTML tags, punctuation, digits, tokenization, stopwords, lemmatization).
2. Compare standard preprocessing vs BERT-specific contextual preprocessing.
3. Perform stratified dataset splitting (70% Train, 15% Val, 15% Test).
4. Save processed artifacts for downstream modeling.
"""),
        nbf.v4.new_code_cell("""import os, sys
sys.path.append("..")

import pandas as pd
from src.preprocess import clean_text, clean_text_bert, build_isot_dataset
print("Preprocessing modules imported.")
"""),
        nbf.v4.new_markdown_cell("""### 1. Preprocessing Function Verification on Sample Text"""),
        nbf.v4.new_code_cell("""sample_article = \"\"\"
<p>BREAKING: Go to https://fakenews.com/exclusive! Trump visited the White House at 12:30 PM & signed 5 bills; shocking details emerged!!</p>
\"\"\"

print("Original text:")
print(sample_article)

print("\\nStandard clean_text (for TF-IDF / LSTM):")
print(clean_text(sample_article))

print("\\nBERT clean_text_bert (for Transformer):")
print(clean_text_bert(sample_article))
"""),
        nbf.v4.new_markdown_cell("""### 2. Execute Full Dataset Pipeline and Stratified Splitting"""),
        nbf.v4.new_code_cell("""train_path, val_path, test_path, full_path = build_isot_dataset()

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

print(f"Total processed dataset: {len(pd.read_csv(full_path)):,} rows")
print(f"Training split:          {len(train_df):,} rows (70%)")
print(f"Validation split:        {len(val_df):,} rows (15%)")
print(f"Testing split:           {len(test_df):,} rows (15%)")
display(train_df[["title", "clean_text", "label"]].head(3))
""")
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "02_preprocessing.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Generated notebooks/02_preprocessing.ipynb")

def create_classical_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(r"""# 03 - Classical Machine Learning Models
### Project: Fake News Detection Using Natural Language Processing
**Course:** CSET485 – AI and Society, Bennett University  
**Team ID:** 39 | **Team Members:** Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka

---

### Objectives
1. Extract TF-IDF features (unigram + bigram, `sublinear_tf=True`, `max_features=50000`).
2. Train **Logistic Regression** (Baseline model).
3. Train **Support Vector Machine** (`LinearSVC`).
4. Perform hyperparameter tuning on inverse regularization parameter $C \in [0.01, 0.1, 1.0, 10.0]$.
5. Evaluate on ISOT Test set and LIAR Cross-Dataset Generalization set.
"""),
        nbf.v4.new_code_cell("""import os, sys
sys.path.append("..")

import pandas as pd
from src.features import TfidfFeatureExtractor
from src.train_classical import run_classical_pipeline
from src.evaluate import get_metrics_table

print("Modules imported.")
"""),
        nbf.v4.new_markdown_cell("""### Run Classical Training & Evaluation Pipeline"""),
        nbf.v4.new_code_cell("""run_classical_pipeline()
"""),
        nbf.v4.new_markdown_cell("""### View Current Metrics Summary"""),
        nbf.v4.new_code_cell("""summary = get_metrics_table()
display(summary)
""")
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "03_classical_models.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Generated notebooks/03_classical_models.ipynb")

def create_lstm_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 04 - Bidirectional LSTM Deep Learning Model
### Project: Fake News Detection Using Natural Language Processing
**Course:** CSET485 – AI and Society, Bennett University  
**Team ID:** 39 | **Team Members:** Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka

---

### Architecture
- **Embedding Layer:** 20,000 vocabulary size $\\times$ 128 embedding dimensions
- **Bidirectional LSTM:** 128 units per direction (256 combined units), Dropout = 0.3
- **Dense Layer:** 64 units, ReLU activation, Dropout = 0.3
- **Output Layer:** 1 unit, Sigmoid activation (Binary Cross-Entropy Loss)
"""),
        nbf.v4.new_code_cell("""import os, sys
sys.path.append("..")

from src.train_lstm import run_lstm_pipeline, tune_lstm_hyperparameters
print("LSTM modules imported.")
"""),
        nbf.v4.new_markdown_cell("""### 1. Train Bidirectional LSTM Model"""),
        nbf.v4.new_code_cell("""model, history = run_lstm_pipeline(epochs=5, batch_size=64, lr=0.001)
"""),
        nbf.v4.new_markdown_cell("""### 2. Hyperparameter Sensitivity Experiments"""),
        nbf.v4.new_code_cell("""tune_lstm_hyperparameters()
""")
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "04_lstm_model.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Generated notebooks/04_lstm_model.ipynb")

def create_bert_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 05 - BERT Transformer Fine-Tuning & Model Comparison
### Project: Fake News Detection Using Natural Language Processing
**Course:** CSET485 – AI and Society, Bennett University  
**Team ID:** 39 | **Team Members:** Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka

---

### Objectives
1. Fine-tune HuggingFace `bert-base-uncased` with `BertForSequenceClassification`.
2. Optimize using AdamW with linear warmup learning rate schedule.
3. Compare BERT against Classical and LSTM baselines.
4. Evaluate cross-dataset generalization robustness on the LIAR benchmark.
5. Generate end-to-end comparative visualizations.
"""),
        nbf.v4.new_code_cell("""import os, sys
sys.path.append("..")

from src.train_bert import run_bert_pipeline, tune_bert_learning_rates
from src.visualize import generate_all_visualizations
from src.evaluate import get_metrics_table

print("BERT modules imported.")
"""),
        nbf.v4.new_markdown_cell("""### 1. Fine-Tune BERT Model"""),
        nbf.v4.new_code_cell("""model, tokenizer = run_bert_pipeline(subsample_train=600, epochs=2)
"""),
        nbf.v4.new_markdown_cell("""### 2. Hyperparameter Sensitivity Analysis"""),
        nbf.v4.new_code_cell("""tune_bert_learning_rates()
"""),
        nbf.v4.new_markdown_cell("""### 3. Generate Complete Visualizations & Final Metrics Summary"""),
        nbf.v4.new_code_cell("""generate_all_visualizations()
metrics = get_metrics_table()
display(metrics)
""")
    ]
    with open(os.path.join(NOTEBOOKS_DIR, "05_bert_model.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Generated notebooks/05_bert_model.ipynb")

def main():
    create_eda_notebook()
    create_preprocessing_notebook()
    create_classical_notebook()
    create_lstm_notebook()
    create_bert_notebook()
    print("All 5 Jupyter notebooks generated successfully!")

if __name__ == "__main__":
    main()
