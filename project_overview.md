# Project Overview: Fake News Detection Using Natural Language Processing

## Project Identity
- **Title:** Fake News Detection Using Natural Language Processing
- **Course:** CSET485 – AI and Society, Bennett University
- **Team ID:** 39
- **Team Members:** Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka

---

## 1. Goal

Build a binary text classification system that takes a news article (headline + body text, or just body text) as input and classifies it as **REAL** or **FAKE**. The system must:

1. Implement multiple models — from classical ML to deep learning — on the same dataset.
2. Compare their performance using standard classification metrics.
3. Demonstrate that a transformer-based model (BERT) outperforms simpler baselines.
4. Produce a clean, reproducible pipeline with separate modules for each stage.

---

## 2. Tech Stack

| Component | Tool / Library |
|---|---|
| Language | Python 3.10+ |
| Data handling | pandas, numpy |
| NLP preprocessing | nltk, re, string |
| Classical ML | scikit-learn |
| Deep Learning | PyTorch or TensorFlow/Keras |
| Transformers | HuggingFace `transformers`, `datasets` |
| Evaluation | scikit-learn metrics |
| Visualization | matplotlib, seaborn |
| Environment | Google Colab (GPU runtime) or local with CUDA |
| Version control | Git + GitHub |

---

## 3. Project Directory Structure

```
fake-news-detection/
│
├── data/
│   ├── raw/                    # Original downloaded datasets (do not modify)
│   │   ├── isot/
│   │   │   ├── True.csv
│   │   │   └── Fake.csv
│   │   └── liar/
│   │       ├── train.tsv
│   │       ├── valid.tsv
│   │       └── test.tsv
│   └── processed/              # Cleaned, merged, split datasets
│       ├── isot_full.csv
│       ├── isot_train.csv
│       ├── isot_val.csv
│       └── isot_test.csv
│
├── notebooks/                  # Jupyter notebooks (exploratory + demo)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_models.ipynb
│   ├── 04_lstm_model.ipynb
│   └── 05_bert_model.ipynb
│
├── src/
│   ├── preprocess.py           # Text cleaning and preprocessing functions
│   ├── features.py             # TF-IDF vectorizer, tokenizer wrappers
│   ├── train_classical.py      # Train LR, SVM, Naive Bayes
│   ├── train_lstm.py           # Build and train LSTM model
│   ├── train_bert.py           # Fine-tune BERT
│   ├── evaluate.py             # Unified evaluation and metric reporting
│   └── utils.py                # Shared helper functions
│
├── models/                     # Saved model checkpoints
│   ├── lr_model.pkl
│   ├── svm_model.pkl
│   ├── lstm_model.h5 (or .pt)
│   └── bert_finetuned/
│
├── results/
│   ├── metrics_summary.csv     # All models' metrics in one table
│   ├── confusion_matrices/
│   └── plots/
│
├── requirements.txt
├── README.md
└── project_overview.md         # This file
```

---

## 4. Datasets

### 4.1 Primary: ISOT Fake News Dataset
- **Source:** University of Victoria
- **Files:** `True.csv` (21,417 real articles) and `Fake.csv` (23,481 fake articles)
- **Columns:** `title`, `text`, `subject`, `date`
- **Labels:** Derived — Real.csv articles → label `1`, Fake.csv → label `0`
- **Domain:** Political news and world events (Reuters-sourced real articles; PolitiFact/various fake sources)
- **Use:** Primary training, validation, and testing dataset

### 4.2 Secondary: LIAR Dataset
- **Source:** William Yang Wang, UCSB (2017)
- **Format:** TSV files with 14 columns
- **Relevant columns:** statement text, label (6-class: pants-fire, false, barely-true, half-true, mostly-true, true)
- **Label mapping for binary use:** pants-fire + false + barely-true → FAKE (0); half-true + mostly-true + true → REAL (1)
- **Use:** Cross-dataset generalization test — train on ISOT, test on LIAR to assess robustness

---

## 5. Data Pipeline (Step by Step)

### Step 1: Load and Merge ISOT
- Read `True.csv` and `Fake.csv` separately using pandas.
- Add a `label` column: 1 for real, 0 for fake.
- Concatenate both DataFrames into a single DataFrame.
- Shuffle the combined DataFrame with a fixed random seed (e.g., `random_state=42`).
- Save as `isot_full.csv`.

### Step 2: Text Construction
- For each article, combine `title` and `text` into a single field: `combined_text = title + " " + text`.
- This is the primary input to all models.

### Step 3: Cleaning (`preprocess.py`)
Perform the following operations on the `combined_text` field in order:
1. Convert to lowercase.
2. Remove URLs (regex: `http\S+` or `www\S+`).
3. Remove HTML tags (regex: `<.*?>`).
4. Remove punctuation (using `string.punctuation`).
5. Remove digits and numbers.
6. Tokenize using NLTK word tokenizer.
7. Remove English stopwords (NLTK stopwords list).
8. Apply lemmatization (NLTK WordNetLemmatizer).
9. Rejoin tokens into a cleaned string.
- Store cleaned text in a new column: `clean_text`.
- Note: For BERT, do NOT apply steps 7–9 (stopword removal and lemmatization). BERT uses its own tokenizer and benefits from full text.

### Step 4: Train/Validation/Test Split
- Split `isot_full.csv` into:
  - Training set: 70%
  - Validation set: 15%
  - Test set: 15%
- Use stratified splitting to maintain label balance.
- Save each split as a separate CSV.

---

## 6. Models

Implement the following four models in order of complexity. Each model is trained, validated, and evaluated independently on the same test set.

---

### Model 1: Logistic Regression (Baseline)

**Purpose:** Establish a simple, interpretable baseline.

**Feature extraction:**
- Apply TF-IDF vectorization on `clean_text`.
- Parameters: `max_features=50000`, `ngram_range=(1,2)` (unigrams + bigrams), `sublinear_tf=True`.
- Fit the vectorizer on training data only. Transform val and test using the fitted vectorizer.

**Training:**
- Use `sklearn.linear_model.LogisticRegression`.
- Parameters: `max_iter=1000`, `C=1.0`, `solver='lbfgs'`.
- Train on TF-IDF-transformed training set.

**Evaluation:**
- Predict on test set.
- Report: Accuracy, Precision, Recall, F1-score (macro average).
- Generate confusion matrix.

**Save:** Pickle the fitted TF-IDF vectorizer and the LR model.

---

### Model 2: Support Vector Machine (SVM)

**Purpose:** Strong classical baseline, often competitive with deep learning on text.

**Feature extraction:** Same TF-IDF vectorizer as Model 1 (reuse the fitted vectorizer).

**Training:**
- Use `sklearn.svm.LinearSVC` (faster than kernel SVM for large text data).
- Parameters: `C=1.0`, `max_iter=2000`.

**Evaluation:** Same as Model 1.

---

### Model 3: LSTM (Bidirectional)

**Purpose:** Capture sequential and contextual dependencies in text.

**Preprocessing (additional):**
- Build a vocabulary from the training set `clean_text`. Cap vocabulary at 20,000 most frequent words.
- Map each word to an integer index. Reserve index 0 for padding.
- Pad/truncate sequences to a fixed length of 300 tokens.

**Architecture:**
1. Embedding Layer: 20,000 vocab × 128 dimensions. Initialize randomly (or with GloVe pre-trained weights if available).
2. Bidirectional LSTM Layer: 128 units in each direction (256 total). Apply dropout of 0.3.
3. Dense Layer: 64 units, ReLU activation.
4. Dropout: 0.3.
5. Output Layer: 1 unit, Sigmoid activation (binary classification).

**Training:**
- Loss: Binary Cross-Entropy.
- Optimizer: Adam, learning rate = 0.001.
- Batch size: 64.
- Epochs: 10 (with early stopping: patience=3, monitor validation loss).
- Save the best model checkpoint (lowest validation loss).

**Evaluation:** Same metrics as above. Also plot training/validation loss curves.

---

### Model 4: BERT Fine-tuning

**Purpose:** State-of-the-art transformer model; expected to outperform all baselines.

**Model choice:** `bert-base-uncased` from HuggingFace.

**Preprocessing (BERT-specific):**
- Use `BertTokenizer` from HuggingFace.
- Do NOT use the heavily cleaned `clean_text`. Use lightly cleaned text (only URL removal and HTML stripping).
- Tokenize with: `max_length=256`, `padding='max_length'`, `truncation=True`.
- Output: `input_ids`, `attention_mask`, `token_type_ids`.

**Architecture:**
- Load `BertForSequenceClassification` with `num_labels=2`.
- The BERT encoder's weights are loaded from the pre-trained checkpoint and fine-tuned.
- The classification head is a linear layer on top of the [CLS] token representation.

**Training:**
- Optimizer: AdamW, learning rate = 2e-5, weight decay = 0.01.
- Batch size: 16 (use gradient accumulation of 2 steps if GPU memory is limited, effective batch = 32).
- Epochs: 3 (BERT fine-tuning converges fast; more epochs risk overfitting).
- Use a linear learning rate warmup for the first 10% of steps.
- Save the best model checkpoint.

**Evaluation:** Same metrics. Also report per-class precision/recall.

---

## 7. Evaluation Module (`evaluate.py`)

This module should:
1. Accept a model, a test set, and a model type identifier.
2. Compute and return a dictionary of: Accuracy, Precision (macro), Recall (macro), F1-score (macro).
3. Print a full classification report (per-class breakdown).
4. Generate and save a confusion matrix heatmap using seaborn.
5. Aggregate all models' results into a single summary DataFrame and save as `results/metrics_summary.csv`.

Metrics definitions:
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-score** = 2 × (Precision × Recall) / (Precision + Recall)

---

## 8. Hyperparameter Tuning

For the report, demonstrate tuning on at least two hyperparameters per model:

**Logistic Regression / SVM:**
- Vary C: [0.01, 0.1, 1.0, 10.0]
- Evaluate on validation set for each value of C.
- Report validation accuracy for each.

**LSTM:**
- Vary learning rate: [0.01, 0.001, 0.0001]
- Vary batch size: [32, 64, 128]
- Report validation accuracy for each.

**BERT:**
- Vary learning rate: [5e-5, 2e-5, 1e-5]
- Report validation accuracy for each.

---

## 9. Cross-Dataset Generalization (LIAR)

After training all models on ISOT:
1. Load the LIAR test split.
2. Apply the binary label mapping described in Section 4.2.
3. Preprocess identically to the ISOT pipeline.
4. Run inference using the best-performing model (expected: BERT).
5. Report accuracy and F1-score on LIAR test set.
6. Compare ISOT test performance vs LIAR test performance to highlight generalization gap.

---

## 10. Visualizations to Generate

1. **Class distribution bar chart** — ISOT dataset (real vs fake count).
2. **Word clouds** — one for fake articles, one for real articles (top 100 words after preprocessing).
3. **Training curves** — loss and accuracy vs epoch for LSTM and BERT.
4. **Confusion matrices** — one per model, as heatmaps.
5. **Performance comparison bar chart** — all four models side-by-side on accuracy and F1-score.
6. **Hyperparameter sensitivity plots** — validation accuracy vs hyperparameter value.

---

## 11. Expected Approximate Results (for reference)

These are literature-informed expectations. Actual results will vary:

| Model | Expected Accuracy (ISOT) |
|---|---|
| Logistic Regression | ~94–96% |
| SVM | ~95–97% |
| Bidirectional LSTM | ~97–98% |
| BERT fine-tuned | ~99%+ |

Note: ISOT is a relatively clean dataset. Cross-dataset results on LIAR will be significantly lower (~65–75%) for all models, which is expected and should be discussed.

---

## 12. Implementation Order (Recommended)

1. Download and explore datasets (EDA notebook).
2. Build and test `preprocess.py`.
3. Train and evaluate Logistic Regression → confirm pipeline works end-to-end.
4. Train and evaluate SVM.
5. Build LSTM → train → evaluate.
6. Fine-tune BERT → evaluate.
7. Run hyperparameter tuning experiments.
8. Run cross-dataset test on LIAR.
9. Generate all plots and save to `results/`.
10. Compile `metrics_summary.csv`.

---

## 13. GitHub Repository Requirements

- All source files in `src/` must be committed.
- All notebooks in `notebooks/` must have outputs cleared before final commit (to reduce repo size).
- `requirements.txt` must list all dependencies with version pins.
- `README.md` must include: project description, setup instructions, how to run each model, and a results summary table.
- Do NOT commit large data files. Add `data/raw/` to `.gitignore`. Include a `data/README.md` with download links instead.
- Model checkpoints in `models/` may be committed if under 100MB, otherwise add to `.gitignore` and document how to reproduce them.
