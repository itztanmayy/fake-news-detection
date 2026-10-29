# Fake News Detection Using Natural Language Processing
### Coursework Project Report & Implementation | Course: CSET485 – AI and Society
**School of Engineering and Applied Sciences, Bennett University**

---

## 👥 Team Details — Team ID: 39

| Name | Role | Bennett University Email |
| :--- | :--- | :--- |
| **Tanmay Awal** | NLP Pipeline Architecture & Model Engineering | tanmay@bennett.edu.in |
| **Pratham Mishra** | Exploratory Data Analysis & Feature Engineering | pratham@bennett.edu.in |
| **Arpit Sharma** | Deep Learning & Transformer Optimization | arpit@bennett.edu.in |
| **Akshat Dhaka** | Evaluation, Error Analysis & UI Deployment | akshat@bennett.edu.in |

---

## 📌 Executive Summary & Problem Formulation
The viral dissemination of misinformation on digital social media platforms presents profound ethical, political, and societal challenges. This project implements an end-to-end Machine Learning and Natural Language Processing framework capable of automatically identifying and classifying fake news articles. 

We formulate fake news detection as a **supervised binary text classification problem**:
$$\mathcal{F}: \mathcal{T} \to \{0, 1\}$$
where $\mathcal{T}$ represents an input news document (title + full body text), $1$ denotes a **REAL / Factual** article, and $0$ denotes a **FAKE / Misinformation** article.

We evaluate a progressive hierarchy of NLP paradigms:
1. **Classical Machine Learning:** Logistic Regression & Support Vector Machine (LinearSVC) over Term Frequency-Inverse Document Frequency (TF-IDF) n-gram vector representations.
2. **Deep Sequential Learning:** Bidirectional Long Short-Term Memory network (BiLSTM) with dense word embeddings capturing bidirectional contextual flow.
3. **Pretrained Transformer Representation:** HuggingFace `bert-base-uncased` fine-tuned with AdamW and linear learning rate warmup.
4. **Cross-Dataset Generalization Analysis:** Rigorous out-of-domain evaluation on William Yang Wang's **LIAR benchmark**, exposing the stylometric brittleness of in-domain news classifiers.

---

## 🏗️ Architecture & Pipeline Overview

```
                      +-----------------------------+
                      | Raw News Datasets           |
                      | - ISOT (44,898 articles)    |
                      | - LIAR Benchmark (1,267)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Preprocessing Engine        |
                      | - Regex Cleaning & Normaliz |
                      | - Stopword & Lemmatization  |
                      | - Dual-stream (Clean & BERT)|
                      +--------------+--------------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
        +----------------------+             +----------------------+
        | Sparse N-Gram Space  |             | Dense Token Space    |
        | - Sublinear TF-IDF   |             | - Custom BiLSTM Vocab|
        | - Unigram + Bigrams  |             | - WordPiece BPE BERT |
        | - 50,000 Dimensions  |             | - Max Length 64-200  |
        +----------+-----------+             +----------+-----------+
                   |                                    |
          +--------+--------+                  +--------+--------+
          |                 |                  |                 |
          v                 v                  v                 v
   +--------------+  +--------------+   +--------------+  +--------------+
   | Logistic Reg |  | Linear SVM   |   | BiLSTM (PyT) |  | BERT Transf. |
   | L2, C=1.0    |  | C=1.0, Hinge |   | 2-Dir Hidden |  | bert-base-unc|
   +-------+------+  +-------+------+   +-------+------+  +-------+------+
           |                 |                  |                 |
           +-----------------+------------------+-----------------+
                                     |
                                     v
                      +-----------------------------+
                      | Comprehensive Evaluation    |
                      | - Accuracy, Macro P / R / F1|
                      | - Confusion Matrix Heatmaps |
                      | - Cross-Domain Domain Shift |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Production Deployment       |
                      | - Streamlit Interactive App |
                      | - Real-time Inference Engine|
                      +-----------------------------+
```

---

## 📊 Datasets & Preprocessing

### 1. In-Domain Dataset: ISOT Fake News Dataset
- **Volume:** **44,898 articles** (21,417 Real articles from Reuters; 23,481 Fake articles flagged by PolitiFact and Wikipedia).
- **Partition:** Stratified **70% Training** (31,422 articles), **15% Validation** (6,733 articles), and **15% Test** (6,734 articles).
- **Balanced Class Distribution:** 52.3% Fake vs 47.7% Real.

### 2. Generalization Benchmark: LIAR Dataset
- **Origin:** William Yang Wang (ACL 2017), PolitiFact truth-o-meter statements.
- **Evaluation Size:** 1,267 independent short claims mapped into binary categories.

### 3. Dual-Stream Preprocessing Strategy
- **Stream A (Full Cleaning for Classical & BiLSTM):**
  - Lowercasing, removal of URLs, email addresses, and bracketed editorial notes.
  - Punctuation stripping, regex alphanumeric token filtering.
  - English NLTK stopwords elimination.
  - Morphological lemmatization via WordNetLemmatizer.
- **Stream B (Contextual Cleaning for Transformers):**
  - Retains casing signals, question marks, exclamation marks, and syntax structure required by BERT's multi-head self-attention mechanisms.

---

## 🔬 Model Architectures & Methodologies

### 1. Logistic Regression (TF-IDF Baseline)
- **Features:** 50,000 sublinear TF-IDF features ($1$-gram and $2$-gram combinations).
- **Objective:** Minimizes binary cross-entropy with $L_2$ regularization penalty:
  $$\min_{\mathbf{w}} \frac{1}{2}\mathbf{w}^T\mathbf{w} + C \sum_{i=1}^N \log(1 + \exp(-y_i \mathbf{w}^T \mathbf{x}_i))$$
- **Optimal Hyperparameter:** $C = 1.0$.

### 2. Support Vector Machine (LinearSVC)
- **Features:** High-dimensional TF-IDF vectors normalized to unit Euclidean norm.
- **Formulation:** Soft-margin hyperplane optimization maximizing geometric margin:
  $$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$$
- **Optimal Hyperparameter:** $C = 1.0$.

### 3. Bidirectional LSTM (PyTorch)
- **Embedding Layer:** Vocabulary size 20,000 tokens mapped to 128-dimensional dense continuous vectors.
- **Bidirectional Recurrent Layer:** Forward LSTM ($\vec{h}_t$) concatenated with Backward LSTM ($\overleftarrow{h}_t$) yields a 256-dimensional hidden state sequence:
  $$h_t = [\vec{h}_t \,\|\, \overleftarrow{h}_t]$$
- **Classification Head:** Global Max Pooling $\to$ Fully-Connected Layer (64 units, ReLU) $\to$ Dropout ($p=0.3$) $\to$ Linear Output (1 logit) with `BCEWithLogitsLoss`.
- **Optimizer:** Adam ($\alpha = 0.001$, $\beta_1=0.9$, $\beta_2=0.999$, batch size 64).

### 4. BERT Fine-Tuning (`bert-base-uncased`)
- **Base Architecture:** 12 Transformer layers, 12 attention heads, 768 hidden dimension (110M parameters).
- **Optimization Strategy:** AdamW optimizer with decoupled weight decay (0.01), linear warmup schedule across the first 10% steps, and upper transformer blocks fine-tuning.
- **Input Representation:** Subword WordPiece tokenization with `[CLS]` classification token, positional encodings, and attention masks.

---

## 📈 Empirical Results & Benchmark Comparisons

### In-Domain Performance on ISOT Test Set (6,734 Samples)

| Model Architecture | Feature Representation | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | TF-IDF (1,2-grams, 50k) | **99.33%** | 0.9933 | 0.9933 | **0.9933** |
| **Support Vector Machine (LinearSVC)** | TF-IDF (1,2-grams, 50k) | **99.82%** | 0.9982 | 0.9982 | **0.9982** |
| **Bidirectional LSTM (PyTorch)** | Dense Word Embeddings (128-d) | **99.73%** | 0.9974 | 0.9973 | **0.9973** |
| **BERT Fine-Tuned (HuggingFace)** | Subword Attention (768-d) | **96.80%** | 0.9685 | 0.9678 | **0.9681** |

> [!NOTE]
> All models demonstrate high discrimination on the ISOT benchmark (>96-99%), successfully distinguishing Reuters journalistic conventions from opinion-blog misinformation.

---

### Out-of-Domain Generalization Gap on LIAR Benchmark (1,267 Samples)

| Model Architecture | In-Domain (ISOT) Acc. | Out-of-Domain (LIAR) Acc. | Generalization Gap ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 99.33% | 45.62% | **-53.71%** |
| **Support Vector Machine** | 99.82% | 44.51% | **-55.31%** |
| **Bidirectional LSTM** | 99.73% | 43.65% | **-56.08%** |
| **BERT Fine-Tuned** | 96.80% | 48.20% | **-48.60%** |

### Key Theoretical & Empirical Insights:
1. **The In-Domain vs. Cross-Domain Paradox:** While models achieve near-perfect metrics on in-domain news articles, accuracy plummets to ~44-48% on short, out-of-domain political claims.
2. **Stylometric Overfitting:** Classifiers trained on full-length articles heavily exploit artifactual markers (e.g., editorial attribution, Reuters city datelines, article length) rather than invariant semantic truthfulness.
3. **Transformer Resilience:** Fine-tuned BERT exhibits greater transfer resilience due to deep contextual language understanding, demonstrating why pre-trained language models are critical for robust societal AI deployment.

---

## 📊 Visualizations & Artifacts Generated

All high-resolution figures are automatically generated and saved under `results/`:

| Visualization | Artifact File | Description |
| :--- | :--- | :--- |
| **Class Distribution** | `results/plots/class_distribution.png` | Real vs Fake sample balance across 44,898 articles |
| **Real vs Fake Word Clouds** | `results/plots/wordclouds_real_vs_fake.png` | Top 100 distinctive vocabulary items |
| **BiLSTM Training Curves** | `results/plots/lstm_training_curves.png` | Train/Val Loss and Accuracy across training epochs |
| **BERT Training Curves** | `results/plots/bert_training_curves.png` | Cross-Entropy Loss & Accuracy progression |
| **LR Hyperparameter Tuning** | `results/plots/classical_hyperparameter_tuning.png` | Regularization parameter $C \in [0.01, 10.0]$ |
| **BiLSTM Tuning Curves** | `results/plots/lstm_hyperparameter_tuning.png` | Learning rate and batch size sensitivity |
| **BERT Tuning Curves** | `results/plots/bert_hyperparameter_tuning.png` | Learning rate sensitivity $[5e-5, 2e-5, 1e-5]$ |
| **Model Comparison Bar Chart**| `results/plots/models_performance_comparison.png` | Side-by-side Accuracy and Macro F1 comparison |
| **Generalization Gap Chart** | `results/plots/cross_dataset_generalization_gap.png` | ISOT vs LIAR out-of-domain performance contrast |
| **Confusion Matrices** | `results/confusion_matrices/*.png` | Normalized confusion matrix heatmaps for all models |

---

## 💻 Interactive Streamlit Application (`app.py`)

An interactive web application is provided for real-time model inference, exploratory data analysis, and comparative inspection.

### Key Capabilities:
- **Real-Time Classification:** Enter any news headline and body text to receive an instantaneous prediction (REAL vs FAKE) with probability confidence bars.
- **Model Selector:** Choose among trained checkpoints available in `models/`.
- **Optional Samples:** Quick-test preloaded real and fake stories, or enter any headline and article body.
- **Interactive Metrics Dashboard:** Inspect live metric tables and confusion matrices.
- **Visual Analytics Tab:** View word clouds, class balance charts, and cross-dataset shift graphs.

### To Launch the Web Application:
```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Access the application at `http://localhost:8501`.

The classifier also includes **Gemini (Google)** with Google Search grounding.
It searches current web sources, displays returned citations, and can still mark
a story unverifiable when the available evidence is insufficient. Put your API
key in the project-root `.env` file:
```powershell
GEMINI_API_KEY=your-api-key
```
Alternatively, set it in PowerShell before starting Streamlit:
```powershell
$env:GEMINI_API_KEY = "your-api-key"
python -m streamlit run app.py
```
The `.env` file is git-ignored; never commit or share the key. Treat Gemini's
assessment as a starting point and review its cited sources before sharing a story.

---

## 📂 Repository Structure

```
Fake News Detection System/
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Verified Python dependencies
├── .gitignore                 # Standard Python gitignore
├── README.md                  # Master documentation report
├── data/
│   ├── README.md              # Dataset provenance documentation
│   ├── raw/
│   │   ├── isot/              # True.csv and Fake.csv
│   │   └── liar/              # train.tsv, valid.tsv, test.tsv
│   └── processed/
│       ├── isot_train.csv     # 31,422 training samples
│       ├── isot_val.csv       # 6,733 validation samples
│       ├── isot_test.csv      # 6,734 testing samples
│       └── isot_full.csv      # 44,889 cleaned samples
├── models/
│   ├── tfidf_vectorizer.pkl   # Fitted Scikit-Learn TF-IDF vectorizer
│   ├── lr_model.pkl           # Trained Logistic Regression model
│   ├── svm_model.pkl          # Trained LinearSVC model
│   ├── lstm_vocab.pkl         # LSTM token vocabulary dictionary
│   ├── lstm_model.pt          # PyTorch BiLSTM state dictionary
│   └── bert_finetuned/        # Saved HuggingFace BERT checkpoint & tokenizer
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis & visual profiling
│   ├── 02_preprocessing.ipynb # Text cleaning, tokenization & dataset splits
│   ├── 03_classical_models.ipynb # TF-IDF, Logistic Regression & Linear SVM
│   ├── 04_lstm_model.ipynb    # PyTorch BiLSTM modeling & training curves
│   └── 05_bert_model.ipynb    # HuggingFace BERT fine-tuning & evaluation
├── results/
│   ├── metrics_summary.csv    # Consolidated performance benchmark table
│   ├── confusion_matrices/    # Heatmap PNGs for each model
│   └── plots/                 # All diagnostic and comparison charts
└── src/
    ├── __init__.py            # Python package initialization
    ├── utils.py               # Path definitions, device selector, seed locking
    ├── download_data.py       # Automated dataset downloader
    ├── preprocess.py          # Parallelized text cleaning & splitting engine
    ├── features.py            # TF-IDF, LSTM vocabulary, and BERT datasets
    ├── evaluate.py            # Classification metrics & confusion matrix plotters
    ├── train_classical.py     # Training and tuning script for LR and SVM
    ├── train_lstm.py          # PyTorch BiLSTM training and tuning script
    ├── train_bert.py          # HuggingFace BERT fine-tuning & tuning script
    ├── visualize.py           # Master visualization generator script
    └── create_notebooks.py    # Automated notebook generator script
```

---

## 🚀 Step-by-Step Reproduction Guide

### 1. Environment Setup
Clone the repository and install required packages:
```bash
py -3 -m pip install -r requirements.txt
```

### 2. Ingest Data & Execute Preprocessing
```bash
py -3 src/download_data.py
py -3 src/preprocess.py
```

### 3. Train Classical Baseline Models (LR & SVM)
```bash
py -3 src/train_classical.py
```

### 4. Train Deep Sequential Model (BiLSTM)
```bash
py -3 src/train_lstm.py
```

### 5. Fine-Tune Pretrained Transformer (BERT)
```bash
py -3 src/train_bert.py
```

### 6. Compile Visualizations & Metrics
```bash
py -3 src/visualize.py
```

### 7. Run Interactive Streamlit UI
```bash
python -m streamlit run app.py
```

---

## 📜 Ethical Considerations & AI and Society Implications
- **Dual-Use Risks:** Automated fake news detection systems could potentially be inverted to generate adversarial misinformation that evades automated filters.
- **Fairness & Censorship:** Overly aggressive classification models risk suppressing legitimate political discourse, non-standard dialectal expression, or satire.
- **Human-in-the-Loop Imperative:** Automated NLP classifiers must be deployed as assistive tools alongside professional human fact-checkers, rather than autonomous gatekeepers of digital discourse.

---
**Course:** CSET485 – AI and Society | **Semester:** Odd Semester 2026 | **Bennett University**  
*Team ID: 39 | Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka*
