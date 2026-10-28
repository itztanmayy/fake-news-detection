# Dataset Documentation: Fake News Detection System

**Course:** CSET485 – AI and Society, Bennett University  
**Team ID:** 39  
**Team Members:** Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka  

---

## 1. Primary In-Domain Dataset: ISOT Fake News Dataset
- **Authors:** H. Ahmed, I. Traore, S. Saad (University of Victoria, ISOT Research Lab)
- **Source:** University of Victoria Information Security and Object Technology (ISOT) Research Group / Kaggle / Hugging Face.
- **Composition:**
  - `True.csv`: **21,417** verified factual news articles sourced from Reuters.com.
  - `Fake.csv`: **23,481** fabricated articles flagged as misinformation by PolitiFact and Wikipedia fake-news listings.
  - **Total:** **44,898** news articles (balanced ~48% Real, 52% Fake).
- **Attributes:**
  - `title`: Headline of the news article.
  - `text`: Full body text of the article.
  - `subject`: Domain category (politicsNews, worldnews, Government News, Middle-east, US_News, News).
  - `date`: Publication date.
- **Data Partitions:**
  - Stratified 70% Train, 15% Validation, 15% Test split:
    - `data/processed/isot_train.csv` (31,422 articles)
    - `data/processed/isot_val.csv` (6,733 articles)
    - `data/processed/isot_test.csv` (6,734 articles)
    - `data/processed/isot_full.csv` (44,889 cleaned articles)

---

## 2. Cross-Dataset Generalization Benchmark: LIAR Benchmark
- **Author:** William Yang Wang (UC Santa Barbara, ACL 2017)
- **Paper:** *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection*
- **Source:** PolitiFact truth-o-meter statements.
- **Structure:**
  - Short political statements rated across 6 fine-grained categories: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`.
- **Binary Mapping for Cross-Domain Testing:**
  - `0 (Fake)`: `pants-fire`, `false`, `barely-true`
  - `1 (Real)`: `mostly-true`, `true`, `half-true`
- **Files Stored Under `data/raw/liar/`:**
  - `train.tsv` (10,240 records)
  - `valid.tsv` (1,284 records)
  - `test.tsv` (1,267 records)

---

## 3. Directory Layout
```
data/
├── raw/
│   ├── isot/
│   │   ├── True.csv
│   │   └── Fake.csv
│   └── liar/
│       ├── train.tsv
│       ├── valid.tsv
│       └── test.tsv
├── processed/
│   ├── isot_full.csv
│   ├── isot_train.csv
│   ├── isot_val.csv
│   └── isot_test.csv
└── README.md
```

---

## 4. Automatic Ingestion & Cleaning
To reproduce the full ingestion and preprocessing pipeline from scratch:
```bash
py -3 src/download_data.py
py -3 src/preprocess.py
```
Preprocessed CSVs include:
- `clean_text`: Full token cleaning with lowercasing, regex sanitization, stopwords removal, and WordNet lemmatization (for TF-IDF and BiLSTM).
- `clean_text_bert`: Contextual cleaning preserving sentence punctuation, capitalization cues, and natural grammar (for HuggingFace BERT).

