"""
Streamlit Web Application for Fake News Detection System.
Course: CSET485 - AI and Society, Bennett University
Team ID: 39 | Members: Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka
"""
import os
import sys
if os.path.abspath(os.path.dirname(__file__)) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import torch

from src.utils import MODELS_DIR, RESULTS_DIR, PLOTS_DIR, CM_DIR, get_device
from src.preprocess import clean_text, clean_text_bert
from src.features import TfidfFeatureExtractor, LSTMVocab
from src.train_lstm import BiLSTMClassifier
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("📰 Fake News Detection System")
st.markdown("""
**Course:** CSET485 – AI and Society, Bennett University  
**Team ID:** 39 | **Team Members:** Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka
---
""")

# Sidebar Navigation
sidebar_option = st.sidebar.radio(
    "Navigation",
    ["🔎 Live Article Classifier", "📊 Model Comparison & Metrics", "📈 Visualizations & EDA", "ℹ️ About the Project"]
)

@st.cache_resource
def load_classical_models():
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    lr_path = os.path.join(MODELS_DIR, "lr_model.pkl")
    svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    
    tfidf = TfidfFeatureExtractor.load(tfidf_path) if os.path.exists(tfidf_path) else None
    lr = pd.read_pickle(lr_path) if os.path.exists(lr_path) else None
    svm = pd.read_pickle(svm_path) if os.path.exists(svm_path) else None
    return tfidf, lr, svm

@st.cache_resource
def load_lstm_model():
    vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
    weights_path = os.path.join(MODELS_DIR, "lstm_model.pt")
    
    if not (os.path.exists(vocab_path) and os.path.exists(weights_path)):
        return None, None
        
    vocab = LSTMVocab.load(vocab_path)
    device = get_device()
    model = BiLSTMClassifier(vocab_size=len(vocab.word2idx), embed_dim=128, hidden_dim=128, dense_dim=64, dropout_p=0.3)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return vocab, model

@st.cache_resource
def load_bert_model():
    bert_dir = os.path.join(MODELS_DIR, "bert_finetuned")
    if not os.path.exists(bert_dir) or not os.path.exists(os.path.join(bert_dir, "config.json")):
        return None, None
    device = get_device()
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    model = BertForSequenceClassification.from_pretrained(bert_dir).to(device)
    model.eval()
    return tokenizer, model

# PAGE 1: LIVE CLASSIFIER
if sidebar_option == "🔎 Live Article Classifier":
    st.subheader("Test an Article with Machine Learning & Deep Learning Models")
    
    model_choice = st.selectbox(
        "Choose Classification Model:",
        ["Logistic Regression (TF-IDF Baseline)", "Support Vector Machine (LinearSVC)", "Bidirectional LSTM (PyTorch)", "BERT Fine-Tuned (HuggingFace Transformers)"]
    )
    
    sample_options = [
        "Select a sample article...",
        "REAL: U.S. House passes landmark infrastructure investment bill",
        "FAKE: Secret alien treaty signed in underground bunker according to leaked document",
        "REAL: NASA rover discovers organic molecules on Mars surface",
        "FAKE: Pope Francis endorses Donald Trump for President shocking Vatican officials"
    ]
    sample_select = st.selectbox("Or choose a pre-filled sample news article:", sample_options)
    
    default_title = ""
    default_text = ""
    if sample_select == sample_options[1]:
        default_title = "U.S. House passes landmark infrastructure investment bill"
        default_text = "WASHINGTON (Reuters) - The United States House of Representatives approved a major bipartisan infrastructure bill on Friday evening, providing over $1 trillion in federal funding for roads, bridges, public transit, and clean drinking water infrastructure nationwide."
    elif sample_select == sample_options[2]:
        default_title = "Secret alien treaty signed in underground bunker"
        default_text = "Shocking documents leaked by an anonymous whistleblower inside the Pentagon prove global elites met in a classified underground base in Nevada to sign a galactic peace treaty. Mainstream media refuses to report this massive cover-up!"
    elif sample_select == sample_options[3]:
        default_title = "NASA rover discovers organic molecules on Mars surface"
        default_text = "WASHINGTON (Reuters) - NASA's Perseverance rover has collected promising rock samples within an ancient river delta in Jezero Crater that contain complex carbon-based molecules, mission scientists announced at a press briefing on Tuesday."
    elif sample_select == sample_options[4]:
        default_title = "Pope Francis endorses Donald Trump for President"
        default_text = "VATICAN CITY - In an unprecedented move, Pope Francis has officially endorsed Donald Trump for President of the United States, citing conservative values and the defense of religious freedom. The Vatican confirmed the endorsement via social media."

    title_input = st.text_input("Article Headline / Title:", value=default_title)
    text_input = st.text_area("Article Body Text:", value=default_text, height=180)
    
    if st.button("🚀 Analyze News Article", type="primary"):
        if not title_input.strip() and not text_input.strip():
            st.error("Please enter a title or text to analyze.")
        else:
            combined = title_input + " " + text_input
            with st.spinner("Analyzing text and evaluating linguistic patterns..."):
                tfidf, lr, svm = load_classical_models()
                vocab, lstm_model = load_lstm_model()
                bert_tokenizer, bert_model = load_bert_model()
                
                prediction = None
                confidence = None
                model_used = model_choice
                
                if "Logistic Regression" in model_choice:
                    if tfidf is not None and lr is not None:
                        clean = clean_text(combined)
                        features = tfidf.transform([clean])
                        prob = lr.predict_proba(features)[0]
                        pred_idx = np.argmax(prob)
                        prediction = "REAL NEWS" if pred_idx == 1 else "FAKE NEWS"
                        confidence = prob[pred_idx]
                    else:
                        st.warning("Logistic Regression model checkpoint not found. Please train models first.")
                        
                elif "Support Vector Machine" in model_choice:
                    if tfidf is not None and svm is not None:
                        clean = clean_text(combined)
                        features = tfidf.transform([clean])
                        decision = svm.decision_function(features)[0]
                        pred_idx = 1 if decision > 0 else 0
                        prediction = "REAL NEWS" if pred_idx == 1 else "FAKE NEWS"
                        # Platt-like sigmoid calibration for confidence display
                        confidence = 1 / (1 + np.exp(-abs(decision)))
                    else:
                        st.warning("SVM model checkpoint not found. Please train models first.")
                        
                elif "Bidirectional LSTM" in model_choice:
                    if vocab is not None and lstm_model is not None:
                        clean = clean_text(combined)
                        encoded = vocab.encode(clean)
                        device = get_device()
                        with torch.no_grad():
                            inp = torch.tensor([encoded], dtype=torch.long).to(device)
                            logit = lstm_model(inp)
                            prob = torch.sigmoid(logit).item()
                        pred_idx = 1 if prob >= 0.5 else 0
                        prediction = "REAL NEWS" if pred_idx == 1 else "FAKE NEWS"
                        confidence = prob if pred_idx == 1 else (1.0 - prob)
                    else:
                        st.warning("BiLSTM model checkpoint not found. Please train models first.")
                        
                elif "BERT" in model_choice:
                    if bert_tokenizer is not None and bert_model is not None:
                        clean_bert = clean_text_bert(combined)
                        device = get_device()
                        inputs = bert_tokenizer(clean_bert, max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = bert_model(**inputs)
                            probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
                        pred_idx = np.argmax(probs)
                        prediction = "REAL NEWS" if pred_idx == 1 else "FAKE NEWS"
                        confidence = probs[pred_idx]
                    else:
                        st.warning("BERT model checkpoint not found. Please train models first.")

                if prediction is not None:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if prediction == "REAL NEWS":
                            st.success(f"### Result: ✅ {prediction}")
                        else:
                            st.error(f"### Result: 🚨 {prediction}")
                        st.write(f"**Selected Model:** {model_used}")
                        st.write(f"**Confidence Score:** `{confidence * 100:.2f}%`")
                        st.progress(float(confidence))
                    with col2:
                        st.markdown("#### Text Cleaning Preview")
                        st.markdown(f"**Original Length:** {len(combined)} characters")
                        st.text_area("Preprocessed Text Representation:", clean_text(combined)[:300] + "...", height=100)

# PAGE 2: METRICS TABLE
elif sidebar_option == "📊 Model Comparison & Metrics":
    st.subheader("Experimental Results & Unified Metrics Summary")
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        st.dataframe(df_summary.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}"
        }), use_container_width=True)
    else:
        st.info("No metrics recorded yet. Run `src/train_classical.py`, `src/train_lstm.py`, and `src/train_bert.py` to populate.")
        
    st.markdown("---")
    st.markdown("### Confusion Matrices")
    cm_files = [f for f in os.listdir(CM_DIR) if f.endswith(".png")] if os.path.exists(CM_DIR) else []
    if cm_files:
        cols = st.columns(min(len(cm_files), 2))
        for idx, cf in enumerate(cm_files):
            col = cols[idx % 2]
            with col:
                st.image(os.path.join(CM_DIR, cf), caption=cf.replace("_cm.png", "").upper() + " Confusion Matrix", use_container_width=True)
    else:
        st.write("Confusion matrix plots will appear here after model evaluation.")

# PAGE 3: VISUALIZATIONS
elif sidebar_option == "📈 Visualizations & EDA":
    st.subheader("Data & Model Visualizations")
    plot_files = [
        ("Class Distribution", "class_distribution.png"),
        ("Word Clouds (Real vs Fake)", "wordclouds_real_vs_fake.png"),
        ("Model Performance Comparison", "models_performance_comparison.png"),
        ("Cross-Dataset Generalization Gap", "cross_dataset_generalization_gap.png"),
        ("Classical Hyperparameter Tuning", "classical_hyperparameter_tuning.png"),
        ("LSTM Training Curves", "lstm_training_curves.png"),
        ("LSTM Hyperparameter Tuning", "lstm_hyperparameter_tuning.png"),
        ("BERT Training Curves", "bert_training_curves.png")
    ]
    for title, fname in plot_files:
        path = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"#### {title}")
            st.image(path, use_container_width=True)
            st.markdown("---")

# PAGE 4: ABOUT
else:
    st.subheader("About the Project")
    st.markdown("""
    ### Fake News Detection Using Natural Language Processing
    - **Course:** CSET485 – AI and Society, Bennett University
    - **Team ID:** 39
    - **Team Members:** Tanmay Awal, Pratham Mishra, Arpit Sharma, Akshat Dhaka
    
    #### Pipeline Architecture
    1. **Data Ingestion & Stratified Splitting:** 44,898 articles from the ISOT dataset (21,417 Real + 23,481 Fake) split into 70% Train, 15% Val, 15% Test.
    2. **Multi-Stage Preprocessing:**
       - Full cleaning for TF-IDF / LSTM (Lowercasing, URLs/HTML removal, punctuation removal, digits removal, tokenization, stopword filtering, WordNet lemmatization).
       - Light contextual cleaning for BERT.
    3. **Model Implementations:**
       - **Baseline:** Logistic Regression with Sublinear TF-IDF (1-2 ngrams, 50k features).
       - **Classical:** Linear Support Vector Machine (LinearSVC).
       - **Deep Learning:** Bidirectional LSTM with 128 embedding dims, 128 hidden units each direction, Dropout (0.3), and Dense head.
       - **Transformer:** Fine-Tuned BERT (`bert-base-uncased`) with AdamW and linear learning rate warmup.
    4. **Cross-Dataset Generalization:** Evaluated against William Yang Wang's LIAR dataset to rigorously assess out-of-domain robustness.
    """)
