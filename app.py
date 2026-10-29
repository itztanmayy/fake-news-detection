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
    page_title="News Signal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root { --ink: #202b25; --muted: #59645d; --canvas: #f3f5f1; --line: #dce2dc; --green: #2e5b45; --red: #a9483c; }
[data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #e9ede8; border-right: 1px solid var(--line); }
.block-container { max-width: 1280px; padding: 1.75rem 2.5rem 3.5rem; }
html, body, [class*="st-"], button, input, textarea { font-family: "Aptos", "Segoe UI", sans-serif; }
h1, h2, h3, h4 { color: var(--ink); }
h1 { font-family: Georgia, "Times New Roman", serif !important; font-size: 2rem !important; font-weight: 500 !important; }
h2, h3 { font-weight: 600 !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] { color: var(--muted); }
.brand-row { display: flex; align-items: center; gap: 0.8rem; margin: 0.2rem 0 1.4rem; }
.brand-mark { display: grid; place-items: center; width: 38px; height: 38px; flex: 0 0 38px; border-radius: 4px; background: var(--green); color: white; font: 600 15px Georgia, serif; }
.brand-name { margin: 0; color: var(--ink); font: 500 28px Georgia, "Times New Roman", serif; }
.brand-rule { height: 1px; margin: 0 0 1.7rem; background: var(--line); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #dfe6df; }
[data-testid="stBaseButton-primary"] { min-height: 2.8rem; padding: 0 1.2rem; border: 1px solid var(--green); border-radius: 4px; background: var(--green); color: white; font-weight: 600; transition: background 150ms ease; }
[data-testid="stBaseButton-primary"] p { color: white !important; }
[data-testid="stBaseButton-primary"]:hover { border-color: #244a37; background: #244a37; color: white; }
[data-testid="stBaseButton-primary"]:focus-visible { outline: 3px solid #93b7a0; outline-offset: 2px; }
[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea { border-color: #cbd4cd !important; border-radius: 4px !important; background: white !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #59645d !important; opacity: 1; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--green) !important; box-shadow: 0 0 0 1px var(--green) !important; }
[data-testid="stAlert"] { border-radius: 4px; }
[data-testid="stProgress"] > div > div { background: var(--green); }
.result-panel { padding: 1.15rem 1.25rem; border: 1px solid var(--line); border-radius: 4px; background: white; }
.result-label { margin-bottom: 0.25rem; color: var(--muted); font-size: 0.82rem; }
.result-value { margin: 0; font: 500 1.55rem Georgia, "Times New Roman", serif; }
.result-value.real { color: var(--green); }
.result-value.fake { color: var(--red); }
.result-meta { margin-top: 0.55rem; color: var(--muted); font-size: 0.9rem; }
@media (max-width: 700px) { .block-container { padding: 1.2rem 1rem 2.5rem; } .brand-name { font-size: 25px; } }
button[aria-label="Expand sidebar"] [data-testid="stIconMaterial"],
button[aria-label="Collapse sidebar"] [data-testid="stIconMaterial"] { font-size: 0 !important; }
button[aria-label="Expand sidebar"] [data-testid="stIconMaterial"]::after { content: ">"; font: 22px/1 "Aptos", "Segoe UI", sans-serif; }
button[aria-label="Collapse sidebar"] [data-testid="stIconMaterial"]::after { content: "<"; font: 22px/1 "Aptos", "Segoe UI", sans-serif; }
details > summary [data-testid="stIconMaterial"] { font-size: 0 !important; }
details > summary [data-testid="stIconMaterial"]::after { content: ">"; font: 18px/1 "Aptos", "Segoe UI", sans-serif; }
details[open] > summary [data-testid="stIconMaterial"]::after { content: "v"; }
@media (max-width: 700px) {
    .block-container { padding: 1.2rem 1rem 2.5rem; }
    .brand-name { font-size: 25px; }
    button[aria-label="Expand sidebar"] { position: fixed !important; top: 10px !important; right: 12px !important; left: auto !important; z-index: 1002; }
}
</style>
<div class="brand-row"><div class="brand-mark">NS</div><div class="brand-name">News Signal</div></div>
<div class="brand-rule"></div>
""", unsafe_allow_html=True)

# Sidebar Navigation
sidebar_option = st.sidebar.radio(
    "Sections",
    ["Classifier", "Results", "Visuals", "About"]
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
if sidebar_option == "Classifier":
    st.subheader("Article classification")

    available_model_options = []
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_path) and os.path.exists(os.path.join(MODELS_DIR, "lr_model.pkl")):
        available_model_options.append("Logistic Regression")
    if os.path.exists(tfidf_path) and os.path.exists(os.path.join(MODELS_DIR, "svm_model.pkl")):
        available_model_options.append("Linear SVM")
    if all(os.path.exists(os.path.join(MODELS_DIR, filename)) for filename in ("lstm_vocab.pkl", "lstm_model.pt")):
        available_model_options.append("BiLSTM")
    if not available_model_options:
        st.error("No complete trained model checkpoints were found in the models folder. Train a model before analyzing articles.")

    model_column, sample_column = st.columns(2)
    with model_column:
        model_choice = st.selectbox(
            "Model",
            available_model_options or ["No trained models found"],
            disabled=not available_model_options,
        )
    
    sample_options = [
        "No example",
        "REAL: U.S. House passes landmark infrastructure investment bill",
        "FAKE: Secret alien treaty signed in underground bunker according to leaked document",
        "REAL: NASA rover discovers organic molecules on Mars surface",
        "FAKE: Pope Francis endorses Donald Trump for President shocking Vatican officials"
    ]
    with sample_column:
        sample_select = st.selectbox("Example article", sample_options)
    
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

    title_input = st.text_input("Headline", value=default_title, placeholder="Enter a headline")
    text_input = st.text_area("Article text", value=default_text, height=190, placeholder="Paste the article text")
    
    if st.button("Analyze article", type="primary", disabled=not available_model_options):
        if not title_input.strip() and not text_input.strip():
            st.error("Please enter a title or text to analyze.")
        else:
            combined = title_input + " " + text_input
            with st.spinner("Analyzing text and evaluating linguistic patterns..."):
                prediction = None
                confidence = None
                model_used = model_choice
                
                if model_choice == "Logistic Regression":
                    tfidf, lr, _ = load_classical_models()
                    if tfidf is not None and lr is not None:
                        clean = clean_text(combined)
                        features = tfidf.transform([clean])
                        prob = lr.predict_proba(features)[0]
                        pred_idx = np.argmax(prob)
                        prediction = "REAL NEWS" if pred_idx == 1 else "FAKE NEWS"
                        confidence = prob[pred_idx]
                    else:
                        st.warning("Logistic Regression model checkpoint not found. Please train models first.")
                        
                elif model_choice == "Linear SVM":
                    tfidf, _, svm = load_classical_models()
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
                        
                elif model_choice == "BiLSTM":
                    vocab, lstm_model = load_lstm_model()
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
                    bert_tokenizer, bert_model = load_bert_model()
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
                    result_class = "real" if prediction == "REAL NEWS" else "fake"
                    st.markdown(
                        f'<div class="result-panel"><div class="result-label">Prediction</div>'
                        f'<p class="result-value {result_class}">{prediction}</p>'
                        f'<div class="result-meta">{model_used} · {confidence * 100:.1f}% confidence</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(float(confidence))
                    with st.expander("Cleaned text preview"):
                        st.text(clean_text(combined)[:500])
                    st.caption("A model prediction is not independent fact-checking.")

# PAGE 2: METRICS TABLE
elif sidebar_option == "Results":
    st.subheader("Model results")
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
        
    st.divider()
    st.markdown("### Confusion matrices")
    cm_files = [f for f in os.listdir(CM_DIR) if f.endswith(".png")] if os.path.exists(CM_DIR) else []
    if cm_files:
        cols = st.columns(min(len(cm_files), 2))
        for idx, cf in enumerate(cm_files):
            col = cols[idx % 2]
            with col:
                st.image(os.path.join(CM_DIR, cf), caption=cf.replace("_cm.png", "").upper() + " confusion matrix", width="stretch")
    else:
        st.write("Confusion matrix plots will appear here after model evaluation.")

# PAGE 3: VISUALS
elif sidebar_option == "Visuals":
    st.subheader("Visuals")
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
            st.image(path, width="stretch")
            st.divider()

# PAGE 4: ABOUT
else:
    st.subheader("About")
    st.markdown("""
    Three locally trained classifiers are available for comparison:

    | Model | Input representation |
    | --- | --- |
    | Logistic Regression | TF-IDF features |
    | Linear SVM | TF-IDF features |
    | Bidirectional LSTM | Learned word embeddings |

    Predictions reflect patterns in training data and are not independent fact-checking.
    """)
