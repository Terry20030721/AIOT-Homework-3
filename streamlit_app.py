#!/usr/bin/env python3
"""Streamlit dashboard for visual analysis of the spam classifier.

Behavior:
- On startup, automatically loads `datasets/processed/cleaned.csv`.
- Splits data (80/20) deterministically and retrains an SVM pipeline (with calibration) for analysis.
- Renders three charts immediately: Data Overview, Top Tokens by Class, Model Performance (ROC & PR).
- If dataset missing, shows a friendly error and usage instructions.
"""
from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


DEFAULT_DATA_PATH = "datasets/processed/cleaned.csv"


@st.cache_data
def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure expected columns
    if not {"label", "text"}.issubset(set(df.columns)):
        raise ValueError("Dataset must contain 'label' and 'text' columns")
    # Coerce text to string and drop rows with empty text
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    # Normalize labels
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    # Keep only known labels 'spam'/'ham'
    df = df[df["label"].isin(["spam", "ham"])].reset_index(drop=True)
    return df


def train_analysis_pipeline(
    texts: pd.Series,
    labels: pd.Series,
    seed: int = 42,
) -> Tuple[Pipeline, pd.Series, pd.Series]:
    """Train a TF-IDF + SVM pipeline (wrapped with calibration) and return fitted pipeline and y_test predictions/probabilities.

    Returns fitted pipeline and raw probabilities for positive class on test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    # Build pipeline where classifier is calibrated to produce probabilities
    base_clf = LinearSVC(random_state=seed, max_iter=10000)
    # CalibratedClassifierCV changed constructor kwarg name across scikit-learn versions
    # Newer versions use 'estimator', older used 'base_estimator'. Detect at runtime.
    try:
        from inspect import signature

        sig = signature(CalibratedClassifierCV.__init__)
        if "estimator" in sig.parameters:
            calibrated = CalibratedClassifierCV(estimator=base_clf, cv=3)
        else:
            calibrated = CalibratedClassifierCV(base_estimator=base_clf, cv=3)
    except Exception:
        # Fallback: try common arg name and let it raise if incompatible
        try:
            calibrated = CalibratedClassifierCV(estimator=base_clf, cv=3)
        except TypeError:
            calibrated = CalibratedClassifierCV(base_estimator=base_clf, cv=3)
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", calibrated)])

    pipeline.fit(X_train, y_train)

    # predict probabilities for positive class 'spam'
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test)[:, 1]
    else:
        # fallback: use decision_function then min-max scale to [0,1]
        df_scores = pipeline.decision_function(X_test)
        probs = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-9)

    return pipeline, y_test.reset_index(drop=True), pd.Series(probs)


def plot_data_overview(df: pd.DataFrame) -> plt.Figure:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", color=["#1f77b4", "#ff7f0e"], ax=ax)
    ax.set_title("Data Overview: spam vs ham")
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    plt.tight_layout()
    return fig

def plot_top_tokens(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    try:
        X = vectorizer.fit_transform(df["text"])
        tokens = np.array(vectorizer.get_feature_names_out())
    except ValueError:
        # empty vocabulary (e.g., too short texts or all filtered tokens)
        for i, label in enumerate(["ham", "spam"]):
            axes[i].text(0.5, 0.5, "No tokens available for plotting", ha="center", va="center")
            axes[i].set_axis_off()
        return fig

    for i, label in enumerate(["ham", "spam"]):
        mask = df["label"] == label
        if mask.sum() == 0:
            axes[i].set_title(f"No samples for {label}")
            axes[i].set_axis_off()
            continue
        class_counts = X[mask.values].sum(axis=0).A1
        if class_counts.sum() == 0:
            axes[i].text(0.5, 0.5, "No tokens available for this class", ha="center", va="center")
            axes[i].set_axis_off()
            continue
        top_idx = np.argsort(class_counts)[-top_n:][::-1]
        top_tokens = tokens[top_idx]
        top_vals = class_counts[top_idx]
        axes[i].barh(range(len(top_tokens))[::-1], top_vals[::-1], color="#4c72b0")
        axes[i].set_yticks(range(len(top_tokens)))
        axes[i].set_yticklabels(top_tokens)
        axes[i].set_title(f"Top {top_n} tokens for {label}")

    plt.tight_layout()
    return fig

def plot_model_performance(y_test: pd.Series, probs: pd.Series) -> plt.Figure:
    # ROC
    fpr, tpr, _ = roc_curve((y_test == "spam").astype(int), probs)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve((y_test == "spam").astype(int), probs)
    ap = average_precision_score((y_test == "spam").astype(int), probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    axes[1].step(recall, precision, color="purple", where="post", label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    return fig

def main() -> None:
    st.set_page_config(page_title="Spam Classifier Dashboard", layout="wide")
    st.title("Spam Classification - Visual Analysis")

    data_path = DEFAULT_DATA_PATH

    if not os.path.exists(data_path):
        st.error(
            f"Dataset not found at {data_path}. Please run preprocessing first. See README for instructions."
        )
        st.markdown("`python scripts/preprocess.py` to download and clean the dataset.")
        return

    try:
        df = load_data(data_path)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    st.sidebar.header("Settings")
    top_n = st.sidebar.slider("Top N tokens", min_value=5, max_value=50, value=20)
    seed = st.sidebar.number_input("Random seed", value=42)

    st.sidebar.markdown("**Note:** The dashboard will automatically train an analysis pipeline on startup.")

    # Show data overview
    st.header("1) Data Overview")
    fig1 = plot_data_overview(df)
    st.pyplot(fig1)

    # Top tokens
    st.header("2) Top Tokens by Class")
    fig2 = plot_top_tokens(df, top_n)
    st.pyplot(fig2)

    # Train analysis pipeline and show model performance
    st.header("3) Model Performance (Test)")
    with st.spinner("Training analysis pipeline..."):
        try:
            pipeline, y_test, probs = train_analysis_pipeline(df["text"], df["label"], seed=seed)
        except Exception as e:
            st.error(f"Training failed: {e}")
            return

    fig3 = plot_model_performance(y_test, probs)
    st.pyplot(fig3)

    # Live inference / test utility
    st.header("4) Live Inference")
    st.markdown(
        "Type text into the box below or generate a random example from the dataset, then press **Predict** to see the predicted label and spam probability."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        input_text = st.text_area("Input text", value="", height=120)
        if input_text.strip() == "":
            st.info("You can paste a message here or generate a random example from the dataset to try live inference.")
    with col2:
        st.write("\n")
        sample_mode = st.selectbox("Example source", ["Random sample (dataset)", "Random spam template", "Random ham template"])
        if st.button("Generate example"):
            if sample_mode == "Random sample (dataset)":
                # pick a random row from df
                sample = df.sample(n=1).iloc[0]
                gen_text = sample["text"]
                gen_label = sample["label"]
            elif sample_mode == "Random spam template":
                templates = [
                    "Free entry in 2 a wkly comp to win FA Cup final tkts",
                    "Win cash now! Text WIN to 12345 to claim your prize",
                    "You have won a lottery. Call 0901234567 to claim",
                ]
                import random

                gen_text = random.choice(templates)
                gen_label = "spam"
            else:
                templates = [
                    "I'll be there in 10 mins, don't worry",
                    "Can we meet tomorrow at 3pm?",
                    "Thanks for the update, will do it later",
                ]
                import random

                gen_text = random.choice(templates)
                gen_label = "ham"

            # populate input_text with generated sample
            st.session_state["_live_input"] = gen_text
            st.session_state["_live_label"] = gen_label

    # prefer session_state value when populated by generator
    live_text = st.session_state.get("_live_input", input_text)
    live_label = st.session_state.get("_live_label")

    if st.button("Predict"):
        if not live_text or live_text.strip() == "":
            st.error("Please provide input text (or generate an example) before predicting.")
        else:
            try:
                pred = pipeline.predict([live_text])[0]
                if hasattr(pipeline, "predict_proba"):
                    prob = pipeline.predict_proba([live_text])[0][1]
                else:
                    # fallback to decision_function scaled to [0,1]
                    try:
                        score = pipeline.decision_function([live_text])[0]
                        # sigmoid-ish scaling
                        prob = 1 / (1 + np.exp(-score))
                    except Exception:
                        prob = float(np.nan)

                st.subheader("Prediction")
                st.write(f"**Predicted label:** {pred}")
                st.write(f"**Spam probability:** {prob:.4f}" if not np.isnan(prob) else "Probability unavailable")
                if live_label is not None:
                    st.write(f"**Ground-truth (sample):** {live_label}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # expose session_state controls for generated example
    if "_live_input" in st.session_state:
        st.markdown("---")
        st.markdown("**Generated example (editable):**")
        st.code(st.session_state["_live_input"])


if __name__ == "__main__":
    main()
