#!/usr/bin/env python3
"""Train a TF-IDF + SVM pipeline and persist artifacts to models/."""
from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


def main() -> None:
    p = argparse.ArgumentParser(description="Train TF-IDF + SVM pipeline")
    p.add_argument("--data", default="datasets/processed/cleaned.csv", help="Path to cleaned CSV")
    p.add_argument("--out-dir", default="models", help="Models output directory")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LinearSVC(random_state=42)),
        ]
    )

    print("Performing cross-validation (5-fold)...")
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"CV accuracy: mean={scores.mean():.4f} std={scores.std():.4f}")

    print("Fitting pipeline on full data...")
    pipeline.fit(X, y)

    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, "pipeline.joblib")
    joblib.dump(pipeline, dest)
    print(f"Saved pipeline to {dest}")


if __name__ == "__main__":
    main()
