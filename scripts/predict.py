#!/usr/bin/env python3
"""Simple CLI to load trained pipeline and predict a single text input."""
from __future__ import annotations

import argparse
import os
import sys

import joblib


def main() -> None:
    p = argparse.ArgumentParser(description="Predict spam/ham for input text")
    p.add_argument("text", nargs="?", help="Text to classify (if omitted, read from stdin)")
    p.add_argument("--model", default="models/pipeline.joblib", help="Path to pipeline artifact")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"Model artifact not found at {args.model}", file=sys.stderr)
        sys.exit(2)

    pipeline = joblib.load(args.model)

    if args.text:
        inp = args.text
    else:
        inp = sys.stdin.read().strip()

    if not inp:
        print("No input text provided", file=sys.stderr)
        sys.exit(3)

    pred = pipeline.predict([inp])[0]
    print(pred)


if __name__ == "__main__":
    main()
