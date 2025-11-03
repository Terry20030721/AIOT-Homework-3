#!/usr/bin/env python3
"""Download and preprocess the SMS spam dataset.

Creates:
 - datasets/raw/sms_spam_no_header.csv
 - datasets/processed/cleaned.csv (with header: label,text)
"""
from __future__ import annotations

import argparse
import os
import re
import urllib.request

import pandas as pd

URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-"
    "Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def download_if_missing(path: str) -> None:
    if os.path.exists(path):
        print(f"Found existing {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading dataset to {path}...")
    urllib.request.urlretrieve(URL, path)


def clean_text(s: str) -> str:
    s = (s or "").lower()
    # remove non-word characters (keep unicode word chars and spaces)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> None:
    p = argparse.ArgumentParser(description="Download and preprocess SMS spam dataset")
    p.add_argument(
        "--raw",
        default="datasets/raw/sms_spam_no_header.csv",
        help="Path to raw CSV (will be downloaded if missing)",
    )
    p.add_argument(
        "--out",
        default="datasets/processed/cleaned.csv",
        help="Output cleaned CSV path",
    )
    args = p.parse_args()

    download_if_missing(args.raw)

    # Dataset has no header: first column label, second column text
    df = pd.read_csv(args.raw, header=None, encoding="utf-8", names=["label", "text"])
    df["text"] = df["text"].astype(str).apply(clean_text)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote cleaned data to {args.out} (rows={len(df)})")


if __name__ == "__main__":
    main()
