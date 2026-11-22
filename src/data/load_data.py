"""Load data DSCO 2025 (deteksi URL phishing)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

TARGET = "ClassLabel"      # 1 = legitimate, 0 = phishing
ID = "ID"
PROVIDED = ["url_length", "has_ip_address", "dot_count", "https_flag", "url_entropy", "token_count", "subdomain_count",
            "query_param_count", "tld_length", "path_length", "has_hyphen_in_domain", "number_of_digits", "tld_popularity",
            "suspicious_file_extension", "domain_name_length", "percentage_numeric_chars"]


def find_data_dir(local: str | Path = "../data/raw") -> Path:
    """Folder yang berisi train.csv kompetisi ini (di /kaggle/input, ikuti symlink)."""
    for root, _, files in os.walk("/kaggle/input", followlinks=True):
        if {"train.csv", "test.csv", "sample_submission.csv"} <= set(files):
            if "URL" in pd.read_csv(Path(root) / "train.csv", nrows=1).columns:
                return Path(root)
    return Path(local)


def load(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir)
    train = pd.read_csv(d / "train.csv")
    train[TARGET] = train[TARGET].astype(int)
    return train, pd.read_csv(d / "test.csv"), pd.read_csv(d / "sample_submission.csv")
