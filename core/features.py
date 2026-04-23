"""
core/features.py

Transforms raw metadata into a fixed-size, normalized feature vector per time window.
Each window represents one week of communication behaviour.

Features per window:
  0: mean emails per day
  1: std emails per day
  2: mean hour of day (normalised)
  3: fraction sent on weekends
  4: mean recipient count
  5: new contact domain ratio (contacts not seen in prior window)

These 6 features map directly to 6 qubits in the quantum circuit.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def engineer_features(df: pd.DataFrame, window_days: int = config.FEATURE_WINDOW_DAYS) -> pd.DataFrame:
    """
    Slice the metadata into rolling windows and compute features for each.
    Returns a DataFrame where each row is one window's feature vector.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date

    min_date = df["date"].min()
    max_date = df["date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D")

    windows = []
    known_domains = set()

    for i in range(0, len(all_dates) - window_days + 1, window_days):
        start = all_dates[i].date()
        end = all_dates[i + window_days - 1].date()

        mask = (df["date"] >= start) & (df["date"] <= end)
        window = df[mask]

        if len(window) < 3:
            continue

        # emails per day
        emails_per_day = window.groupby("date").size()
        mean_epd = emails_per_day.mean()
        std_epd = emails_per_day.std() if len(emails_per_day) > 1 else 0.0

        # timing
        mean_hour = window["hour"].mean() / 23.0
        weekend_fraction = window["weekday"].apply(lambda d: d >= 5).mean()

        # recipients
        mean_recipients = window["recipient_count"].mean()

        # new domain ratio
        current_domains = set(window["sender_domain"].unique())
        if known_domains:
            new_domains = current_domains - known_domains
            new_domain_ratio = len(new_domains) / max(1, len(current_domains))
        else:
            new_domain_ratio = 0.0
        known_domains.update(current_domains)

        windows.append({
            "window_start": str(start),
            "window_end": str(end),
            "email_count": len(window),
            "mean_epd": mean_epd,
            "std_epd": std_epd,
            "mean_hour_norm": mean_hour,
            "weekend_fraction": weekend_fraction,
            "mean_recipients": mean_recipients,
            "new_domain_ratio": new_domain_ratio,
        })

    feat_df = pd.DataFrame(windows)
    return feat_df


FEATURE_COLS = [
    "mean_epd",
    "std_epd",
    "mean_hour_norm",
    "weekend_fraction",
    "mean_recipients",
    "new_domain_ratio",
]


def normalize_features(feat_df: pd.DataFrame, scaler: MinMaxScaler = None):
    """
    Normalize feature columns to [0, 1] for angle encoding into qubits.
    Returns (normalized_array, fitted_scaler).
    """
    X = feat_df[FEATURE_COLS].values
    if scaler is None:
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)
    else:
        X_norm = scaler.transform(X)
    return X_norm, scaler


def load_and_prepare(path: str = config.METADATA_PATH):
    df = pd.read_csv(path)
    feat_df = engineer_features(df)
    X, scaler = normalize_features(feat_df)
    feat_df[FEATURE_COLS] = X
    feat_df.to_csv(config.FEATURES_PATH, index=False)
    print(f"Engineered {len(feat_df)} feature windows, saved to {config.FEATURES_PATH}")
    return feat_df, scaler


if __name__ == "__main__":
    load_and_prepare()
