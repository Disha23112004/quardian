"""
core/synthetic.py

Generates a labelled synthetic dataset for training and benchmarking.
Normal behaviour is drawn from realistic distributions.
Anomalies simulate four threat patterns:

  1. Surveillance drip  — very low volume, unusual hours, many new domains
  2. Bulk exfil spike   — sudden volume spike, high recipient count
  3. Timing shift       — consistent off-hours activity (e.g. 2–4 AM)
  4. Contact graph burn — extreme new_domain_ratio, low volume
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

RNG = np.random.default_rng(42)


def generate_normal(n: int) -> np.ndarray:
    return np.column_stack([
        RNG.normal(25, 8, n).clip(1, 80),       # mean_epd
        RNG.normal(5, 2, n).clip(0, 20),         # std_epd
        RNG.normal(0.5, 0.1, n).clip(0, 1),      # mean_hour_norm (midday-ish)
        RNG.beta(2, 8, n),                        # weekend_fraction (low)
        RNG.normal(1.5, 0.5, n).clip(1, 10),     # mean_recipients
        RNG.beta(1, 8, n),                        # new_domain_ratio (low)
    ])


def generate_surveillance_drip(n: int) -> np.ndarray:
    """Low volume, off-hours, many new domains."""
    return np.column_stack([
        RNG.normal(3, 1, n).clip(1, 8),          # mean_epd — very low
        RNG.normal(0.5, 0.3, n).clip(0, 3),      # std_epd — very stable
        RNG.normal(0.12, 0.05, n).clip(0, 0.25), # mean_hour_norm — ~3 AM
        RNG.beta(1, 2, n),                        # weekend_fraction — slightly elevated
        RNG.normal(1.1, 0.2, n).clip(1, 3),      # mean_recipients — always 1
        RNG.beta(8, 2, n),                        # new_domain_ratio — very high
    ])


def generate_bulk_exfil(n: int) -> np.ndarray:
    """Sudden volume spike, many recipients."""
    return np.column_stack([
        RNG.normal(90, 20, n).clip(60, 200),     # mean_epd — huge spike
        RNG.normal(30, 10, n).clip(10, 80),      # std_epd — high variance
        RNG.normal(0.5, 0.15, n).clip(0, 1),     # mean_hour_norm — normal hours
        RNG.beta(2, 5, n),                        # weekend_fraction
        RNG.normal(8, 2, n).clip(3, 20),         # mean_recipients — many
        RNG.beta(3, 3, n),                        # new_domain_ratio — moderate
    ])


def generate_timing_shift(n: int) -> np.ndarray:
    """Consistent 2–4 AM activity."""
    return np.column_stack([
        RNG.normal(20, 5, n).clip(5, 40),        # mean_epd — normal-ish
        RNG.normal(4, 1, n).clip(1, 10),         # std_epd
        RNG.normal(0.12, 0.03, n).clip(0, 0.2), # mean_hour_norm — 3 AM
        RNG.beta(4, 4, n),                        # weekend_fraction — elevated
        RNG.normal(1.8, 0.4, n).clip(1, 5),     # mean_recipients
        RNG.beta(2, 5, n),                        # new_domain_ratio
    ])


def generate_contact_graph_burn(n: int) -> np.ndarray:
    """Extreme new contact ratio — someone probing or compromised address book."""
    return np.column_stack([
        RNG.normal(10, 3, n).clip(2, 25),        # mean_epd — lowish
        RNG.normal(3, 1, n).clip(0.5, 8),        # std_epd
        RNG.normal(0.45, 0.1, n).clip(0, 1),     # mean_hour_norm
        RNG.beta(2, 6, n),                        # weekend_fraction
        RNG.normal(1.2, 0.3, n).clip(1, 4),     # mean_recipients
        RNG.beta(9, 1, n),                        # new_domain_ratio — almost all new
    ])


FEATURE_COLS = [
    "mean_epd", "std_epd", "mean_hour_norm",
    "weekend_fraction", "mean_recipients", "new_domain_ratio"
]

ANOMALY_TYPES = {
    0: "normal",
    1: "surveillance_drip",
    2: "bulk_exfil",
    3: "timing_shift",
    4: "contact_graph_burn",
}


def generate_dataset(n_normal: int = 400, n_per_anomaly: int = 50) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler

    normal = generate_normal(n_normal)
    surv = generate_surveillance_drip(n_per_anomaly)
    bulk = generate_bulk_exfil(n_per_anomaly)
    timing = generate_timing_shift(n_per_anomaly)
    burn = generate_contact_graph_burn(n_per_anomaly)

    X = np.vstack([normal, surv, bulk, timing, burn])
    labels = np.array(
        [0] * n_normal +
        [1] * n_per_anomaly +
        [2] * n_per_anomaly +
        [3] * n_per_anomaly +
        [4] * n_per_anomaly
    )
    is_anomaly = (labels > 0).astype(int)

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    df = pd.DataFrame(X_norm, columns=FEATURE_COLS)
    df["anomaly_type"] = labels
    df["anomaly_type_name"] = [ANOMALY_TYPES[l] for l in labels]
    df["is_anomaly"] = is_anomaly

    shuffle_idx = RNG.permutation(len(df))
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    return df, scaler


def generate_and_save():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df, scaler = generate_dataset()
    df.to_csv(config.SYNTHETIC_PATH, index=False)
    print(f"Generated {len(df)} samples ({df['is_anomaly'].sum()} anomalies) → {config.SYNTHETIC_PATH}")
    return df, scaler


if __name__ == "__main__":
    generate_and_save()
