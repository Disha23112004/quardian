"""
quantum/detector.py

Real-time anomaly detection on live or new metadata.
Loads a trained model and scores incoming feature windows.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from quantum.circuit import score as quantum_score
from quantum.trainer import load_params
from core.synthetic import FEATURE_COLS


ANOMALY_DESCRIPTIONS = {
    "surveillance_drip": (
        "Low-volume, off-hours contact pattern with many new sender domains. "
        "Consistent with passive monitoring or slow credential probing."
    ),
    "bulk_exfil": (
        "Sudden spike in email volume and recipients. "
        "Consistent with mass forwarding or data exfiltration."
    ),
    "timing_shift": (
        "Consistent activity at unusual hours (e.g. 2–4 AM). "
        "May indicate compromised account being used while owner sleeps."
    ),
    "contact_graph_burn": (
        "Extremely high proportion of new sender domains in this window. "
        "May indicate address book harvesting or phishing campaign."
    ),
    "normal": "No anomalies detected.",
}


def classify_anomaly_type(x: np.ndarray) -> str:
    """
    Heuristic classifier to name the anomaly type from feature values.
    Used to give human-readable alerts.
    """
    mean_epd, std_epd, mean_hour_norm, weekend_frac, mean_recipients, new_domain_ratio = x

    if mean_epd > 0.75 and mean_recipients > 0.6:
        return "bulk_exfil"
    if mean_hour_norm < 0.2 and std_epd < 0.2:
        return "surveillance_drip"
    if mean_hour_norm < 0.2:
        return "timing_shift"
    if new_domain_ratio > 0.75:
        return "contact_graph_burn"
    return "normal"


class QuardianDetector:
    def __init__(self, model_path: str = config.MODEL_PATH, threshold: float = config.ANOMALY_THRESHOLD):
        self.params = load_params(model_path)
        self.threshold = threshold
        print(f"Detector loaded from {model_path}")

    def score_window(self, x: np.ndarray) -> dict:
        """Score a single feature window."""
        s = quantum_score(x, self.params)
        is_anomaly = s > self.threshold
        anomaly_type = classify_anomaly_type(x) if is_anomaly else "normal"
        return {
            "anomaly_score": round(float(s), 4),
            "is_anomaly": bool(is_anomaly),
            "anomaly_type": anomaly_type,
            "description": ANOMALY_DESCRIPTIONS.get(anomaly_type, ""),
        }

    def score_dataframe(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Score a full feature DataFrame (one window per row)."""
        results = []
        X = feat_df[FEATURE_COLS].values
        for i, x in enumerate(X):
            r = self.score_window(x)
            r["window_start"] = feat_df.iloc[i].get("window_start", i)
            r["window_end"] = feat_df.iloc[i].get("window_end", i)
            r["email_count"] = feat_df.iloc[i].get("email_count", 0)
            results.append(r)
        return pd.DataFrame(results)

    def monitor(self, features_path: str = config.FEATURES_PATH) -> pd.DataFrame:
        """Load saved feature file and run detection."""
        feat_df = pd.read_csv(features_path)
        results = self.score_dataframe(feat_df)
        n_anomalies = results["is_anomaly"].sum()
        print(f"\nScanned {len(results)} windows → {n_anomalies} anomalies detected")
        if n_anomalies > 0:
            print("\n--- Flagged windows ---")
            flagged = results[results["is_anomaly"]]
            for _, row in flagged.iterrows():
                print(f"  [{row['window_start']} – {row['window_end']}] "
                      f"Score: {row['anomaly_score']} | {row['anomaly_type']}")
                print(f"    {row['description']}")
        return results


if __name__ == "__main__":
    detector = QuardianDetector()
    if os.path.exists(config.FEATURES_PATH):
        detector.monitor()
    else:
        print("No features file found. Run main.py --mode demo first.")
