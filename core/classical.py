"""
core/classical.py

Classical anomaly detection baselines for benchmarking against the quantum model.
Implements Isolation Forest and Local Outlier Factor.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from core.synthetic import FEATURE_COLS


def run_isolation_forest(X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    clf = IsolationForest(
        contamination=config.CONTAMINATION,
        n_estimators=200,
        random_state=42
    )
    clf.fit(X_train)
    raw_scores = clf.score_samples(X_test)
    # Convert: more negative = more anomalous → flip and normalise to [0,1]
    scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    preds = (scores > config.ANOMALY_THRESHOLD).astype(int)
    return scores, preds, _metrics(y_test, scores, preds, "Isolation Forest")


def run_lof(X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    clf = LocalOutlierFactor(
        contamination=config.CONTAMINATION,
        novelty=True,
        n_neighbors=20
    )
    clf.fit(X_train)
    raw_scores = clf.score_samples(X_test)
    scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    preds = (scores > config.ANOMALY_THRESHOLD).astype(int)
    return scores, preds, _metrics(y_test, scores, preds, "Local Outlier Factor")


def _metrics(y_true, scores, preds, name):
    try:
        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)
    except Exception:
        auc, ap = 0.0, 0.0

    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, target_names=["normal", "anomaly"], output_dict=True)

    return {
        "model": name,
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "precision": round(report["anomaly"]["precision"], 4),
        "recall": round(report["anomaly"]["recall"], 4),
        "f1": round(report["anomaly"]["f1-score"], 4),
        "confusion_matrix": cm.tolist(),
    }


def benchmark(df: pd.DataFrame):
    """
    Run both classical models on the synthetic dataset.
    Returns a dict of results.
    """
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_COLS].values
    y = df["is_anomaly"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - config.TRAIN_SPLIT, random_state=42, stratify=y
    )

    _, _, if_metrics = run_isolation_forest(X_train, X_test, y_test)
    _, _, lof_metrics = run_lof(X_train, X_test, y_test)

    print("\n=== Classical Baseline Results ===")
    for m in [if_metrics, lof_metrics]:
        print(f"\n{m['model']}")
        print(f"  ROC-AUC:       {m['roc_auc']}")
        print(f"  Avg Precision: {m['avg_precision']}")
        print(f"  Precision:     {m['precision']}")
        print(f"  Recall:        {m['recall']}")
        print(f"  F1:            {m['f1']}")

    return {
        "isolation_forest": if_metrics,
        "lof": lof_metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    from core.synthetic import generate_dataset
    df, _ = generate_dataset()
    benchmark(df)
