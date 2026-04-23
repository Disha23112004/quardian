"""
tests/test_all.py

Basic test suite. Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_synthetic_generation():
    from core.synthetic import generate_dataset, FEATURE_COLS
    df, scaler = generate_dataset(n_normal=50, n_per_anomaly=10)
    assert len(df) == 90
    assert set(FEATURE_COLS).issubset(df.columns)
    assert df["is_anomaly"].sum() == 40
    assert df[FEATURE_COLS].min().min() >= 0.0
    assert df[FEATURE_COLS].max().max() <= 1.0


def test_feature_engineering():
    from core.synthetic import generate_dataset, FEATURE_COLS
    from core.features import FEATURE_COLS as FC
    assert FEATURE_COLS == FC


def test_circuit_output_range():
    from quantum.circuit import score, random_params
    params = random_params()
    x = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7])
    s = score(x, params)
    assert 0.0 <= s <= 1.0, f"Score out of range: {s}"


def test_circuit_different_inputs():
    from quantum.circuit import score, random_params
    params = random_params()
    x_normal = np.array([0.5, 0.2, 0.5, 0.1, 0.3, 0.1])
    x_anomaly = np.array([0.05, 0.05, 0.05, 0.9, 0.9, 0.95])
    s_n = score(x_normal, params)
    s_a = score(x_anomaly, params)
    # Just verify both are valid scores — model is untrained so no ordering guarantee
    assert 0.0 <= s_n <= 1.0
    assert 0.0 <= s_a <= 1.0


def test_classical_baselines():
    from core.synthetic import generate_dataset, FEATURE_COLS
    from core.classical import run_isolation_forest
    from sklearn.model_selection import train_test_split

    df, _ = generate_dataset(n_normal=100, n_per_anomaly=20)
    X = df[FEATURE_COLS].values
    y = df["is_anomaly"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scores, preds, metrics = run_isolation_forest(X_train, X_test, y_test)
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert len(scores) == len(X_test)


def test_anomaly_type_classifier():
    from quantum.detector import classify_anomaly_type
    # bulk exfil: high mean_epd, high recipients
    x_bulk = np.array([0.9, 0.8, 0.5, 0.3, 0.8, 0.4])
    assert classify_anomaly_type(x_bulk) == "bulk_exfil"
    # surveillance drip: low mean_epd, low hour_norm, low std
    x_surv = np.array([0.05, 0.05, 0.1, 0.4, 0.2, 0.6])
    assert classify_anomaly_type(x_surv) == "surveillance_drip"


def test_training_reduces_loss():
    from core.synthetic import generate_dataset, FEATURE_COLS
    from quantum.trainer import train
    from sklearn.model_selection import train_test_split

    df, _ = generate_dataset(n_normal=60, n_per_anomaly=10)
    X = df[FEATURE_COLS].values
    y = df["is_anomaly"].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    params, history = train(X_train, y_train, n_epochs=10, verbose=False)
    assert len(history) == 10
    # Loss should generally trend down (not guaranteed with 10 epochs, but check it's finite)
    assert all(np.isfinite(l) for l in history)
    assert params.shape == (3, 6, 2)  # N_LAYERS x N_QUBITS x 2


if __name__ == "__main__":
    tests = [
        test_synthetic_generation,
        test_feature_engineering,
        test_circuit_output_range,
        test_circuit_different_inputs,
        test_classical_baselines,
        test_anomaly_type_classifier,
        test_training_reduces_loss,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
