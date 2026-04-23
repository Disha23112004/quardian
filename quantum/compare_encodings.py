"""
quantum/compare_encodings.py

Compares angle encoding vs amplitude encoding on the synthetic dataset.
This is the core experiment that answers the research question:
does the choice of encoding strategy affect anomaly detection performance?

Run with:
  python quantum/compare_encodings.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pennylane import numpy as pnp
import pennylane as qml

import config
from core.synthetic import generate_dataset, FEATURE_COLS
from quantum.circuit import (
    circuit, score,
    circuit_amplitude, score_amplitude,
    random_params, random_params_amplitude,
    N_QUBITS, N_QUBITS_AMP, N_LAYERS,
)


def train_encoding(X_train, y_train, circuit_fn, score_fn,
                   params_init, n_qubits, n_epochs=config.N_EPOCHS,
                   lr=config.LEARNING_RATE, label=""):

    params = pnp.array(params_init, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)

    # Semi-supervised: use all labelled samples
    X_fit = X_train
    y_fit = y_train.astype(float)
    n = len(X_fit)

    print(f"\nTraining [{label}] — {n_epochs} epochs, {n_qubits} qubits")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        idx = np.random.permutation(n)
        for i in idx:
            x_i = pnp.array(X_fit[i], requires_grad=False)
            y_i = float(y_fit[i])

            def cost(p):
                raw = circuit_fn(x_i, p)
                pred = pnp.clip((1.0 - raw) / 2.0, 1e-7, 1 - 1e-7)
                return -(y_i * pnp.log(pred) + (1 - y_i) * pnp.log(1 - pred))

            params, loss_val = opt.step_and_cost(cost, params)
            epoch_loss += float(loss_val)

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {epoch_loss/n:.4f}")

    return np.array(params)


def evaluate_encoding(score_fn, params, X_test, y_test,
                      threshold=config.ANOMALY_THRESHOLD, label=""):
    scores = np.array([score_fn(x, params) for x in X_test])
    preds  = (scores > threshold).astype(int)

    try:
        auc = roc_auc_score(y_test, scores)
        ap  = average_precision_score(y_test, scores)
    except Exception:
        auc, ap = 0.0, 0.0

    f1 = f1_score(y_test, preds, zero_division=0)

    return {"model": label, "roc_auc": round(auc, 4),
            "avg_precision": round(ap, 4), "f1": round(f1, 4),
            "scores": scores, "predictions": preds}


def run():
    print("=" * 60)
    print("ENCODING COMPARISON: Angle vs Amplitude")
    print("=" * 60)

    df, _ = generate_dataset(n_normal=400, n_per_anomaly=50)
    X = df[FEATURE_COLS].values
    y = df["is_anomaly"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - config.TRAIN_SPLIT,
        random_state=42, stratify=y
    )

    # --- Angle encoding ---
    params_angle = train_encoding(
        X_train, y_train,
        circuit_fn=circuit,
        score_fn=score,
        params_init=random_params(),
        n_qubits=N_QUBITS,
        label="Angle Encoding"
    )
    angle_results = evaluate_encoding(
        score, params_angle, X_test, y_test, label="Angle Encoding"
    )

    # --- Amplitude encoding ---
    params_amp = train_encoding(
        X_train, y_train,
        circuit_fn=circuit_amplitude,
        score_fn=score_amplitude,
        params_init=random_params_amplitude(),
        n_qubits=N_QUBITS_AMP,
        label="Amplitude Encoding"
    )
    amp_results = evaluate_encoding(
        score_amplitude, params_amp, X_test, y_test, label="Amplitude Encoding"
    )

    # --- Print comparison ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Model':<25} {'ROC-AUC':>8} {'Avg Prec':>9} {'F1':>8}")
    print("-" * 53)
    for r in [angle_results, amp_results]:
        print(f"{r['model']:<25} {r['roc_auc']:>8} {r['avg_precision']:>9} {r['f1']:>8}")

    # --- Interpretation ---
    print("\n--- Interpretation ---")
    if angle_results["roc_auc"] > amp_results["roc_auc"]:
        print("Angle encoding outperforms amplitude encoding on this dataset.")
        print("Likely reason: angle encoding preserves per-feature locality,")
        print("which suits the tabular, independent nature of metadata features.")
    elif amp_results["roc_auc"] > angle_results["roc_auc"]:
        print("Amplitude encoding outperforms angle encoding on this dataset.")
        print("Likely reason: amplitude encoding captures global feature")
        print("correlations more efficiently via the quantum state space.")
    else:
        print("Both encodings perform similarly — the variational layers")
        print("are likely the dominant factor over encoding choice.")

    # Save results
    import json
    results = {
        "angle_encoding": {k: v for k, v in angle_results.items()
                           if k not in ("scores", "predictions")},
        "amplitude_encoding": {k: v for k, v in amp_results.items()
                               if k not in ("scores", "predictions")},
    }
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(os.path.join(config.DATA_DIR, "encoding_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {config.DATA_DIR}/encoding_comparison.json")

    return results


if __name__ == "__main__":
    run()
