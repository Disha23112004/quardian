"""
quantum/trainer.py

Training loop for the variational quantum circuit.

Loss function: binary cross-entropy between circuit anomaly score and label.
Optimiser: PennyLane's Adam optimiser (parameter-shift compatible).

Training strategy:
  - Train only on NORMAL samples (unsupervised baseline)
  - The circuit learns to output low scores for normal behaviour
  - Anomalies are detected at inference time as high-scoring outliers
  
  Optionally: semi-supervised mode uses a small labelled set to push
  anomaly scores up for known anomaly types.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from quantum.circuit import circuit, random_params, N_QUBITS, N_LAYERS, score


def bce_loss(y_pred: float, y_true: float, eps: float = 1e-7) -> float:
    y_pred = pnp.clip(y_pred, eps, 1 - eps)
    return -(y_true * pnp.log(y_pred) + (1 - y_true) * pnp.log(1 - y_pred))


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = config.N_EPOCHS,
    lr: float = config.LEARNING_RATE,
    seed: int = 42,
    semi_supervised: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    Train the VQC.
    
    Returns (trained_params, loss_history).
    """
    params = pnp.array(random_params(seed), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)

    # In unsupervised mode, only use normal samples for training
    if not semi_supervised:
        mask = y_train == 0
        X_fit = X_train[mask]
        y_fit = np.zeros(mask.sum())
    else:
        X_fit = X_train
        y_fit = y_train.astype(float)

    loss_history = []
    n = len(X_fit)

    if verbose:
        print(f"\nTraining VQC: {N_LAYERS} layers, {N_QUBITS} qubits, {n_epochs} epochs")
        print(f"Training samples: {n} ({'semi-supervised' if semi_supervised else 'unsupervised'})")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        idx = np.random.permutation(n)

        for i in idx:
            x_i = pnp.array(X_fit[i], requires_grad=False)
            y_i = float(y_fit[i])

            def cost(p):
                raw = circuit(x_i, p)
                pred = (1.0 - raw) / 2.0
                return bce_loss(pred, y_i)

            params, loss_val = opt.step_and_cost(cost, params)
            epoch_loss += float(loss_val)

        avg_loss = epoch_loss / n
        loss_history.append(avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f}")

    return np.array(params), loss_history


def evaluate(
    params: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = config.ANOMALY_THRESHOLD,
) -> dict:
    """Score all test samples and compute metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

    scores = np.array([score(x, params) for x in X_test])
    preds = (scores > threshold).astype(int)

    try:
        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
    except Exception:
        auc, ap = 0.0, 0.0

    report = classification_report(y_test, preds, target_names=["normal", "anomaly"], output_dict=True)

    return {
        "model": "Quantum VQC",
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "precision": round(report["anomaly"]["precision"], 4),
        "recall": round(report["anomaly"]["recall"], 4),
        "f1": round(report["anomaly"]["f1-score"], 4),
        "scores": scores,
        "predictions": preds,
    }


def save_params(params: np.ndarray, path: str = config.MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, params)
    print(f"Model saved to {path}")


def load_params(path: str = config.MODEL_PATH) -> np.ndarray:
    return np.load(path)


if __name__ == "__main__":
    from core.synthetic import generate_dataset
    from sklearn.model_selection import train_test_split

    df, _ = generate_dataset()
    from core.synthetic import FEATURE_COLS
    X = df[FEATURE_COLS].values
    y = df["is_anomaly"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    params, history = train(X_train, y_train)
    save_params(params)
    metrics = evaluate(params, X_test, y_test)

    print("\n=== Quantum VQC Results ===")
    for k, v in metrics.items():
        if k not in ("scores", "predictions"):
            print(f"  {k}: {v}")
