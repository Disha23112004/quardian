"""
quantum/circuit.py

Variational Quantum Circuit for anomaly detection.

Two encoding strategies are implemented and compared:

1. ANGLE ENCODING (original)
   - Each feature x_i → RY(π * x_i) on qubit i
   - Simple, local, one qubit per feature
   - Well suited for low-dimensional normalised vectors

2. AMPLITUDE ENCODING
   - The full feature vector is normalised to unit norm and
     embedded as the amplitudes of the quantum state
   - Exponentially more compact in theory
   - Implemented via qml.AmplitudeEmbedding
   - Requires the feature vector length to be a power of 2
     (we zero-pad from 6 → 8 features, using 3 qubits)

Both use the same variational layers and CNOT ring entanglement.
Results are compared in quantum/compare_encodings.py.

The circuit is intentionally shallow (3 layers) to run fast on a CPU simulator.
"""

import pennylane as qml
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

N_QUBITS = config.N_QUBITS        # 6 — for angle encoding
N_QUBITS_AMP = 3                   # 2^3 = 8 amplitudes, fits 6 features + 2 padding
N_LAYERS = config.N_LAYERS

dev_angle = qml.device("default.qubit", wires=N_QUBITS)
dev_amp   = qml.device("default.qubit", wires=N_QUBITS_AMP)


# ── Angle encoding ────────────────────────────────────────────────────────────

def angle_encoding(x: np.ndarray):
    """Encode a normalised feature vector as RY rotations."""
    for i in range(N_QUBITS):
        qml.RY(np.pi * x[i], wires=i)


def variational_layer(params: np.ndarray, layer_idx: int, n_qubits: int):
    """
    One variational layer:
      - RY and RZ on each qubit with trainable params
      - CNOT ring entanglement
    params shape: (N_LAYERS, n_qubits, 2)
    """
    for i in range(n_qubits):
        qml.RY(params[layer_idx, i, 0], wires=i)
        qml.RZ(params[layer_idx, i, 1], wires=i)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


@qml.qnode(dev_angle, interface="autograd")
def circuit(x: np.ndarray, params: np.ndarray) -> float:
    """Angle-encoded VQC. Returns <Z> on qubit 0."""
    angle_encoding(x)
    for l in range(N_LAYERS):
        variational_layer(params, l, N_QUBITS)
    return qml.expval(qml.PauliZ(0))


def score(x: np.ndarray, params: np.ndarray) -> float:
    """Map angle-encoded circuit output [-1,1] → [0,1] as anomaly score."""
    raw = circuit(x, params)
    return float((1.0 - raw) / 2.0)


def random_params(seed: int = 42, n_qubits: int = N_QUBITS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, (N_LAYERS, n_qubits, 2))


# ── Amplitude encoding ────────────────────────────────────────────────────────

def _pad_to_power_of_2(x: np.ndarray) -> np.ndarray:
    """Zero-pad x to the next power of 2, then normalise to unit norm."""
    n = len(x)
    target = int(2 ** np.ceil(np.log2(n)))
    padded = np.zeros(target)
    padded[:n] = x
    norm = np.linalg.norm(padded)
    if norm < 1e-9:
        padded[0] = 1.0  # avoid zero vector
    else:
        padded = padded / norm
    return padded


@qml.qnode(dev_amp, interface="autograd")
def circuit_amplitude(x: np.ndarray, params: np.ndarray) -> float:
    """
    Amplitude-encoded VQC.
    x is zero-padded to 8 elements and embedded as state amplitudes
    across 3 qubits (2^3 = 8). Returns <Z> on qubit 0.
    """
    x_padded = _pad_to_power_of_2(x)
    qml.AmplitudeEmbedding(x_padded, wires=range(N_QUBITS_AMP), normalize=False)
    for l in range(N_LAYERS):
        variational_layer(params, l, N_QUBITS_AMP)
    return qml.expval(qml.PauliZ(0))


def score_amplitude(x: np.ndarray, params: np.ndarray) -> float:
    """Map amplitude-encoded circuit output [-1,1] → [0,1] as anomaly score."""
    raw = circuit_amplitude(x, params)
    return float((1.0 - raw) / 2.0)


def random_params_amplitude(seed: int = 42) -> np.ndarray:
    return random_params(seed=seed, n_qubits=N_QUBITS_AMP)


# ── Utilities ─────────────────────────────────────────────────────────────────

def draw():
    """Print text diagrams of both circuits."""
    params_a = random_params()
    params_amp = random_params_amplitude()
    x = np.zeros(N_QUBITS)

    print("=== Angle Encoding Circuit ===")
    print(qml.draw(circuit)(x, params_a))
    print("\n=== Amplitude Encoding Circuit ===")
    print(qml.draw(circuit_amplitude)(x[:N_QUBITS], params_amp))


if __name__ == "__main__":
    draw()
    x = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7])
    params_a = random_params()
    params_amp = random_params_amplitude()
    print(f"\nAngle encoding score:     {score(x, params_a):.4f}")
    print(f"Amplitude encoding score: {score_amplitude(x, params_amp):.4f}")
