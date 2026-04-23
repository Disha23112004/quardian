# Quardian

**Quantum-assisted communication metadata anomaly detector — runs entirely on your machine.**

No message content is ever read. No data leaves your device. The quantum circuit runs locally via PennyLane.

---

## What it does

Quardian collects metadata from your email (timestamps, sender domains, frequency, thread depth) and models your normal communication pattern using a variational quantum circuit (VQC). When something deviates — unusual contact graph changes, off-hours activity, sudden volume spikes — it flags it.

Four threat patterns are detected:

- **Surveillance drip** — low volume, off-hours, many new sender domains. Consistent with passive monitoring or slow credential probing.
- **Bulk exfiltration** — sudden volume spike, high recipient count. Consistent with mass forwarding or data exfiltration.
- **Timing shift** — consistent activity at unusual hours (e.g. 2–4 AM). May indicate a compromised account being used while the owner sleeps.
- **Contact graph burn** — extreme proportion of new sender domains. May indicate address book harvesting or a phishing campaign.

---

## Results

Benchmarked on synthetic surveillance pattern data (600 samples, 4 anomaly types):

| Model | ROC-AUC | Avg Precision | F1 |
|---|---|---|---|
| **VQC — Amplitude Encoding** | **1.000** | **1.000** | **0.710** |
| VQC — Angle Encoding | 0.998 | 0.996 | 0.689 |
| Isolation Forest | 0.982 | 0.961 | 0.367 |
| Local Outlier Factor | 0.303 | 0.253 | 0.046 |

The quantum model achieves more than 2× the F1 of the best classical baseline. Amplitude encoding outperforms angle encoding — global feature correlations characteristic of surveillance patterns are better captured by embedding the full feature vector into the quantum state simultaneously.

Also validated on 1,322 real emails (46 weekly windows): 0 anomalies detected in a normal inbox, with a detectable anomaly score increase during a period of genuine behavioural change (new environment, new contacts), demonstrating sensitivity without over-triggering.

---

## Architecture

- **6 features per window:** mean emails/day, std emails/day, normalised send hour, weekend fraction, mean recipients, new sender domain ratio
- **Angle encoding:** RY(π·xᵢ) on qubit i — one qubit per feature, 6 qubits total
- **Amplitude encoding:** full feature vector embedded as quantum state amplitudes across 3 qubits (2³ = 8 amplitudes, zero-padded)
- **Variational layers:** 3 layers of RY + RZ rotations with CNOT ring entanglement
- **Optimiser:** Adam (PennyLane), binary cross-entropy loss, 80 epochs
- **Measurement:** ⟨Z⟩ on qubit 0, mapped to [0,1] anomaly score

---

## Project structure

```
quardian/
├── core/
│   ├── collector.py        # Email metadata extraction (IMAP, headers only)
│   ├── features.py         # Feature engineering from raw metadata
│   ├── classical.py        # Classical baselines: Isolation Forest, LOF
│   └── synthetic.py        # Synthetic anomaly dataset generator
├── quantum/
│   ├── circuit.py          # VQC: angle + amplitude encoding
│   ├── trainer.py          # Training loop (Adam, BCE loss)
│   ├── detector.py         # Inference + anomaly scoring
│   └── compare_encodings.py # Encoding comparison experiment
├── dashboard/
│   ├── app.py              # Flask local dashboard
│   └── templates/index.html
├── tests/
│   └── test_all.py
├── main.py                 # CLI entry point
├── config.py               # Configuration
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Run on synthetic data (no email needed)
python main.py --mode demo

# Launch dashboard
python main.py --mode dashboard
# Open http://localhost:5050

# Run encoding comparison experiment
python quantum/compare_encodings.py

# Use your own email (Gmail)
# Set QUARDIAN_EMAIL, QUARDIAN_PASSWORD (app password), QUARDIAN_IMAP as env variables
python main.py --mode collect
python main.py --mode train
python main.py --mode monitor
```

---

## Privacy guarantees

- Only email headers are read: From, To, Date, Subject-length, Thread-ID. No body content ever.
- All data stored locally in `data/` which is gitignored.
- The quantum circuit runs on PennyLane's local simulator — no cloud calls.
- The dashboard runs on localhost only.

---

## Research context

Existing VQC anomaly detection research focuses on enterprise network intrusion detection (NSL-KDD, UNSW-NB15 datasets). Quardian is the first application of VQC-based anomaly detection to personal communication metadata for individual privacy protection — a distinct problem domain motivated by the threat to ordinary people rather than organisations.

---

## Author

Disha Parasu — B.Tech CSE (AI & ML), VIT Chennai

github.com/Disha23112004 | linkedin.com/in/disha-parasu
