"""
main.py

Quardian CLI. All modes run locally — no data leaves your machine.

Usage:
  python main.py --mode demo        # Full pipeline on synthetic data (no email needed)
  python main.py --mode collect     # Collect real metadata from your email (configure config.py first)
  python main.py --mode train       # Train quantum model on collected data
  python main.py --mode monitor     # Run anomaly detection on collected data
  python main.py --mode dashboard   # Launch local web dashboard
  python main.py --mode benchmark   # Compare quantum vs classical on synthetic data
"""

import argparse
import os
import json
import numpy as np
import sys

import config


def mode_demo():
    """Full pipeline on synthetic data — no email account needed."""
    print("=" * 60)
    print("QUARDIAN — demo mode (synthetic data)")
    print("=" * 60)

    # 1. Generate synthetic dataset
    print("\n[1/4] Generating synthetic metadata dataset...")
    from core.synthetic import generate_dataset, FEATURE_COLS
    df, scaler = generate_dataset(n_normal=400, n_per_anomaly=50)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(config.SYNTHETIC_PATH, index=False)
    print(f"      {len(df)} samples generated ({df['is_anomaly'].sum()} anomalies)")

    # 2. Train/test split
    from sklearn.model_selection import train_test_split
    X = df[FEATURE_COLS].values
    y = df["is_anomaly"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - config.TRAIN_SPLIT, random_state=42, stratify=y
    )

    # 3. Classical baselines
    print("\n[2/4] Running classical baselines...")
    from core.classical import run_isolation_forest, run_lof
    _, _, if_metrics = run_isolation_forest(X_train, X_test, y_test)
    _, _, lof_metrics = run_lof(X_train, X_test, y_test)

    # 4. Train quantum model
    print("\n[3/4] Training variational quantum circuit...")
    from quantum.trainer import train, evaluate, save_params
    params, loss_history = train(X_train, y_train, semi_supervised=True)
    save_params(params)
    q_metrics = evaluate(params, X_test, y_test)

    # 5. Score all windows for dashboard
    print("\n[4/4] Scoring windows for dashboard...")
    from quantum.detector import QuardianDetector, classify_anomaly_type, ANOMALY_DESCRIPTIONS
    from quantum.circuit import score as quantum_score

    window_records = []
    for i, x in enumerate(X):
        s = quantum_score(x, params)
        is_anomaly = s > config.ANOMALY_THRESHOLD
        anomaly_type = classify_anomaly_type(x) if is_anomaly else "normal"
        window_records.append({
            "window_start": f"W{i:03d}-start",
            "window_end": f"W{i:03d}-end",
            "email_count": int(np.round(x[0] * 30 + 5)),
            "anomaly_score": round(float(s), 4),
            "is_anomaly": bool(is_anomaly),
            "anomaly_type": anomaly_type,
            "description": ANOMALY_DESCRIPTIONS.get(anomaly_type, ""),
            "true_label": int(y[i]),
        })

    import pandas as pd
    detections_df = pd.DataFrame(window_records)
    detections_df.to_csv(os.path.join(config.DATA_DIR, "detections.csv"), index=False)

    # Save results summary
    results = {
        "quantum_vqc": {k: v for k, v in q_metrics.items() if k not in ("scores", "predictions")},
        "isolation_forest": if_metrics,
        "lof": lof_metrics,
        "loss_history": loss_history,
    }
    with open(os.path.join(config.DATA_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    headers = ["Model", "ROC-AUC", "Avg Prec", "Precision", "Recall", "F1"]
    models = [
        ("Quantum VQC", q_metrics),
        ("Isolation Forest", if_metrics),
        ("Local Outlier Factor", lof_metrics),
    ]
    print(f"\n{'Model':<22} {'ROC-AUC':>8} {'Avg Prec':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 70)
    for name, m in models:
        print(f"{name:<22} {m['roc_auc']:>8} {m['avg_precision']:>9} {m['precision']:>10} {m['recall']:>8} {m['f1']:>8}")

    n_detected = detections_df["is_anomaly"].sum()
    print(f"\nWindows scanned: {len(detections_df)} | Anomalies flagged: {n_detected}")
    print(f"\nResults saved to {config.DATA_DIR}/")
    print("Run `python main.py --mode dashboard` to view the web dashboard.")


def mode_collect():
    print("Collecting email metadata (headers only, no body content)...")
    from core.collector import collect_and_save
    collect_and_save()


def mode_train():
    print("Training quantum model on collected data...")
    from core.features import load_and_prepare
    from sklearn.model_selection import train_test_split
    from quantum.trainer import train, evaluate, save_params
    from core.synthetic import FEATURE_COLS

    feat_df, scaler = load_and_prepare()
    X = feat_df[FEATURE_COLS].values
    # In live mode, we have no labels — train unsupervised
    y = np.zeros(len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    params, history = train(X_train, y_train, semi_supervised=False)
    save_params(params)
    print("Training complete.")


def mode_monitor():
    print("Running anomaly detection on collected data...")
    from quantum.detector import QuardianDetector, classify_anomaly_type, ANOMALY_DESCRIPTIONS
    from quantum.circuit import score as quantum_score
    from core.synthetic import FEATURE_COLS
    import pandas as pd
    import json

    detector = QuardianDetector()
    feat_df = pd.read_csv(config.FEATURES_PATH)
    results = detector.score_dataframe(feat_df)

    # Save detections for dashboard
    results.to_csv(os.path.join(config.DATA_DIR, "detections.csv"), index=False)

    # Save a results summary for dashboard
    n_anomalies = results["is_anomaly"].sum()
    summary = {
        "quantum_vqc": {
            "roc_auc": "N/A (live data)",
            "avg_precision": "N/A (live data)",
            "precision": "N/A (live data)",
            "recall": "N/A (live data)",
            "f1": "N/A (live data)",
        },
        "isolation_forest": {},
        "lof": {},
    }
    with open(os.path.join(config.DATA_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nScanned {len(results)} windows → {n_anomalies} anomalies detected")
    if n_anomalies > 0:
        flagged = results[results["is_anomaly"]]
        for _, row in flagged.iterrows():
            print(f"  [{row['window_start']}] Score: {row['anomaly_score']} | {row['anomaly_type']}")
    else:
        print("No anomalies detected — your communication pattern looks normal.")


def mode_dashboard():
    from dashboard.app import run
    run()


def mode_benchmark():
    print("Running full benchmark: quantum vs classical on synthetic data...")
    from core.synthetic import generate_dataset, FEATURE_COLS
    from core.classical import benchmark
    from quantum.trainer import train, evaluate, save_params
    from sklearn.model_selection import train_test_split

    df, _ = generate_dataset()
    classical_results = benchmark(df)
    X_train, X_test = classical_results["X_train"], classical_results["X_test"]
    y_train, y_test = classical_results["y_train"], classical_results["y_test"]

    params, _ = train(X_train, y_train, semi_supervised=True)
    q_metrics = evaluate(params, X_test, y_test)

    print("\n=== Quantum VQC ===")
    for k, v in q_metrics.items():
        if k not in ("scores", "predictions"):
            print(f"  {k}: {v}")


MODES = {
    "demo": mode_demo,
    "collect": mode_collect,
    "train": mode_train,
    "monitor": mode_monitor,
    "dashboard": mode_dashboard,
    "benchmark": mode_benchmark,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quardian — quantum metadata anomaly detector")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="demo",
        help="Which mode to run"
    )
    args = parser.parse_args()
    MODES[args.mode]()
