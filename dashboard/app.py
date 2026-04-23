"""
dashboard/app.py

Local Flask dashboard. Runs on localhost:5050 only.
Displays anomaly scores, flagged windows, and a comparison of quantum vs classical.
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

app = Flask(__name__)


def load_results():
    results_path = os.path.join(config.DATA_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


@app.route("/")
def index():
    results = load_results()
    return render_template("index.html", has_results=results is not None)


@app.route("/api/results")
def api_results():
    results = load_results()
    if results is None:
        return jsonify({"error": "No results yet. Run main.py --mode demo first."}), 404
    return jsonify(results)


@app.route("/api/windows")
def api_windows():
    windows_path = os.path.join(config.DATA_DIR, "detections.csv")
    if not os.path.exists(windows_path):
        return jsonify({"error": "No detection data found."}), 404
    df = pd.read_csv(windows_path)
    df = df.replace({np.nan: None})
    return jsonify(df.to_dict(orient="records"))


def run():
    print("Starting Quardian dashboard at http://localhost:5050")
    print("Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    run()
