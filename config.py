import os
from dotenv import load_dotenv

load_dotenv()

# --- Email config (IMAP) ---
# Set these as environment variables or fill in directly (never commit credentials)
EMAIL_ADDRESS = os.getenv("QUARDIAN_EMAIL", "you@gmail.com")
EMAIL_PASSWORD = os.getenv("QUARDIAN_PASSWORD", "")  # Use an app password for Gmail
IMAP_SERVER = os.getenv("QUARDIAN_IMAP", "imap.gmail.com")
IMAP_PORT = 993
MAX_EMAILS = 2000  # How many recent emails to pull

# --- Feature config ---
FEATURE_WINDOW_DAYS = 7       # Rolling window for frequency features
TOP_CONTACTS = 20             # How many top contacts to track in the graph

# --- Quantum circuit config ---
N_QUBITS = 6                  # Must match feature vector size after reduction
N_LAYERS = 3                  # Variational layers
N_EPOCHS = 80
LEARNING_RATE = 0.05
TRAIN_SPLIT = 0.8

# --- Anomaly detection config ---
ANOMALY_THRESHOLD = 0.65      # Score above this = flagged
CONTAMINATION = 0.05          # Expected anomaly fraction for classical models

# --- Paths ---
DATA_DIR = "data"
MODEL_PATH = "data/quantum_model.npy"
METADATA_PATH = "data/metadata.csv"
FEATURES_PATH = "data/features.csv"
SYNTHETIC_PATH = "data/synthetic.csv"
