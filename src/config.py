"""
Centralized configuration, loaded from environment variables (with .env support).

Keeping all config in one place means the training scripts, the FastAPI service,
and the Kafka producer/consumer all agree on things like the model path and
Kafka topic names without hardcoding them in multiple files.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Model serving (autoencoder) ---
# Folder holding the three artifacts written by src/models/train_autoencoder.py.
# A relative MODEL_DIR is taken relative to the project root, not the current folder.
_model_dir = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR = _model_dir if _model_dir.is_absolute() else PROJECT_ROOT / _model_dir

AE_WEIGHTS_PATH = MODEL_DIR / "autoencoder.pt"
AE_SCALER_PATH = MODEL_DIR / "autoencoder_scaler.joblib"
AE_CONFIG_PATH = MODEL_DIR / "autoencoder_config.json"
# No threshold here on purpose: it belongs to the trained model and is read
# from autoencoder_config.json, so it can never drift out of sync with the weights.

# --- Kafka ---
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_TRANSACTIONS = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
KAFKA_TOPIC_ALERTS = os.getenv("KAFKA_TOPIC_ALERTS", "fraud-alerts")

# --- API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Producer replay speed ---
REPLAY_EVENTS_PER_SECOND = float(os.getenv("REPLAY_EVENTS_PER_SECOND", "5"))
