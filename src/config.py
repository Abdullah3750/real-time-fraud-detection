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

# --- Model serving ---
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "isolation_forest.joblib"))
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.5"))

# --- Kafka ---
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_TRANSACTIONS = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
KAFKA_TOPIC_ALERTS = os.getenv("KAFKA_TOPIC_ALERTS", "fraud-alerts")

# --- API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Producer replay speed ---
REPLAY_EVENTS_PER_SECOND = float(os.getenv("REPLAY_EVENTS_PER_SECOND", "5"))
