"""
Loads the trained autoencoder once and scores transactions.

Kept separate from the FastAPI code so the same scorer can be reused
by the Kafka consumer in phase 8.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import torch
from torch import nn

from src import config


class FraudScorer:
    def __init__(self) -> None:
        cfg = json.loads(config.AE_CONFIG_PATH.read_text())
        self.features = cfg["features"]
        self.scaled_features = cfg["scaled_features"]
        self.threshold = float(cfg["threshold"])

        self.scaler = joblib.load(config.AE_SCALER_PATH)

        self.model = self._build_network(cfg["layers"])
        state = torch.load(config.AE_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _build_network(sizes: list[int]) -> nn.Sequential:
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # no ReLU after the last layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def score(self, transaction: dict) -> dict:
        row = pd.DataFrame([transaction])[self.features]
        row[self.scaled_features] = self.scaler.transform(row[self.scaled_features])
        x = torch.tensor(row.values, dtype=torch.float32)
        with torch.no_grad():
            recon = self.model(x)
        risk = ((recon - x) ** 2).mean().item()
        return {"risk_score": risk, "threshold": self.threshold, "flagged": risk > self.threshold}