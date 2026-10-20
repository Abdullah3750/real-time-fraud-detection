"""Request and response shapes for the scoring API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """One card transaction: the 29 features the autoencoder was trained on."""

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0, description="Transaction amount in the original currency (not scaled)")


class ScoreResponse(BaseModel):
    """What /score sends back."""

    risk_score: float = Field(description="Mean squared reconstruction error; higher = more unusual")
    threshold: float = Field(description="Scores above this are flagged")
    flagged: bool