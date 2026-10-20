"""
FastAPI service: POST /score returns a fraud risk score for one transaction.

Run from the project root:
    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.model import FraudScorer
from src.api.schemas import ScoreResponse, Transaction

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs ONCE when the server starts: load the model
    ml["scorer"] = FraudScorer()
    yield
    # runs once when the server stops
    ml.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Scores card transactions with an autoencoder trained on normal transactions.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "scorer" in ml}


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction) -> ScoreResponse:
    result = ml["scorer"].score(tx.model_dump())
    return ScoreResponse(**result)