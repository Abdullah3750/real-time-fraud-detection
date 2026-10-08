# Fraud Detection API — Real-Time Transaction Risk Scoring

A "Stripe Radar"-style system that scores financial transactions for fraud risk
in real time. Built as a portfolio project for machine learning / data science
graduate applications — it combines classic ML, deep learning, and backend
streaming infrastructure in one coherent system.

> **Status:** 🚧 Work in progress — built incrementally as a learning project.
> See `docs/architecture.md` for the full design and `docs/PROGRESS.md` (once
> added) for what's done so far.

## What it does

1. **Trains anomaly detection models** (Isolation Forest, then a neural
   autoencoder) on the Kaggle "Credit Card Fraud Detection" dataset to learn
   what a *normal* transaction looks like, so it can flag transactions that
   deviate from that pattern.
2. **Serves the model behind a FastAPI endpoint** — `POST /score` takes a
   transaction and returns a risk score + flagged/not-flagged decision.
3. **Replays transactions through Kafka** to simulate a live payments stream:
   a producer publishes transactions at a configurable rate, a consumer reads
   them, scores each one via the model, and publishes any flagged
   transactions to an alerts topic.
4. **Runs as one system via Docker Compose** — Kafka + Zookeeper + the API +
   the streaming consumer, started with a single command.

## Why this project

Fraud/anomaly detection sits at the intersection of applied ML (imbalanced
classification, unsupervised anomaly detection, model evaluation beyond
accuracy) and backend engineering (real-time systems, event streaming,
API design) — both directly relevant to ML/DS graduate study and to how
fraud detection actually works in production fintech systems.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full system design.

```
Kaggle dataset ──► training scripts ──► model artifact (models/*.joblib)
                                                │
                                                ▼
transaction replay ──► Kafka (transactions) ──► consumer ──► FastAPI scorer
   (producer)                                      │
                                                     ▼
                                          Kafka (fraud-alerts) ──► (log / dashboard)
```

## Getting started

> Setup instructions will be filled in as each piece is built. Current state:
> project scaffolding only.

```bash
# 1. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Coming soon) Train the model
python -m src.models.train_isolation_forest

# 4. (Coming soon) Run the full stack
docker compose up
```

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
(Kaggle, ULB Machine Learning Group) — 284,807 transactions made by European
cardholders in September 2013, with 492 labeled frauds (0.172% of all
transactions). Features `V1`–`V28` are PCA-anonymized for confidentiality;
`Time` and `Amount` are the only original features.

## Project structure

```
fraud-detection-api/
├── data/                  # raw & processed data (gitignored)
├── notebooks/             # EDA and experimentation
├── src/
│   ├── data/               # download & preprocessing
│   ├── models/              # training scripts for Isolation Forest / autoencoder
│   ├── api/                 # FastAPI service
│   ├── streaming/            # Kafka producer & consumer
│   └── utils/
├── models/                # trained model artifacts (gitignored)
├── docker/                # Dockerfiles
├── docs/                  # architecture & design notes
├── tests/
└── docker-compose.yml
```

## Results

### Isolation Forest (baseline)

Trained on known-normal transactions only (see `docs/architecture.md` for why),
evaluated on a held-out test set of 56,746 transactions (95 fraud) it never saw
during training:

| Metric | Value |
|---|---|
| PR-AUC | 0.085 (≈51x better than the 0.0017 random-guessing baseline set by the fraud rate) |
| Precision @ chosen threshold | 0.194 |
| Recall @ chosen threshold | 0.253 |
| F1 @ chosen threshold | 0.219 |

In plain terms: of 95 real fraud cases in the test set, this baseline catches
24 and misses 71; of the 124 transactions it flags, 24 are real fraud. Modest
in absolute terms, but a genuine, well-above-chance signal from a model that
was never shown a single labeled fraud example during training — purely
unsupervised anomaly detection is a harder problem than supervised
classification, and this number is the baseline the autoencoder (and any
supervised stretch-goal model) gets compared against. See
`reports/isolation_forest/` for the full evaluation plots (precision-recall
curve, confusion matrix, score distribution by class).

### Autoencoder

_Not yet trained — next phase._

## License

MIT
