# Architecture

## System overview

```
                         ┌─────────────────────┐
                         │  Kaggle dataset      │
                         │  (creditcard.csv)    │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Training scripts     │   src/models/
                         │  - Isolation Forest    │   train_isolation_forest.py
                         │  - Autoencoder          │   train_autoencoder.py
                         └──────────┬───────────┘
                                    │  model artifact
                                    ▼
                         ┌──────────────────────┐
                         │  models/*.joblib /    │
                         │  *.pt                 │
                         └──────────┬───────────┘
                                    │ loaded by both
              ┌─────────────────────┼─────────────────────┐
              ▼                                            ▼
  ┌───────────────────────┐                  ┌───────────────────────────┐
  │  FastAPI service        │                  │  Kafka consumer            │
  │  POST /score             │◄────optional────│  (scores each event using  │
  │  (synchronous, for        │   internal call │   the same scorer module) │
  │   ad-hoc / demo requests) │                  └─────────────┬─────────────┘
  └───────────────────────┘                                    │
                                                                 │ flagged txns
                                                                 ▼
  ┌───────────────────────┐                  ┌───────────────────────────┐
  │  Producer                │───publishes────►│  Kafka topic:               │
  │  replays dataset rows      │  transactions   │  transactions                │
  │  at N events/sec           │                  └─────────────┬─────────────┘
  └───────────────────────┘                                    │ consumed by
                                                                 ▼ consumer above
                                                    ┌───────────────────────────┐
                                                    │  Kafka topic: fraud-alerts  │
                                                    │  (flagged transactions)      │
                                                    └───────────────────────────┘
```

## Why these components

**Isolation Forest first, autoencoder second.** Isolation Forest is a strong,
fast, easy-to-interpret baseline for anomaly detection — it isolates outliers
by random recursive partitioning, and anomalies (fraud) tend to isolate in
fewer splits than normal points. Once that baseline works end-to-end, the
autoencoder adds a deep-learning angle: it learns to reconstruct *normal*
transactions well, and a high reconstruction error signals a transaction the
model wasn't trained to expect. Comparing the two is itself a talking point:
tree-based vs. neural anomaly detection, and how they behave differently on
imbalanced data.

**Why anomaly detection instead of plain supervised classification**, even
though the dataset has fraud labels. The labels are used for *evaluation*,
but training as unsupervised anomaly detection mirrors the real-world
constraint that fraud patterns evolve and labeled fraud is always scarce and
lagging — a system that learns "normal" and flags deviations generalizes
better to fraud patterns it hasn't seen than one purely trained to recognize
known fraud examples. (A supervised baseline, e.g. logistic regression or
XGBoost with class weighting, is a natural stretch goal to compare against.)

**Why Kafka.** A single synchronous `/score` endpoint can score one
transaction per request, which is realistic for a payment-time API call, but
doesn't demonstrate the "stream of transactions flowing through a detection
pipeline" that systems like Stripe Radar actually are. Kafka's producer/
consumer model lets us decouple "transactions arriving" from "transactions
being scored," which is how real payment platforms operate: many producers
(payment processors, in this case simulated), one or more consumer groups
doing the scoring, and downstream consumers (alerting, dashboards, case
review) reading flagged events independently.

**Why Docker Compose.** Kafka needs a broker (and, depending on version, a
Zookeeper or KRaft controller) running alongside the app — Compose lets the
whole system start with one command and is the standard way to package a
project like this for a GitHub reviewer to actually run.

## Evaluation approach

With ~0.17% fraud rate, accuracy is meaningless (predicting "not fraud"
always would score 99.8%). Models are evaluated on:

- **Precision / Recall / F1** at the chosen risk threshold
- **PR-AUC** (Precision-Recall AUC) — more informative than ROC-AUC under
  heavy class imbalance
- **Confusion matrix** — to make the false-positive/false-negative trade-off
  concrete (in fraud detection, a false negative costs money, a false
  positive costs customer trust — worth discussing explicitly)

## Open design questions / stretch goals

- Add a supervised model (XGBoost/LightGBM with class weighting) as a
  comparison point against the unsupervised approaches
- Persist flagged transactions to a small database (SQLite/Postgres) for a
  simple review dashboard
- Add basic drift monitoring: track the score distribution over the replayed
  stream and flag if it shifts
- Deploy the API (e.g. Render/Fly.io) for a live demo link in the README
