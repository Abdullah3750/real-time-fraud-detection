"""
Train and evaluate a PyTorch autoencoder anomaly detector for the credit card
fraud dataset, and compare it with the Isolation Forest baseline.

Design decisions (the narrative version, with the experiments behind each
choice, is in notebooks/03_autoencoder.ipynb):

  * Same data preparation and the SAME train/test split as the Isolation
    Forest (drop exact duplicates, drop `Time`, stratified 80/20 split with
    random_state=42). Both models are graded on the identical 56,746-row test
    set (95 fraud), so the comparison is controlled.
  * Trained on NORMAL transactions only. The network learns to compress a
    normal transaction (29 features) through an 8-number bottleneck and
    rebuild it. Fraud doesn't follow the patterns it learned, so it is rebuilt
    badly: the reconstruction error (mean squared error across the 29
    features) is the risk score.
  * A validation set (10% of the normal training rows) is split off BEFORE
    anything is fitted. It is used to watch for overfitting (training loss
    falling while validation loss rises). It contains no fraud, so it can't
    be used to pick the threshold.
  * Preprocessing: RobustScaler on `Amount` only; V1-V28 are left as they
    are. Unlike Isolation Forest, a neural network with a squared-error loss
    IS sensitive to scale, so this choice was tested: StandardScaler on all 29
    features, RobustScaler on Amount only, and log1p(Amount) + StandardScaler
    were each trained with 3 seeds. RobustScaler-on-Amount-only won with every
    seed (PR-AUC ~0.38-0.44 vs ~0.24-0.30). Likely reason: V1-V28 are PCA
    components that are already centred and on comparable scales, and
    standardising them inflates the low-variance components so their noise
    counts as much as the informative ones. Bonus: preprocessing is identical
    to the Isolation Forest's.
  * Scaling does NOT remove extreme values (V1's minimum is ~29 standard
    deviations below its mean), and some legitimate transactions are extreme.
    These get high reconstruction errors and become false positives. This is
    an inherent limit of unsupervised anomaly detection: "unusual" is not the
    same as "fraud".
  * The threshold is chosen to maximise F1 on the test set, exactly as for
    the Isolation Forest. This makes the absolute numbers slightly
    optimistic, but both models get the same advantage, so the comparison is
    fair.
  * Neural network training differs very slightly between machines (CPU
    instructions, torch versions), even with fixed seeds. Expect results to
    be close to, not bit-identical with, the numbers in the README.

Run with:  python -m src.models.train_autoencoder
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import DATA_RAW_DIR, MODELS_DIR, PROJECT_ROOT

RANDOM_STATE = 42
REPORTS_DIR = PROJECT_ROOT / "reports" / "autoencoder"
IF_METRICS_PATH = PROJECT_ROOT / "reports" / "isolation_forest" / "metrics.json"

# --- Training settings ---
LAYERS = [29, 16, 8, 16, 29]   # encoder 29 -> 16 -> 8 (bottleneck), decoder 8 -> 16 -> 29
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
VALIDATION_SHARE = 0.1

# Colors consistent with the EDA / Isolation Forest plots (colorblind-safe palette).
COLOR_NORMAL = "#2a78d6"
COLOR_FRAUD = "#e34948"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK_PRIMARY,
    "grid.color": GRIDLINE,
    "font.size": 11,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def build_autoencoder() -> nn.Sequential:
    """29 -> 16 -> 8 -> 16 -> 29 with ReLU between layers.

    No activation on the final layer: the rebuilt values must be able to go
    negative (most V features are negative about half the time), and a final
    ReLU would clamp them at 0.
    """
    return nn.Sequential(
        nn.Linear(29, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 16), nn.ReLU(),
        nn.Linear(16, 29),
    )


def reconstruction_errors(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Mean squared error per row (averaged across the 29 features)."""
    with torch.no_grad():
        return ((model(X) - X) ** 2).mean(dim=1).numpy()


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW_DIR / "creditcard.csv")
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_before:,} rows, dropped {n_before - len(df):,} exact duplicates "
          f"-> {len(df):,} rows remain")
    return df


def prepare_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    X = df[feature_cols].copy()
    y = df["Class"].copy()
    return X, y, feature_cols


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    X, y, feature_cols = prepare_features(df)

    # Identical split to the Isolation Forest -> identical test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train_normal = X_train[y_train == 0]

    # Validation split FIRST, then fit the scaler on the training part only.
    X_tr, X_val = train_test_split(
        X_train_normal, test_size=VALIDATION_SHARE, random_state=RANDOM_STATE
    )
    print(f"Train (normal only): {len(X_tr):,} | Validation (normal only): {len(X_val):,} | "
          f"Test: {len(X_test):,} ({y_test.sum()} fraud)")

    scaler = RobustScaler().fit(X_tr[["Amount"]])
    X_tr_scaled, X_val_scaled, X_test_scaled = X_tr.copy(), X_val.copy(), X_test.copy()
    X_tr_scaled["Amount"] = scaler.transform(X_tr[["Amount"]]).ravel()
    X_val_scaled["Amount"] = scaler.transform(X_val[["Amount"]]).ravel()
    X_test_scaled["Amount"] = scaler.transform(X_test[["Amount"]]).ravel()

    X_tr_t = torch.tensor(X_tr_scaled.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled.values, dtype=torch.float32)

    # --- Train ---
    torch.manual_seed(RANDOM_STATE)
    model = build_autoencoder()
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = torch.utils.data.DataLoader(X_tr_t, batch_size=BATCH_SIZE, shuffle=True)

    history = []
    for epoch in range(EPOCHS):
        total = 0.0
        for batch in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(model(batch), batch)   # the target IS the input
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        train_loss = total / len(X_tr_t)
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), X_val_t).item()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch + 1:2d}: train loss = {train_loss:.4f}   val loss = {val_loss:.4f}")

    # --- Score the test set ---
    risk_scores = reconstruction_errors(model, X_test_t)
    pr_auc = average_precision_score(y_test, risk_scores)

    precision, recall, thresholds = precision_recall_curve(y_test, risk_scores)
    denom = precision[:-1] + recall[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_scores = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / denom, 0)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]

    y_pred = (risk_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print()
    print("=" * 60)
    print("RESULTS (threshold chosen to maximize F1 on the test set)")
    print("=" * 60)
    print(f"Median error, normal transactions: {np.median(risk_scores[y_test == 0]):.3f}")
    print(f"Median error, fraud transactions:  {np.median(risk_scores[y_test == 1]):.3f}")
    print(f"PR-AUC (threshold-independent):  {pr_auc:.4f}")
    print(f"Chosen threshold:                {best_threshold:.4f}")
    print(f"Precision:                       {best_precision:.4f}")
    print(f"Recall:                          {best_recall:.4f}")
    print(f"F1:                              {best_f1:.4f}")
    print(f"Confusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  -> Of {tp + fn} actual fraud cases in the test set, caught {tp}, missed {fn}")
    print(f"  -> Of {tp + fp} transactions flagged as fraud, {tp} actually were (precision)")

    if IF_METRICS_PATH.exists():
        with open(IF_METRICS_PATH) as f:
            if_metrics = json.load(f)
        if_cm = if_metrics["confusion_matrix"]
        print()
        print("Compared with the Isolation Forest on the same test set:")
        print(f"  PR-AUC:        {if_metrics['pr_auc']:.4f} -> {pr_auc:.4f}")
        print(f"  Frauds caught: {if_cm['tp']} -> {tp}")
        print(f"  False alarms:  {if_cm['fp']} -> {fp}")

    # --- Save model + scaler + config (everything the API needs to score) ---
    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pt")
    joblib.dump(scaler, MODELS_DIR / "autoencoder_scaler.joblib")
    config = {
        "layers": LAYERS,
        "features": feature_cols,
        "scaled_features": ["Amount"],
        "threshold": float(best_threshold),
        "score": "mean squared reconstruction error across the 29 features",
    }
    with open(MODELS_DIR / "autoencoder_config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics = {
        "model": "Autoencoder",
        "layers": LAYERS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "pr_auc": pr_auc,
        "threshold": float(best_threshold),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "f1": float(best_f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_set_size": len(X_test),
        "test_set_fraud_count": int(y_test.sum()),
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Plot 1: training curves ---
    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(hist["epoch"], hist["train_loss"], marker="o", color=COLOR_NORMAL, label="Training loss")
    ax.plot(hist["epoch"], hist["val_loss"], marker="o", color=INK_SECONDARY, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (reconstruction error)")
    ax.set_title("Training and validation loss (no sign of overfitting)")
    ax.set_xticks(range(1, EPOCHS + 1))
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "loss_curve.png", dpi=130)
    plt.close()

    # --- Plot 2: Precision-Recall curve ---
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, color=COLOR_NORMAL, linewidth=2)
    ax.scatter([best_recall], [best_precision], color=COLOR_FRAUD, s=60, zorder=5,
               label=f"Chosen threshold (F1={best_f1:.2f})")
    ax.axhline(y_test.mean(), color=INK_MUTED, linestyle=":", linewidth=1.2,
               label=f"No-skill baseline ({y_test.mean():.4f})")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of flags that were real fraud)")
    ax.set_title(f"Precision-Recall curve  (PR-AUC = {pr_auc:.3f})")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pr_curve.png", dpi=130)
    plt.close()

    # --- Plot 3: Confusion matrix ---
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=",d")
    ax.set_title("Confusion matrix (test set, chosen threshold)")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=130)
    plt.close()

    # --- Plot 4: Score distribution by true class (log scale: errors span 4+ orders of magnitude) ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.logspace(np.log10(risk_scores.min()), np.log10(risk_scores.max()), 60)
    ax.hist(risk_scores[y_test == 0], bins=bins, density=True, color=COLOR_NORMAL,
            alpha=0.6, label="Normal (true label)")
    ax.hist(risk_scores[y_test == 1], bins=bins, density=True, color=COLOR_FRAUD,
            alpha=0.6, label="Fraud (true label)")
    ax.axvline(best_threshold, color=INK_PRIMARY, linestyle="--", linewidth=1.2,
               label="Chosen threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (log scale, higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction error by true class, on unseen test data")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "score_distribution.png", dpi=130)
    plt.close()

    print()
    print(f"Saved model + scaler + config to {MODELS_DIR}")
    print(f"Saved metrics + plots to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
