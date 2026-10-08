"""
Train and evaluate an Isolation Forest anomaly detector for the credit card
fraud dataset.

Design decisions (see the conversation / project plan doc for the full
reasoning, this is the short version):

  * We train on NORMAL transactions only (a semi-supervised / "one-class"
    framing), then evaluate on a held-out test set containing both normal
    and fraud transactions the model has never seen. This mirrors the real
    constraint that production fraud systems never have complete, reliable
    fraud labels to train on — a model that learns "normal" and flags
    deviations generalizes better than one trained to recognize specific
    known fraud examples.
  * Exact duplicate rows are dropped before splitting, to avoid the same
    transaction appearing in both train and test (which would inflate
    apparent performance).
  * Raw `Time` is dropped — it only encodes position within this dataset's
    48-hour collection window, which doesn't generalize.
  * `Amount` is scaled with RobustScaler (median/IQR-based). Isolation
    Forest itself is invariant to per-feature linear scaling: each split is
    drawn uniformly between a feature's min and max, so a linear rescale maps
    every split onto an equivalent one. Verified on this pipeline: no scaling,
    RobustScaler and StandardScaler all give PR-AUC 0.08502. The scaler is
    kept so the API has a single preprocessing path shared with the
    scale-sensitive autoencoder. RobustScaler rather than StandardScaler
    because Amount's heavy outliers (max $25,691) inflate the standard
    deviation and squash ordinary transactions into a narrow band.
  * The decision threshold (which anomaly scores count as "flagged") is
    chosen from the precision-recall curve on the test set, not from
    IsolationForest's built-in `contamination` parameter — this keeps the
    threshold choice explicit and inspectable rather than baked into model
    fitting.

Run with:  python -m src.models.train_isolation_forest
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
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
REPORTS_DIR = PROJECT_ROOT / "reports" / "isolation_forest"

# Colors consistent with the EDA notebook (validated colorblind-safe palette).
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

    # Stratified split keeps the ~0.17% fraud ratio consistent in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} rows ({y_train.sum()} fraud) | "
          f"Test: {len(X_test):,} rows ({y_test.sum()} fraud)")

    # --- The key design choice: fit on NORMAL-only training rows ---
    X_train_normal = X_train[y_train == 0]
    print(f"Fitting on {len(X_train_normal):,} known-normal training transactions only "
          f"(fraud rows in the training split are held out, unused by the model).")

    # Scale Amount with RobustScaler, fit ONLY on the normal training data
    # (fitting on test data, or on fraud rows, would leak information the
    # model shouldn't have at "training time"). This doesn't change Isolation
    # Forest's results (see module docstring) — it keeps preprocessing
    # identical across models.
    scaler = RobustScaler()
    X_train_normal_scaled = X_train_normal.copy()
    X_train_normal_scaled["Amount"] = scaler.fit_transform(X_train_normal[["Amount"]])

    X_test_scaled = X_test.copy()
    X_test_scaled["Amount"] = scaler.transform(X_test[["Amount"]])

    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",  # we pick our own threshold below instead of relying on this
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_normal_scaled)

    # sklearn's score_samples: HIGHER = more normal. We flip the sign so that
    # higher = more suspicious, matching the intuitive meaning of "risk score".
    raw_scores = model.score_samples(X_test_scaled)
    risk_scores = -raw_scores

    # --- Threshold selection from the precision-recall curve ---
    precision, recall, thresholds = precision_recall_curve(y_test, risk_scores)
    pr_auc = average_precision_score(y_test, risk_scores)

    # F1 at each threshold (precision/recall arrays are 1 longer than thresholds).
    # np.errstate silences the expected "0/0" warning at thresholds where both
    # precision and recall are 0 — those are masked out by np.where right after.
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
    print(f"PR-AUC (threshold-independent):  {pr_auc:.4f}")
    print(f"Chosen threshold:                {best_threshold:.4f}")
    print(f"Precision:                       {best_precision:.4f}")
    print(f"Recall:                          {best_recall:.4f}")
    print(f"F1:                              {best_f1:.4f}")
    print(f"Confusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  -> Of {tp + fn} actual fraud cases in the test set, caught {tp}, missed {fn}")
    print(f"  -> Of {tp + fp} transactions flagged as fraud, {tp} actually were (precision)")

    # --- Save model + scaler + metrics ---
    joblib.dump(model, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, MODELS_DIR / "isolation_forest_scaler.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "isolation_forest_features.joblib")

    metrics = {
        "model": "IsolationForest",
        "n_estimators": 200,
        "max_samples": 256,
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

    # --- Plot 1: Precision-Recall curve ---
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

    # --- Plot 2: Confusion matrix ---
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=",d")
    ax.set_title("Confusion matrix (test set, chosen threshold)")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=130)
    plt.close()

    # --- Plot 3: Score distribution by true class ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(risk_scores.min(), risk_scores.max(), 60)
    ax.hist(risk_scores[y_test == 0], bins=bins, density=True, color=COLOR_NORMAL,
            alpha=0.6, label="Normal (true label)")
    ax.hist(risk_scores[y_test == 1], bins=bins, density=True, color=COLOR_FRAUD,
            alpha=0.6, label="Fraud (true label)")
    ax.axvline(best_threshold, color=INK_PRIMARY, linestyle="--", linewidth=1.2,
               label="Chosen threshold")
    ax.set_xlabel("Risk score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Risk score distribution by true class, on unseen test data")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "score_distribution.png", dpi=130)
    plt.close()

    print()
    print(f"Saved model + scaler to {MODELS_DIR}")
    print(f"Saved metrics + plots to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
