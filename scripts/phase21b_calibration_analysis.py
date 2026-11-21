"""
Phase 21B-8: Calibration & Probability Distribution Analysis Script.

Evaluates:
1. Multiclass Brier score
2. Multiclass Log Loss
3. Expected Calibration Error (ECE)
4. High-confidence error rate (confidence > 0.70)
5. Saves results/tables/phase21b_calibration_results.csv
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phase20_multiclass_models import CANONICAL_CLASSES

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "three_class"
PRED_DIR = PROJECT_ROOT / "results" / "predictions" / "phase21b"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_calibration_metrics(df_pred: pd.DataFrame, df_true: pd.DataFrame, model_id: str):
    merged = pd.merge(df_pred, df_true[["id", "label"]], on="id")
    y_true = merged["label"].values
    probs = merged[["prob_accepted", "prob_other", "prob_rejected"]].values

    # Brier Score (Multiclass)
    y_dummies = pd.get_dummies(y_true)[CANONICAL_CLASSES].values
    brier = np.mean(np.sum((probs - y_dummies) ** 2, axis=1))

    # Log Loss
    loss_val = log_loss(y_dummies, probs)

    # Top-1 confidence and correctness
    top_conf = np.max(probs, axis=1)
    preds_idx = np.argmax(probs, axis=1)
    pred_labels = np.array([CANONICAL_CLASSES[i] for i in preds_idx])
    correct = (pred_labels == y_true)

    # Expected Calibration Error (ECE) with 10 bins
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_mask = (top_conf > bin_boundaries[i]) & (top_conf <= bin_boundaries[i + 1])
        bin_size = bin_mask.sum()
        if bin_size > 0:
            bin_acc = correct[bin_mask].mean()
            bin_conf = top_conf[bin_mask].mean()
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    # High-confidence error rate (confidence > 0.70)
    high_conf_mask = top_conf > 0.70
    n_high_conf = int(high_conf_mask.sum())
    high_conf_errors = int((~correct[high_conf_mask]).sum()) if n_high_conf > 0 else 0
    high_conf_err_rate = round(high_conf_errors / n_high_conf * 100.0, 2) if n_high_conf > 0 else 0.0

    return {
        "model_id": model_id,
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(loss_val), 4),
        "ece_10_bins": round(float(ece), 4),
        "mean_confidence": round(float(top_conf.mean()), 4),
        "high_conf_cases_gt_70": n_high_conf,
        "high_conf_errors": high_conf_errors,
        "high_conf_error_rate_pct": high_conf_err_rate,
    }


def run_calibration_analysis():
    print("=" * 80)
    print("PHASE 21B-8: CALIBRATION & PROBABILITY ANALYSIS")
    print("=" * 80)

    te_df = pd.read_parquet(DATA_DIR / "test_masked.parquet")
    df_m4 = pd.read_csv(PRED_DIR / "m4_test_predictions.csv", dtype={"id": str})
    df_m8 = pd.read_csv(PRED_DIR / "m8_test_predictions.csv", dtype={"id": str})

    cal_m4 = compute_calibration_metrics(df_m4, te_df, "M4 (Section-Aware TF-IDF)")
    cal_m8 = compute_calibration_metrics(df_m8, te_df, "M8 (Hybrid Fusion Ensemble)")

    cal_df = pd.DataFrame([cal_m4, cal_m8])
    out_csv = TABLES_DIR / "phase21b_calibration_results.csv"
    cal_df.to_csv(out_csv, index=False)
    print(f"Saved Calibration Results to {out_csv}")
    print(cal_df)


if __name__ == "__main__":
    run_calibration_analysis()

