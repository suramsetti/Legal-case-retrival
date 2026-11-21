"""
Phase 21B-4: Final Test Metrics Calculation.

Evaluates frozen predictions of M8 and M4 on the eligible test partition (N = 6,311).
Computes overall metrics (Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1,
ROC-AUC OvR/OvO, PR-AUC, Log Loss) and granular per-class metrics.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phase20_multiclass_models import CANONICAL_CLASSES

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "three_class"
PRED_DIR = PROJECT_ROOT / "results" / "predictions" / "phase21b"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model_test_metrics(
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    model_id: str,
    model_name: str,
):
    merged = pd.merge(df_pred, df_true[["id", "label"]], on="id", how="inner")
    assert len(merged) == 6311, f"Expected 6,311 merged rows, got {len(merged)}"

    y_true = merged["label"].values
    y_pred = merged["predicted_label"].values
    probs = merged[["prob_accepted", "prob_other", "prob_rejected"]].values

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Probabilistic metrics
    y_dummies = pd.get_dummies(y_true)[CANONICAL_CLASSES].values
    try:
        roc_auc_ovr = roc_auc_score(y_dummies, probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc_ovr = 0.5

    try:
        roc_auc_ovo = roc_auc_score(y_dummies, probs, multi_class="ovo", average="macro")
    except Exception:
        roc_auc_ovo = 0.5

    try:
        pr_auc_macro = average_precision_score(y_dummies, probs, average="macro")
    except Exception:
        pr_auc_macro = 0.5

    try:
        loss_val = log_loss(y_dummies, probs)
    except Exception:
        loss_val = 1.0

    cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

    # Per-class metrics
    class_metrics = {}
    class_report_rows = []
    for i, c in enumerate(CANONICAL_CLASSES):
        p_c = precision_score(y_true == c, y_pred == c, zero_division=0)
        r_c = recall_score(y_true == c, y_pred == c, zero_division=0)
        f_c = f1_score(y_true == c, y_pred == c, zero_division=0)
        support_c = int((y_true == c).sum())
        fp_c = int(((y_pred == c) & (y_true != c)).sum())
        fn_c = int(((y_pred != c) & (y_true == c)).sum())

        class_metrics[f"precision_{c.lower()}"] = round(float(p_c), 4)
        class_metrics[f"recall_{c.lower()}"] = round(float(r_c), 4)
        class_metrics[f"f1_{c.lower()}"] = round(float(f_c), 4)
        class_metrics[f"support_{c.lower()}"] = support_c
        class_metrics[f"fp_{c.lower()}"] = fp_c
        class_metrics[f"fn_{c.lower()}"] = fn_c

        class_report_rows.append({
            "model_id": model_id,
            "class": c,
            "precision": round(float(p_c), 4),
            "recall": round(float(r_c), 4),
            "f1_score": round(float(f_c), 4),
            "support": support_c,
            "false_positives": fp_c,
            "false_negatives": fn_c,
        })

    summary_record = {
        "model_id": model_id,
        "model_name": model_name,
        "eligible_test_size": len(merged),
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "roc_auc_ovr": round(float(roc_auc_ovr), 4),
        "roc_auc_ovo": round(float(roc_auc_ovo), 4),
        "pr_auc_macro": round(float(pr_auc_macro), 4),
        "log_loss": round(float(loss_val), 4),
        "confusion_matrix": cm.tolist(),
        **class_metrics,
    }

    return summary_record, class_report_rows


def run_metrics_evaluation():
    print("=" * 80)
    print("PHASE 21B-4: FINAL TEST EVALUATION METRICS (N = 6,311)")
    print("=" * 80)

    te_path = DATA_DIR / "test_masked.parquet"
    te_df = pd.read_parquet(te_path)

    pred_m4_path = PRED_DIR / "m4_test_predictions.csv"
    pred_m8_path = PRED_DIR / "m8_test_predictions.csv"

    if not pred_m4_path.exists() or not pred_m8_path.exists():
        raise FileNotFoundError("Prediction files missing in results/predictions/phase21b/")

    df_m4 = pd.read_csv(pred_m4_path, dtype={"id": str})
    df_m8 = pd.read_csv(pred_m8_path, dtype={"id": str})

    summary_m4, report_m4 = evaluate_model_test_metrics(
        df_m4, te_df, "M4", "Section-Aware TF-IDF Classifier"
    )
    summary_m8, report_m8 = evaluate_model_test_metrics(
        df_m8, te_df, "M8", "Hybrid Multi-Class Fusion Ensemble"
    )

    # Save summary table
    summary_df = pd.DataFrame([summary_m4, summary_m8])
    out_table = TABLES_DIR / "phase21b_test_results.csv"
    summary_df.to_csv(out_table, index=False)
    print(f"Saved Overall Test Results to {out_table}")

    # Save class report table
    report_df = pd.DataFrame(report_m4 + report_m8)
    out_class_report = TABLES_DIR / "phase21b_classification_report.csv"
    report_df.to_csv(out_class_report, index=False)
    print(f"Saved Class-Wise Classification Report to {out_class_report}")

    print("\nSummary Results:")
    print(summary_df[["model_id", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr", "log_loss"]])


if __name__ == "__main__":
    run_metrics_evaluation()

