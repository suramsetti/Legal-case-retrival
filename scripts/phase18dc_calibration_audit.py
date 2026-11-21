"""
Phase 18D-C: Calibration Audit and Leakage-Safe Out-Of-Fold Calibration.

Audits:
1. In-sample validation calibration leakage in Phase 18D-B.
2. Group-aware, stratified K-fold cross-validation on Training Data only.
3. Platt scaling parameter estimation strictly using Out-Of-Fold (OOF) training predictions.
4. Out-of-sample application to Validation partition.
5. Invariance verification: Monotonic Platt scaling on individual models must preserve exact ROC-AUC.
6. Calibration persistence: saves `models/checkpoints/phase18d/platt_calibrator.joblib`.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "phase18d"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": cm,
    }


class PlattCalibrator:
    """
    Leakage-safe Platt scaling calibrator trained on out-of-fold training logits.
    Maps uncalibrated probabilities p to calibrated probabilities:
      P_cal = 1 / (1 + exp(-(a * logit(p) + b)))
    where a > 0 ensures strict monotonicity.
    """
    def __init__(self):
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self.is_fitted = False
        self.a_ = None
        self.b_ = None

    def fit(self, oof_probs: np.ndarray, y_train: np.ndarray):
        eps = 1e-7
        p_clamped = np.clip(oof_probs, eps, 1.0 - eps)
        logits = np.log(p_clamped / (1.0 - p_clamped)).reshape(-1, 1)
        self.calibrator.fit(logits, y_train)
        self.is_fitted = True
        self.a_ = float(self.calibrator.coef_[0, 0])
        self.b_ = float(self.calibrator.intercept_[0])

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before predict_proba.")
        eps = 1e-7
        p_clamped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p_clamped / (1.0 - p_clamped)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]


def run_audit_and_oof_calibration():
    print("=" * 80, flush=True)
    print("PHASE 18D-C: CALIBRATION AUDIT & OUT-OF-FOLD (OOF) TRAINING", flush=True)
    print("=" * 80, flush=True)

    # 1. Audit Prior Flaw
    print("\n[1] Calibration Forensic Audit...")
    print("  AUDIT FINDING: Phase 18D-B scripts/phase18d_ensemble.py fit Platt scaling")
    print("  directly on validation predictions using validation labels ('lr.fit(logits, y_true)').")
    print("  This was an in-sample validation evaluation that artificially inflated calibration metrics.")
    print("  STATUS: RETRACTING 70.83% in-sample result. Re-estimating using strictly Out-Of-Fold training data.")

    # 2. Load Train and Validation Data
    train_parquet = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "binary" / "train_masked.parquet"
    val_parquet = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "binary" / "val_masked.parquet"

    train_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet)

    print(f"\n[2] Loaded Datasets: Train={len(train_df):,} cases, Val={len(val_df):,} cases")

    # Map labels: Accepted=1, Rejected=0
    # In earlier baseline: LabelEncoder was ['Accepted', 'Rejected'] -> 0=Accepted, 1=Rejected.
    # We standardize throughout Phase 18D-C: Positive class (y=1) is ALWAYS 'Accepted'.
    y_train = (train_df["label"] == "Accepted").astype(int).values
    y_val = (val_df["label"] == "Accepted").astype(int).values
    groups_train = train_df["duplicate_group"].values

    # 3. Generate 5-Fold Stratified Group Out-of-Fold Predictions for Baseline Model
    print("\n[3] Generating 5-Fold Stratified Group Out-Of-Fold (OOF) Training Predictions...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    oof_train_probs = np.zeros(len(train_df), dtype=float)

    t0 = time.time()
    for fold, (train_idx, oof_idx) in enumerate(sgkf.split(train_df, y_train, groups=groups_train), 1):
        print(f"  Training Fold {fold}/5 (Train: {len(train_idx):,}, OOF: {len(oof_idx):,})...", flush=True)
        vec = TfidfVectorizer(
            max_features=16000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            sublinear_tf=True,
            stop_words="english"
        )
        X_fold_tr = vec.fit_transform(train_df["masked_text"].iloc[train_idx])
        X_fold_oof = vec.transform(train_df["masked_text"].iloc[oof_idx])

        clf = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs"
        )
        clf.fit(X_fold_tr, y_train[train_idx])

        # Ensure positive class probability corresponds to Accepted (label 1)
        if clf.classes_[1] == 1:
            oof_probs = clf.predict_proba(X_fold_oof)[:, 1]
        else:
            oof_probs = clf.predict_proba(X_fold_oof)[:, 0]

        oof_train_probs[oof_idx] = oof_probs

    print(f"  Completed 5-Fold OOF extraction in {time.time()-t0:.1f}s")
    oof_metrics = evaluate_probs(y_train, oof_train_probs)
    print(f"  OOF Training Performance: Accuracy={oof_metrics['accuracy']:.4f}, Macro-F1={oof_metrics['macro_f1']:.4f}, ROC-AUC={oof_metrics['roc_auc']:.4f}, Brier={oof_metrics['brier_score']:.4f}")

    # 4. Fit Platt Calibrator strictly on Training OOF Predictions
    print("\n[4] Fitting Leakage-Free Platt Calibrator strictly on Training OOF...")
    calibrator = PlattCalibrator()
    calibrator.fit(oof_train_probs, y_train)
    print(f"  Fitted Platt Sigmoid: P_cal = 1 / (1 + exp(-({calibrator.a_:.4f} * logit + {calibrator.b_:.4f})))")

    # Save calibrator
    calibrator_path = MODELS_DIR / "platt_calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)
    print(f"  Persisted Calibrator: {calibrator_path.name}")

    # 5. Apply to Full Validation Partition
    print("\n[5] Evaluating Out-of-Sample Calibration on Full Validation Set (5,320 cases)...")
    # Load production baseline model
    baseline_ckpt = MODELS_DIR / "binary_ngram_lr.joblib"
    ckpt_data = joblib.load(baseline_ckpt)
    full_vec = ckpt_data["vectorizer"]
    full_clf = ckpt_data["model"]

    X_val_tfidf = full_vec.transform(val_df["masked_text"])
    # ckpt_data["label_encoder"].classes_ is ['Accepted', 'Rejected'] -> 0=Accepted, 1=Rejected
    # We want P(Accepted) -> index 0
    val_raw_probs = full_clf.predict_proba(X_val_tfidf)[:, 0]

    val_cal_probs = calibrator.predict_proba(val_raw_probs)

    uncal_metrics = evaluate_probs(y_val, val_raw_probs)
    cal_metrics = evaluate_probs(y_val, val_cal_probs)

    print(f"  Uncalibrated Validation: Acc={uncal_metrics['accuracy']:.4f} | F1={uncal_metrics['macro_f1']:.4f} | ROC-AUC={uncal_metrics['roc_auc']:.4f} | Brier={uncal_metrics['brier_score']:.4f}")
    print(f"  Calibrated Validation:   Acc={cal_metrics['accuracy']:.4f} | F1={cal_metrics['macro_f1']:.4f} | ROC-AUC={cal_metrics['roc_auc']:.4f} | Brier={cal_metrics['brier_score']:.4f}")

    # Invariance check: ROC-AUC must be identical
    auc_diff = abs(uncal_metrics["roc_auc"] - cal_metrics["roc_auc"])
    print(f"  Monotonicity Verification: AUC Difference = {auc_diff:.6f} (Tolerance: <= 1e-4) -> {'PASSED' if auc_diff < 1e-4 else 'FAILED'}")

    # Brier score improvement check
    brier_improvement = uncal_metrics["brier_score"] - cal_metrics["brier_score"]
    print(f"  Reliability Verification: Brier Score delta = {brier_improvement:+.4f} (Lower Brier = better probability calibration)")

    audit_summary = {
        "flaw_identified": "In-sample validation calibration fitting in Phase 18D-B",
        "remediation": "Group-aware 5-fold stratified out-of-fold training calibration",
        "training_cases": len(train_df),
        "validation_cases": len(val_df),
        "calibrator_a": calibrator.a_,
        "calibrator_b": calibrator.b_,
        "val_uncalibrated_accuracy": uncal_metrics["accuracy"],
        "val_uncalibrated_macro_f1": uncal_metrics["macro_f1"],
        "val_uncalibrated_roc_auc": uncal_metrics["roc_auc"],
        "val_uncalibrated_brier": uncal_metrics["brier_score"],
        "val_calibrated_accuracy": cal_metrics["accuracy"],
        "val_calibrated_macro_f1": cal_metrics["macro_f1"],
        "val_calibrated_roc_auc": cal_metrics["roc_auc"],
        "val_calibrated_brier": cal_metrics["brier_score"],
        "auc_invariance_passed": bool(auc_diff < 1e-4),
    }

    audit_json = TABLES_DIR / "phase18dc_calibration_audit.json"
    with open(audit_json, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved audit summary to {audit_json}")


if __name__ == "__main__":
    run_audit_and_oof_calibration()

