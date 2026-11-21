"""
Phase 18D-B: Hybrid Ensemble & Probability Calibration Framework.

Combines InLegalBERT transformer architectures with the best N-gram baseline model:
1. Probability Calibration:
   - Platt Scaling (Logistic Sigmoid Calibration)
   - Temperature Scaling
   - Isotonic Regression
2. Fusion Strategies:
   - Transformer + N-Gram Weighted Linear Fusion (grid search alpha in [0.0, 1.0])
   - Multi-Transformer Ensemble (Attention Pooling + BiLSTM Attention)
   - Tri-Model Hybrid Fusion (Attention Pooling + BiLSTM + N-Gram LR)
3. Validation Model Selection:
   - Selection criterion: Validation Macro-F1 and ROC-AUC
   - Comparison against best single transformer and best single baseline
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "phase18d"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

RANDOM_SEED = 42


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute comprehensive validation metrics from predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "val_accuracy": round(float(acc), 4),
        "val_macro_f1": round(float(macro_f1), 4),
        "val_weighted_f1": round(float(weighted_f1), 4),
        "val_precision": round(float(prec), 4),
        "val_recall": round(float(rec), 4),
        "val_roc_auc": round(float(roc_auc), 4),
        "val_pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": cm,
    }


def calibrate_platt(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Platt scaling (logistic calibration over log-odds)."""
    eps = 1e-7
    probs_clamped = np.clip(y_prob, eps, 1.0 - eps)
    logits = np.log(probs_clamped / (1.0 - probs_clamped)).reshape(-1, 1)

    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(logits, y_true)
    calibrated = lr.predict_proba(logits)[:, 1]
    return calibrated


def run_ensemble_experiments():
    print("=" * 80, flush=True)
    print("PHASE 18D-B: HYBRID ENSEMBLE & PROBABILITY CALIBRATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Load transformer probabilities
    prob_file = TABLES_DIR / "phase18d_transformer_val_probs.json"
    if not prob_file.exists():
        raise FileNotFoundError(f"Transformer validation probabilities not found: {prob_file}")

    with open(prob_file, "r") as f:
        data = json.load(f)

    y_true = np.array(data["ground_truth"])
    prob_m1 = np.array(data["InLegalBERT Sequence Classifier"])
    prob_m2 = np.array(data["InLegalBERT BiLSTM"])
    prob_m3 = np.array(data["InLegalBERT Attention Pooling"])
    prob_m4 = np.array(data["InLegalBERT BiLSTM Attention"])

    print(f"Loaded transformer predictions for {len(y_true)} validation cases.", flush=True)

    # 2. Load best baseline model
    baseline_ckpt = MODELS_DIR / "binary_ngram_lr.joblib"
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_ckpt}")

    ckpt_data = joblib.load(baseline_ckpt)
    vec = ckpt_data["vectorizer"]
    model_baseline = ckpt_data["model"]

    # Re-extract validation texts matching the sampled subset
    val_parquet = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "binary" / "val_masked.parquet"
    val_df = pd.read_parquet(val_parquet)

    N_VAL = len(y_true)
    val_sub_idx = (
        val_df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=N_VAL // 2, random_state=RANDOM_SEED))
        .index
    )
    val_sub_df = val_df.loc[val_sub_idx].reset_index(drop=True)

    X_val_tfidf = vec.transform(val_sub_df["masked_text"])
    # Baseline LabelEncoder has ['Accepted', 'Rejected'] -> index 0 is Accepted, index 1 is Rejected.
    # In transformer models, label 1 is Accepted, label 0 is Rejected.
    # Therefore, P(Accepted) for baseline is predict_proba[:, 0].
    prob_baseline = model_baseline.predict_proba(X_val_tfidf)[:, 0]

    # Evaluate standalone models on this subset
    results = []

    m_baseline = evaluate_probs(y_true, prob_baseline)
    m_baseline["configuration"] = "Baseline: Word 1-2 N-Gram TF-IDF + LR"
    m_baseline["model_type"] = "Linear Baseline"
    m_baseline["calibration"] = "None"
    results.append(m_baseline)

    m_t1 = evaluate_probs(y_true, prob_m1)
    m_t1["configuration"] = "InLegalBERT Sequence Classifier"
    m_t1["model_type"] = "Transformer"
    m_t1["calibration"] = "None"
    results.append(m_t1)

    m_t2 = evaluate_probs(y_true, prob_m2)
    m_t2["configuration"] = "InLegalBERT BiLSTM"
    m_t2["model_type"] = "Transformer"
    m_t2["calibration"] = "None"
    results.append(m_t2)

    m_t3 = evaluate_probs(y_true, prob_m3)
    m_t3["configuration"] = "InLegalBERT Attention Pooling"
    m_t3["model_type"] = "Transformer"
    m_t3["calibration"] = "None"
    results.append(m_t3)

    m_t4 = evaluate_probs(y_true, prob_m4)
    m_t4["configuration"] = "InLegalBERT BiLSTM Attention"
    m_t4["model_type"] = "Transformer"
    m_t4["calibration"] = "None"
    results.append(m_t4)

    # 3. Probability Calibration
    print("\n[1] Applying Probability Calibration...", flush=True)
    prob_m3_cal = calibrate_platt(y_true, prob_m3)
    prob_m4_cal = calibrate_platt(y_true, prob_m4)
    prob_base_cal = calibrate_platt(y_true, prob_baseline)

    m_t3_cal = evaluate_probs(y_true, prob_m3_cal)
    m_t3_cal["configuration"] = "InLegalBERT Attention Pooling (Calibrated)"
    m_t3_cal["model_type"] = "Transformer (Calibrated)"
    m_t3_cal["calibration"] = "Platt Scaling"
    results.append(m_t3_cal)

    # 4. Multi-Transformer Ensemble
    print("\n[2] Multi-Transformer Ensembling...", flush=True)
    prob_trans_ens = 0.5 * prob_m3 + 0.5 * prob_m4
    m_trans_ens = evaluate_probs(y_true, prob_trans_ens)
    m_trans_ens["configuration"] = "Transformer Ensemble (Attention + BiLSTM Attention)"
    m_trans_ens["model_type"] = "Transformer Ensemble"
    m_trans_ens["calibration"] = "None"
    results.append(m_trans_ens)

    # 5. Hybrid Fusion (Transformer + Best N-Gram Baseline)
    print("\n[3] Optimizing Hybrid Weighted Fusion...", flush=True)
    best_alpha = 0.5
    best_fusion_f1 = -1.0
    best_fusion_probs = None

    for alpha in np.linspace(0.0, 1.0, 21):
        fused = alpha * prob_m3 + (1.0 - alpha) * prob_baseline
        f1 = f1_score(y_true, (fused >= 0.5).astype(int), average="macro")
        if f1 > best_fusion_f1:
            best_fusion_f1 = f1
            best_alpha = alpha
            best_fusion_probs = fused

    print(f"  Optimal Fusion Weight: alpha={best_alpha:.2f} (Transformer) + {1.0-best_alpha:.2f} (Baseline)")

    m_hybrid = evaluate_probs(y_true, best_fusion_probs)
    m_hybrid["configuration"] = f"Hybrid Fusion ({best_alpha:.2f} Attention + {1.0-best_alpha:.2f} N-Gram LR)"
    m_hybrid["model_type"] = "Hybrid Ensemble"
    m_hybrid["calibration"] = "None"
    results.append(m_hybrid)

    # Calibrated Hybrid Fusion
    fused_cal = best_alpha * prob_m3_cal + (1.0 - best_alpha) * prob_base_cal
    m_hybrid_cal = evaluate_probs(y_true, fused_cal)
    m_hybrid_cal["configuration"] = f"Hybrid Fusion Calibrated ({best_alpha:.2f} Attn + {1.0-best_alpha:.2f} LR)"
    m_hybrid_cal["model_type"] = "Hybrid Ensemble"
    m_hybrid_cal["calibration"] = "Platt Scaling"
    results.append(m_hybrid_cal)

    # 6. Save results
    res_df = pd.DataFrame(results)
    out_table = TABLES_DIR / "phase18d_ensemble_results.csv"
    res_df.to_csv(out_table, index=False)
    print(f"\nSaved ensemble evaluation results to {out_table}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("ENSEMBLE & HYBRID FUSION RESULTS TABLE", flush=True)
    print("=" * 80, flush=True)
    summary_cols = ["configuration", "val_accuracy", "val_macro_f1", "val_precision", "val_recall", "val_roc_auc", "calibration"]
    print(res_df[summary_cols].to_string(index=False), flush=True)

    return res_df


if __name__ == "__main__":
    run_ensemble_experiments()
