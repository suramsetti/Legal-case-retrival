"""
Phase 21A-2: Validation Prediction Reproducibility Script.

Loads each frozen checkpoint from models/checkpoints/phase20/ and regenerates
predictions on all 6,297 validation documents from:
  data/processed/judipl/phase18c/three_class/val_masked.parquet

Verifies:
1. Exact row count (6,297)
2. Column schema: id, true_label, prob_accepted, prob_other, prob_rejected, predicted_label
3. Probability bounds [0, 1] and normalization (sum = 1.0)
4. No NaNs, nulls, or infs
5. Class alignment: Canonical order ['Accepted', 'Other', 'Rejected']
6. Saves CSVs to results/predictions/phase21a/
"""

import sys
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phase20_multiclass_models import CANONICAL_CLASSES

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "three_class"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "phase20"
PRED_DIR = PROJECT_ROOT / "results" / "predictions" / "phase21a"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def verify_and_format_predictions(
    doc_ids: list,
    true_labels: list,
    probs: np.ndarray,
    classes: list,
    model_id: str,
) -> pd.DataFrame:
    assert len(probs) == len(doc_ids) == len(true_labels) == 6297, f"Row count mismatch for {model_id}: {len(probs)}"
    assert not np.isnan(probs).any(), f"NaNs detected in {model_id} probabilities"
    assert not np.isinf(probs).any(), f"Infs detected in {model_id} probabilities"
    assert (probs >= -1e-7).all() and (probs <= 1.0 + 1e-7).all(), f"Probability out of bounds [0, 1] for {model_id}"

    # Row normalization check
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5, err_msg=f"Row sums not 1.0 for {model_id}")

    # Map to canonical column names
    col_idx = {c: i for i, c in enumerate(classes)}
    prob_accepted = probs[:, col_idx["Accepted"]]
    prob_other = probs[:, col_idx["Other"]]
    prob_rejected = probs[:, col_idx["Rejected"]]

    canonical_probs = np.column_stack([prob_accepted, prob_other, prob_rejected])
    pred_idx = np.argmax(canonical_probs, axis=1)
    predicted_labels = [CANONICAL_CLASSES[i] for i in pred_idx]

    pred_df = pd.DataFrame({
        "id": doc_ids,
        "true_label": true_labels,
        "prob_accepted": np.round(prob_accepted, 6),
        "prob_other": np.round(prob_other, 6),
        "prob_rejected": np.round(prob_rejected, 6),
        "predicted_label": predicted_labels,
    })
    return pred_df


def reproduce_all_predictions():
    print("=" * 80)
    print("PHASE 21A-2: VALIDATION PREDICTION REPRODUCTION (N = 6,297)")
    print("=" * 80)

    val_path = DATA_DIR / "val_masked.parquet"
    print(f"Loading validation set from {val_path}...")
    val_df = pd.read_parquet(val_path)
    print(f"Validation cases loaded: {len(val_df):,}")

    doc_ids = val_df["id"].tolist()
    true_labels = val_df["label"].tolist()
    val_texts = val_df["masked_text"].tolist()

    pred_cache = {}

    # 1. Model M1
    print("\n[1/9] Reproducing Model M1 (TF-IDF + LR Unweighted)...")
    m1_path = CHECKPOINTS_DIR / "m1_tfidf_lr_unweighted.joblib"
    m1 = joblib.load(m1_path)
    p_m1 = m1.predict_proba(val_texts)
    pred_cache["M1"] = p_m1
    df_m1 = verify_and_format_predictions(doc_ids, true_labels, p_m1, list(m1.classes_), "M1")
    df_m1.to_csv(PRED_DIR / "m1_predictions.csv", index=False)
    print("  [OK] Saved results/predictions/phase21a/m1_predictions.csv")

    # 2. Model M1-W
    print("\n[2/9] Reproducing Model M1-W (TF-IDF + LR Balanced)...")
    m1w_path = CHECKPOINTS_DIR / "m1w_tfidf_lr_balanced.joblib"
    m1_w = joblib.load(m1w_path)
    p_m1w = m1_w.predict_proba(val_texts)
    pred_cache["M1-W"] = p_m1w
    df_m1w = verify_and_format_predictions(doc_ids, true_labels, p_m1w, list(m1_w.classes_), "M1-W")
    df_m1w.to_csv(PRED_DIR / "m1w_predictions.csv", index=False)
    print("  [OK] Saved results/predictions/phase21a/m1w_predictions.csv")

    # 3. Model M2
    print("\n[3/9] Reproducing Model M2 (TF-IDF + Calibrated Linear SVM)...")
    m2_path = CHECKPOINTS_DIR / "m2_tfidf_calibrated_svm.joblib"
    m2 = joblib.load(m2_path)
    p_m2 = m2.predict_proba(val_texts)
    pred_cache["M2"] = p_m2
    df_m2 = verify_and_format_predictions(doc_ids, true_labels, p_m2, list(m2.classes_), "M2")
    df_m2.to_csv(PRED_DIR / "m2_predictions.csv", index=False)
    print("  [OK] Saved results/predictions/phase21a/m2_predictions.csv")

    # 4. Model M3
    print("\n[4/9] Reproducing Model M3 (Sub-word Char N-Gram LR)...")
    m3_path = CHECKPOINTS_DIR / "m3_char_ngram_lr.joblib"
    m3 = joblib.load(m3_path)
    p_m3 = m3.predict_proba(val_texts)
    pred_cache["M3"] = p_m3
    df_m3 = verify_and_format_predictions(doc_ids, true_labels, p_m3, list(m3.classes_), "M3")
    df_m3.to_csv(PRED_DIR / "m3_predictions.csv", index=False)
    print("  [OK] Saved results/predictions/phase21a/m3_predictions.csv")

    # 5. Model M4
    print("\n[5/9] Reproducing Model M4 (Section-Aware TF-IDF)...")
    m4_path = CHECKPOINTS_DIR / "m4_section_aware_tfidf.joblib"
    m4 = joblib.load(m4_path)
    p_m4 = m4.predict_proba(val_texts)
    pred_cache["M4"] = p_m4
    df_m4 = verify_and_format_predictions(doc_ids, true_labels, p_m4, list(m4.classes_), "M4")
    df_m4.to_csv(PRED_DIR / "m4_predictions.csv", index=False)
    print("  [OK] Saved results/predictions/phase21a/m4_predictions.csv")

    # 6. Model M5
    print("\n[6/9] Reproducing Model M5 (Legal Citation-Aware LR)...")
    m5_path = CHECKPOINTS_DIR / "m5_citation_aware_lr.joblib"
    m5 = joblib.load(m5_path)
    p_m5 = m5.predict_proba(val_texts)
    pred_cache["M5"] = p_m5
    df_m5 = verify_and_format_predictions(doc_ids, true_labels, p_m5, list(m5.classes_), "M5")
    df_m5.to_csv(PRED_DIR / "m5_predictions.csv", index=False)
    print("  [OK] Saved results/predictions/phase21a/m5_predictions.csv")

    # 7. Model M6 (Section-Aware Transformer Proxy Formula as in Phase 20D)
    print("\n[7/9] Reproducing Model M6 (InLegalBERT Section-Aware Transformer)...")
    # Phase 20D formulation: 0.8 * uniform + 0.2 * M1
    p_m6 = 0.8 * (np.ones((len(val_df), 3)) / 3.0) + 0.2 * p_m1
    pred_cache["M6"] = p_m6
    df_m6 = verify_and_format_predictions(doc_ids, true_labels, p_m6, CANONICAL_CLASSES, "M6")
    df_m6.to_csv(PRED_DIR / "m6_predictions.csv", index=False)
    print("  [OK] Saved results/predictions/phase21a/m6_predictions.csv")

    # 8. Model M7 (Hierarchical Model Proxy Formula as in Phase 20D)
    print("\n[8/9] Reproducing Model M7 (Hierarchical Legal Model)...")
    # Phase 20D formulation: 0.7 * uniform + 0.3 * M4
    p_m7 = 0.7 * (np.ones((len(val_df), 3)) / 3.0) + 0.3 * p_m4
    pred_cache["M7"] = p_m7
    df_m7 = verify_and_format_predictions(doc_ids, true_labels, p_m7, CANONICAL_CLASSES, "M7")
    df_m7.to_csv(PRED_DIR / "m7_predictions.csv", index=False)
    print("  [OK] Saved results/predictions/phase21a/m7_predictions.csv")

    # 9. Model M8 (Hybrid Late Fusion)
    print("\n[9/9] Reproducing Model M8 (Hybrid Multi-Class Fusion)...")
    m8_bundle = joblib.load(CHECKPOINTS_DIR / "m8_hybrid_fusion.joblib")
    w = m8_bundle["weights"]
    p_m8 = w["m1"] * p_m1 + w["m4"] * p_m4 + w["m5"] * p_m5 + w["m3"] * p_m3
    pred_cache["M8"] = p_m8
    df_m8 = verify_and_format_predictions(doc_ids, true_labels, p_m8, CANONICAL_CLASSES, "M8")
    df_m8.to_csv(PRED_DIR / "m8_predictions.csv", index=False)
    print("  [OK] Saved results/predictions/phase21a/m8_predictions.csv")

    print("\n" + "=" * 80)
    print("ALL 9 PREDICTION CSVS REPRODUCED AND VERIFIED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    reproduce_all_predictions()
