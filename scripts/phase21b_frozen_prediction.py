"""
Phase 21B-3: Frozen Model Prediction Generation on Eligible Test Partition.

Generates predictions for:
- M4: Section-Aware TF-IDF Classifier
- M8: Hybrid Multi-Class Fusion Ensemble (0.50 M1 + 0.25 M4 + 0.15 M5 + 0.10 M3)

Evaluates on the eligible three-class test partition (N = 6,311),
excluding the 2 cross-partition duplicate cases identified in Phase 21B-2.
Labels are NOT accessed or inspected during prediction generation.
"""

import sys
import os
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phase20_multiclass_models import CANONICAL_CLASSES

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "three_class"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "phase20"
PRED_DIR = PROJECT_ROOT / "results" / "predictions" / "phase21b"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

PRED_DIR.mkdir(parents=True, exist_ok=True)


def generate_frozen_predictions():
    print("=" * 80, flush=True)
    print("PHASE 21B-3: FROZEN MODEL TEST PREDICTION GENERATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Load exclusion manifest
    manifest_path = REPORTS_DIR / "phase21b_duplicate_exclusion_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    excluded_ids = set(manifest["excluded_test_ids"])
    print(f"Loaded exclusion manifest: excluding {len(excluded_ids)} IDs: {excluded_ids}", flush=True)

    # 2. Load test dataset (mask text only, without inspecting labels)
    te_path = DATA_DIR / "test_masked.parquet"
    full_te_df = pd.read_parquet(te_path)
    print(f"Total test partition cases: {len(full_te_df):,}", flush=True)

    # Filter eligible
    eligible_mask = ~full_te_df["id"].isin(excluded_ids)
    eligible_df = full_te_df[eligible_mask].reset_index(drop=True)
    n_eligible = len(eligible_df)
    print(f"Eligible test cases for evaluation: {n_eligible:,}", flush=True)
    assert n_eligible == 6311, f"Expected 6,311 eligible cases, got {n_eligible}"

    test_ids = eligible_df["id"].tolist()
    test_texts = eligible_df["masked_text"].tolist()

    # 3. Load checkpoints
    print("\nLoading frozen checkpoints...", flush=True)
    t0 = time.time()
    m4 = joblib.load(CHECKPOINTS_DIR / "m4_section_aware_tfidf.joblib")
    m1 = joblib.load(CHECKPOINTS_DIR / "m1_tfidf_lr_unweighted.joblib")
    m3 = joblib.load(CHECKPOINTS_DIR / "m3_char_ngram_lr.joblib")
    m5 = joblib.load(CHECKPOINTS_DIR / "m5_citation_aware_lr.joblib")
    m8_bundle = joblib.load(CHECKPOINTS_DIR / "m8_hybrid_fusion.joblib")
    w = m8_bundle["weights"]
    print(f"Loaded all checkpoints in {time.time() - t0:.2f}s. M8 fusion weights: {w}", flush=True)

    # 4. Generate M4 Predictions
    print("\nGenerating predictions for Model M4 (Section-Aware TF-IDF)...", flush=True)
    t_start = time.time()
    p_m4_raw = m4.predict_proba(test_texts)
    t_m4 = time.time() - t_start
    print(f"  M4 inference completed in {t_m4:.2f}s ({t_m4 / n_eligible * 1000.0:.2f} ms/doc)", flush=True)

    # Align to canonical order
    m4_classes = list(m4.classes_)
    col_idx_m4 = {c: i for i, c in enumerate(m4_classes)}
    p_m4 = np.column_stack([
        p_m4_raw[:, col_idx_m4["Accepted"]],
        p_m4_raw[:, col_idx_m4["Other"]],
        p_m4_raw[:, col_idx_m4["Rejected"]],
    ])

    pred_idx_m4 = np.argmax(p_m4, axis=1)
    labels_m4 = [CANONICAL_CLASSES[i] for i in pred_idx_m4]

    df_m4 = pd.DataFrame({
        "id": test_ids,
        "predicted_label": labels_m4,
        "prob_accepted": np.round(p_m4[:, 0], 6),
        "prob_other": np.round(p_m4[:, 1], 6),
        "prob_rejected": np.round(p_m4[:, 2], 6),
    })

    # Validate M4
    assert len(df_m4) == 6311
    assert not df_m4[["prob_accepted", "prob_other", "prob_rejected"]].isna().any().any()
    np.testing.assert_allclose(p_m4.sum(axis=1), 1.0, atol=1e-5)
    out_m4 = PRED_DIR / "m4_test_predictions.csv"
    df_m4.to_csv(out_m4, index=False)
    print(f"  [OK] Saved M4 test predictions to {out_m4}", flush=True)

    # 5. Generate Component Predictions for M8
    print("\nGenerating component predictions for Model M8 (Hybrid Fusion)...", flush=True)

    print("  Predicting with M1 (TF-IDF LR)...", flush=True)
    p_m1_raw = m1.predict_proba(test_texts)
    col_idx_m1 = {c: i for i, c in enumerate(m1.classes_)}
    p_m1 = np.column_stack([
        p_m1_raw[:, col_idx_m1["Accepted"]],
        p_m1_raw[:, col_idx_m1["Other"]],
        p_m1_raw[:, col_idx_m1["Rejected"]],
    ])

    print("  Predicting with M5 (Citation + Dense LR)...", flush=True)
    p_m5_raw = m5.predict_proba(test_texts)
    col_idx_m5 = {c: i for i, c in enumerate(m5.classes_)}
    p_m5 = np.column_stack([
        p_m5_raw[:, col_idx_m5["Accepted"]],
        p_m5_raw[:, col_idx_m5["Other"]],
        p_m5_raw[:, col_idx_m5["Rejected"]],
    ])

    print("  Predicting with M3 (Sub-word Char N-Gram LR)...", flush=True)
    p_m3_raw = m3.predict_proba(test_texts)
    col_idx_m3 = {c: i for i, c in enumerate(m3.classes_)}
    p_m3 = np.column_stack([
        p_m3_raw[:, col_idx_m3["Accepted"]],
        p_m3_raw[:, col_idx_m3["Other"]],
        p_m3_raw[:, col_idx_m3["Rejected"]],
    ])

    # Apply Frozen Late Fusion
    print(f"  Combining late fusion: {w['m1']}*M1 + {w['m4']}*M4 + {w['m5']}*M5 + {w['m3']}*M3...", flush=True)
    p_m8 = w["m1"] * p_m1 + w["m4"] * p_m4 + w["m5"] * p_m5 + w["m3"] * p_m3

    pred_idx_m8 = np.argmax(p_m8, axis=1)
    labels_m8 = [CANONICAL_CLASSES[i] for i in pred_idx_m8]

    df_m8 = pd.DataFrame({
        "id": test_ids,
        "predicted_label": labels_m8,
        "prob_accepted": np.round(p_m8[:, 0], 6),
        "prob_other": np.round(p_m8[:, 1], 6),
        "prob_rejected": np.round(p_m8[:, 2], 6),
    })

    # Validate M8
    assert len(df_m8) == 6311
    assert not df_m8[["prob_accepted", "prob_other", "prob_rejected"]].isna().any().any()
    np.testing.assert_allclose(p_m8.sum(axis=1), 1.0, atol=1e-5)
    assert (df_m4["id"] == df_m8["id"]).all(), "ID misalignment between M4 and M8"

    out_m8 = PRED_DIR / "m8_test_predictions.csv"
    df_m8.to_csv(out_m8, index=False)
    print(f"  [OK] Saved M8 test predictions to {out_m8}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"PREDICTION GENERATION COMPLETE FOR ALL {n_eligible:,} ELIGIBLE CASES.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    generate_frozen_predictions()

