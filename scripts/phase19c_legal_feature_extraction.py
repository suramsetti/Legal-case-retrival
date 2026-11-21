"""
Phase 19C: Legal Feature Extraction & Leakage Audit Script.

Extracts structured legal features across the validation set (N=5,320) and conducts a formal
statistical leakage audit to verify that no feature leaks the outcome label.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from src.features.phase19_legal_features import LegalFeatureExtractor

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"


def run_phase19c_feature_extraction_audit():
    print("=" * 80, flush=True)
    print("PHASE 19C: LEGAL FEATURE EXTRACTION & STATISTICAL LEAKAGE AUDIT", flush=True)
    print("=" * 80, flush=True)

    val_path = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "binary" / "val_masked.parquet"
    val_df = pd.read_parquet(val_path)
    print(f"Loaded validation set: {len(val_df):,} judgments", flush=True)

    extractor = LegalFeatureExtractor()
    y_val = (val_df["label"] == "Accepted").astype(int).values

    print("Extracting 15 structured legal features...", flush=True)
    # Extract on validation set
    feats = extractor.extract_feature_matrix(val_df["masked_text"].tolist())
    feat_df = pd.DataFrame(feats, columns=extractor.FEATURE_NAMES)

    print("\nFeature Summary Statistics:")
    print("-" * 75)
    audit_records = []
    max_corr = 0.0

    for name in extractor.FEATURE_NAMES:
        col = feat_df[name].values
        mean_val = np.mean(col)
        std_val = np.std(col)
        min_val = np.min(col)
        max_val = np.max(col)

        # Correlation with outcome
        r, p_val = pointbiserialr(col, y_val)
        abs_r = abs(r) if not np.isnan(r) else 0.0
        max_corr = max(max_corr, abs_r)

        # Leakage threshold check: correlation > 0.35 would indicate leakage
        leakage_status = "SAFE" if abs_r < 0.25 else ("SUSPICIOUS" if abs_r < 0.50 else "LEAKAGE_DETECTED")

        print(f"  {name:<28} Mean: {mean_val:7.2f} | Std: {std_val:7.2f} | Corr(r): {r:+.4f} [{leakage_status}]")
        audit_records.append({
            "feature_name": name,
            "mean": round(float(mean_val), 4),
            "std": round(float(std_val), 4),
            "min": round(float(min_val), 4),
            "max": round(float(max_val), 4),
            "point_biserial_r": round(float(r), 4) if not np.isnan(r) else 0.0,
            "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
            "leakage_status": leakage_status,
        })

    audit_df = pd.DataFrame(audit_records)
    out_csv = TABLES_DIR / "phase19_feature_leakage_audit.csv"
    audit_df.to_csv(out_csv, index=False)
    print(f"\nSaved feature audit to {out_csv}", flush=True)
    print(f"Maximum feature-label correlation: {max_corr:.4f}")
    if max_corr < 0.25:
        print("[VERIFIED SAFE] All legal features have low correlation (|r| < 0.25) with label.")
    else:
        print("[WARNING] One or more features require scrutiny.")


if __name__ == "__main__":
    run_phase19c_feature_extraction_audit()

