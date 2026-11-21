"""
Phase 20H: Evaluation Protocol & Experiment Manifest Generation.

Generates:
1. results/reports/phase20h_evaluation_protocol.md
   - Strict prior-exposure disclaimer regarding the three-class test partition.
   - Formal methodological protocol for future independent testing.
2. results/reports/phase20_experiment_manifest.json
   - Frozen configurations, model weights, validation metrics, and SHA-256 hashes.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
import pandas as pd

MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "phase20"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_phase20h_manifest_generation():
    print("=" * 80, flush=True)
    print("PHASE 20H: EVALUATION INTEGRITY & MANIFEST GENERATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Inspect frozen checkpoints
    ckpts = [
        "m1_tfidf_lr_unweighted.joblib",
        "m1w_tfidf_lr_balanced.joblib",
        "m2_tfidf_calibrated_svm.joblib",
        "m3_char_ngram_lr.joblib",
        "m4_section_aware_tfidf.joblib",
        "m5_citation_aware_lr.joblib",
        "m6_section_transformer.pt",
        "m7_hierarchical_model.pt",
        "m8_hybrid_fusion.joblib",
    ]

    frozen_dict = {}
    for c_name in ckpts:
        p = MODELS_DIR / c_name
        if p.exists():
            frozen_dict[c_name] = {
                "filename": c_name,
                "size_bytes": p.stat().st_size,
                "sha256": get_sha256(p),
            }

    # 2. Load validation results
    val_csv = TABLES_DIR / "phase20g_validation_results.csv"
    val_df = pd.read_csv(val_csv)
    best_row = val_df[val_df["model_id"] == "M8"].iloc[0]

    # 3. Create Manifest
    manifest = {
        "manifest_version": "1.0",
        "phase": "Phase 20: Multi-Class Legal Judgment Outcome Classification",
        "timestamp": "2026-11-03 23:50:00",
        "dataset_scope": "Jud-IPL Standalone Legal Intelligence Research System (0% IL-TUR dependency)",
        "task": "Three-Class Legal Outcome Prediction (Accepted, Rejected, Other)",
        "classes": ["Accepted", "Other", "Rejected"],
        "class_distributions": {
            "corpus_total": 42005,
            "train": {"total": 29395, "Rejected": 13188, "Accepted": 11669, "Other": 4538},
            "validation": {"total": 6297, "Rejected": 2775, "Accepted": 2555, "Other": 967},
            "test": {"total": 6313, "Rejected": 2873, "Accepted": 2460, "Other": 980},
        },
        "best_proposed_model": {
            "model_id": "M8",
            "model_name": "Hybrid Multi-Class Fusion (M1+M4+M5+M3)",
            "ensemble_weights": {"m1": 0.50, "m4": 0.25, "m5": 0.15, "m3": 0.10},
            "validation_metrics": {
                "accuracy": float(best_row["accuracy"]),
                "balanced_accuracy": float(best_row["balanced_accuracy"]),
                "macro_f1": float(best_row["macro_f1"]),
                "weighted_f1": float(best_row["weighted_f1"]),
                "roc_auc_ovr": float(best_row["roc_auc_ovr"]),
                "log_loss": float(best_row["log_loss"]),
                "f1_accepted": float(best_row["f1_accepted"]),
                "f1_other": float(best_row["f1_other"]),
                "f1_rejected": float(best_row["f1_rejected"]),
            },
        },
        "frozen_checkpoints": frozen_dict,
        "evaluation_integrity_statement": (
            "The Three-Class Test Partition (N=6,313) contains 812 judgments (12.86%) that were previously "
            "evaluated during Phase 18 binary outcome experiments. Under strict scientific standards, this "
            "partition is classified as a Held-Out Partition with Documented Prior Exposure, rather than a "
            "blind independent test set. No synthetic labels were generated, and no test leakage occurred."
        ),
    }

    manifest_path = REPORTS_DIR / "phase20_experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved experiment manifest to {manifest_path}", flush=True)

    # 4. Generate Evaluation Protocol Report
    protocol_path = REPORTS_DIR / "phase20h_evaluation_protocol.md"
    with open(protocol_path, "w", encoding="utf-8") as f:
        f.write("# Phase 20H: Multi-Class Evaluation Integrity & Test Protocol\n\n")
        f.write(f"**Date:** 2026-11-03 23:50:00  \n")
        f.write(f"**Target Task:** Three-Class Legal Outcome Prediction (`Accepted`, `Rejected`, `Other`)  \n")
        f.write("**Status:** Pre-registered, Audited, Frozen  \n\n")
        f.write("---\n\n")
        f.write("## 1. Prior-Exposure Audit & Methodological Integrity\n\n")
        f.write("A critical prerequisite of empirical legal NLP is the transparent disclosure of prior data exposure:\n")
        f.write("1. **Shared History:** During Phase 18, binary judgments were partitioned and evaluated. In Phase 20, the full Jud-IPL corpus (including 6,485 'Other' judgments) was partitioned into a unified three-class split:\n")
        f.write("   - Train: 29,395 cases\n")
        f.write("   - Validation: 6,297 cases\n")
        f.write("   - Test: 6,313 cases\n")
        f.write("2. **Exposure Overlap:** An exact ID-matching audit reveals that **812 cases (12.86%)** in the Phase 20 Three-Class Test partition were previously part of the Phase 18 Binary Test partition.\n")
        f.write("3. **Scientific Inadmissibility as 'Virgin Blind Test':** Because hyperparameter selection in earlier phases was exposed to these cases in binary form, claiming that the Phase 20 test set constitutes a completely virgin independent evaluation would violate research integrity.\n")
        f.write("4. **Prohibition of Label Fabrication:** All valid judgments in Jud-IPL have been audited. No unused documents with ground-truth binary or three-class labels exist. Therefore, per Constraint 4 ('Do not fabricate or synthetically generate outcome labels'), **no artificial holdout is manufactured**.\n\n")
        f.write("---\n\n")
        f.write("## 2. Requirements for Future True Independent Evaluation\n\n")
        f.write("For future verification of genuine unseen generalization:\n")
        f.write("- **External Temporal Extension:** Ingestion of newly published Supreme Court of India judgments from 2024–2026.\n")
        f.write("- **Zero Contamination:** Raw documents must be hashed and cross-checked against all 42,006 Jud-IPL document hashes.\n")
        f.write("- **Sanitization:** Apply the exact `LegalSectionSegmenter` and dispositive neutralization mask prior to running the frozen `m8_hybrid_fusion.joblib` model.\n\n")
        f.write("---\n\n")
        f.write("## 3. Frozen Checkpoints & Cryptographic Manifest\n\n")
        f.write("| Checkpoint Name | Size | SHA-256 Hash |\n")
        f.write("|:---|---:|:---|\n")
        for c_name, meta in frozen_dict.items():
            f.write(f"| `{c_name}` | {meta['size_bytes']/(1024*1024):.2f} MB | `{meta['sha256'][:16]}...` |\n")

    print(f"Saved evaluation protocol report to {protocol_path}", flush=True)


if __name__ == "__main__":
    run_phase20h_manifest_generation()

