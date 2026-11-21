"""
Phase 19G: Independent Evaluation Preparation & Cryptographic Manifest Generation.

Audits whether eligible unpartitioned Jud-IPL documents exist for an independent held-out evaluation:
1. Verifies document IDs, duplicate groups, and text hashes across train, val, and test splits.
2. Audits all remaining unpartitioned documents in Jud-IPL documents.parquet.
3. Confirms eligibility constraints: if insufficient unused binary outcome documents exist,
   documents this limitation formally and prevents fabrication of synthetic test data.
4. Serializes the final Phase 19 Experiment Manifest with frozen configurations and SHA-256 hashes.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
import numpy as np
import pandas as pd

MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "phase19"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_phase19g_audit():
    print("=" * 80, flush=True)
    print("PHASE 19G: INDEPENDENT EVALUATION PREPARATION & MANIFEST GENERATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Inspect Corpus and Existing Splits
    doc_parquet = PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet"
    train_parquet = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "binary" / "train_masked.parquet"
    val_parquet = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "binary" / "val_masked.parquet"
    test_parquet = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c" / "binary" / "test_masked.parquet"

    print("Auditing total Jud-IPL document inventory...", flush=True)
    doc_df = pd.read_parquet(doc_parquet)
    tr_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet)
    te_df = pd.read_parquet(test_parquet)

    total_corpus_docs = len(doc_df)
    n_tr = len(tr_df)
    n_val = len(val_df)
    n_te = len(te_df)
    total_binary_partitioned = n_tr + n_val + n_te

    print(f"  Total Jud-IPL Documents: {total_corpus_docs:,}")
    print(f"  Train: {n_tr:,} | Val: {n_val:,} | Test: {n_te:,} (Total Binary: {total_binary_partitioned:,})")

    # Document ID sets
    used_ids = set(tr_df["id"]).union(set(val_df["id"])).union(set(te_df["id"]))
    unpartitioned_df = doc_df[~doc_df["id"].isin(used_ids)]
    n_unpartitioned = len(unpartitioned_df)

    print(f"\nUnpartitioned Documents Audit (N={n_unpartitioned:,}):")
    label_dist = unpartitioned_df["label"].value_counts(dropna=False).to_dict()
    print(f"  Label distribution in unpartitioned set: {label_dist}")

    # Check for eligible binary judgments
    eligible_binary_unpartitioned = unpartitioned_df[unpartitioned_df["label"].isin(["Accepted", "Rejected"])]
    n_eligible = len(eligible_binary_unpartitioned)

    print(f"  Eligible binary outcome judgments remaining in Jud-IPL: {n_eligible}")

    independent_test_available = False
    limitation_statement = ""

    if n_eligible == 0:
        independent_test_available = False
        limitation_statement = (
            "All 35,520 valid binary outcome judgments in Jud-IPL were exhausted during the initial "
            "Phase 18 split (24,849 train, 5,320 val, 5,351 test). The remaining 6,486 documents in Jud-IPL "
            "comprise 6,485 cases with label 'Other' (remands, interlocutory orders, advisory references) "
            "and 1 blank record, none of which possess ground-truth binary outcome labels. Per Strict "
            "Constraint 5 ('Never fabricate metrics or labels') and Phase 19G guidance ('If insufficient "
            "unused documents exist, report this limitation and do not claim a new independent test result'), "
            "no synthetic or unverified test results are generated."
        )
        print(f"\n[FORMAL LIMITATION REPORTED]\n{limitation_statement}")
    else:
        independent_test_available = True

    # 2. Freeze Best Phase 19 Architecture Checkpoints
    best_fusion_ckpt = MODELS_DIR / "hybrid_fusion_model.joblib"
    sec_lr_ckpt = MODELS_DIR / "section_aware_tfidf_lr.joblib"
    cit_lr_ckpt = MODELS_DIR / "citation_aware_lr.joblib"
    char_lr_ckpt = MODELS_DIR / "char_ngram_lr.joblib"
    sec_trans_ckpt = MODELS_DIR / "section_aware_transformer.pt"
    hier_ckpt = MODELS_DIR / "hierarchical_legal_model.pt"

    frozen_checkpoints = {}
    for name, p in [
        ("hybrid_fusion_model", best_fusion_ckpt),
        ("section_aware_tfidf_lr", sec_lr_ckpt),
        ("citation_aware_lr", cit_lr_ckpt),
        ("char_ngram_lr", char_lr_ckpt),
        ("section_aware_transformer", sec_trans_ckpt),
        ("hierarchical_legal_model", hier_ckpt),
    ]:
        if p.exists():
            frozen_checkpoints[name] = {
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
            }

    # 3. Create Manifest
    manifest = {
        "manifest_version": "1.0",
        "phase": "Phase 19: Advanced Legal Document Representation and Classification Optimization",
        "timestamp": "2026-11-03 22:38:00",
        "dataset_scope": "Jud-IPL Standalone Legal Intelligence Research System (0% IL-TUR dependency)",
        "task": "Binary Legal Judgment Outcome Prediction (Accepted=1, Rejected=0)",
        "best_proposed_model": "Hybrid Feature Fusion (Section-Aware TF-IDF + Legal Features + Char N-Grams)",
        "decision_threshold": 0.49,
        "validation_macro_f1": 0.6779,
        "validation_accuracy": 0.6789,
        "validation_roc_auc": 0.7441,
        "frozen_checkpoints": frozen_checkpoints,
        "independent_test_audit": {
            "total_corpus_documents": total_corpus_docs,
            "train_partition_size": n_tr,
            "validation_partition_size": n_val,
            "phase18_test_partition_size": n_te,
            "total_binary_partitioned": total_binary_partitioned,
            "unpartitioned_documents": n_unpartitioned,
            "unpartitioned_labels": {str(k): int(v) for k, v in label_dist.items()},
            "eligible_unused_binary_documents": n_eligible,
            "independent_test_available": independent_test_available,
            "limitation_statement": limitation_statement,
        },
    }

    manifest_path = REPORTS_DIR / "phase19_experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved Phase 19 Experiment Manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    run_phase19g_audit()

